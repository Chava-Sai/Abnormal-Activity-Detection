#!/usr/bin/env python3
"""
Error Taxonomy Analysis for Violence Event Detection.

This script runs video-level inference on an explicit split list or on an
internally generated validation split, extracts false positives / false
negatives at a chosen threshold, and groups them into interpretable failure
modes with frequency counts.

Typical usage:
  - Validation split reproduction:
      python src/error_taxonomy.py --val_split ...
  - Expanded taxonomy on the 877-video training split:
      python src/error_taxonomy.py \
        --list_file list/ucf-i3d-train-split.list \
        --feature_dir ... \
        --threshold 0.6606
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import InMemoryDataset, UCFCrimeDataset, make_train_val_split
from model import ViolenceDetector


CAT_ID_TO_NAME = {
    0: "Normal",
    1: "Abuse",
    2: "Fighting",
    3: "Shooting",
    4: "Explosion",
    5: "Robbery",
    6: "Riot",
}


def infer_input_dim_from_checkpoint(ckpt: dict, fallback: int = 2048) -> int:
    model_args = ckpt.get("args", {})
    if model_args.get("input_dim") is not None:
        return int(model_args["input_dim"])

    weight = ckpt.get("state_dict", {}).get("mil_scorer.net.0.weight")
    if weight is not None:
        return int(weight.shape[1])
    return fallback


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("Requested --device cuda but CUDA is not available")
        return torch.device("cuda")

    if requested == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise ValueError("Requested --device mps but MPS is not available")
        return torch.device("mps")

    if requested == "cpu":
        return torch.device("cpu")

    raise ValueError(f"unsupported device: {requested}")


def feature_path_to_video_name(filepath: str) -> str:
    basename = os.path.basename(filepath)
    if basename.endswith("_i3d.npy"):
        return basename[:-8] + ".mp4"
    if basename.endswith(".npy"):
        return basename[:-4] + ".mp4"
    return basename


def classify_fp_mode(
    trn_scores: np.ndarray,
    mil_scores: np.ndarray,
    _cat_name: str,
    _video_name: str,
    pred_cls: str,
    threshold: float,
) -> tuple[str, str]:
    max_trn = float(np.max(trn_scores))
    mean_trn = float(np.mean(trn_scores))
    max_mil = float(np.max(mil_scores))
    mean_mil = float(np.mean(mil_scores))
    std_trn = float(np.std(trn_scores))
    margin = max_trn - float(threshold)
    high_thr = int(np.sum(trn_scores >= threshold))

    if max_mil < 0.01 and mean_mil < 0.001 and mean_trn > 0.55 and std_trn < 0.05:
        return "FP_TRN_COLLAPSE", "TRN stays uniformly high while MIL is nearly zero"

    if margin < 0.002:
        return "FP_THRESHOLD_MARGIN", "Video score is only slightly above the calibrated threshold"

    if max_mil < 0.05:
        if pred_cls == "Robbery":
            return "FP_ROBBERY_CLASS_COLLAPSE", "Weak evidence, but the classifier collapses to Robbery"
        return "FP_WEAK_MIL_LEAK", "MIL evidence stays weak while TRN crosses threshold"

    if max_mil < 0.5:
        return "FP_MIXED_SUPPORT", "MIL shows partial support, but anomaly evidence remains ambiguous"

    if high_thr >= max(1, int(len(trn_scores) * 0.9)):
        return "FP_STRONG_OVERFIRE", "Both MIL and TRN remain elevated across most segments"

    return "FP_STRONG_MIL_OVERFIRE", "MIL and TRN both fire on a normal video"


def classify_fn_mode(
    trn_scores: np.ndarray,
    mil_scores: np.ndarray,
    cat_name: str,
    _video_name: str,
    pred_cls: str,
    threshold: float,
) -> tuple[str, str]:
    max_trn = float(np.max(trn_scores))
    max_mil = float(np.max(mil_scores))
    mean_mil = float(np.mean(mil_scores))
    margin = float(threshold) - max_trn

    if pred_cls == "Robbery" and cat_name != "Robbery":
        return "FN_ROBBERY_CLASS_COLLAPSE", "Non-Robbery anomaly is pulled toward the dominant Robbery label"

    if margin < 0.002:
        return "FN_THRESHOLD_MARGIN", "Video score is only slightly below the calibrated threshold"

    if mean_mil > 0.5 or max_mil > 0.99:
        return "FN_TRN_SUPPRESSED_STRONG_MIL", "MIL is strong, but TRN keeps the final score below threshold"

    if mean_mil > 0.2:
        return "FN_TRN_SUPPRESSED_MODERATE_MIL", "MIL provides moderate support, but TRN still suppresses detection"

    return "FN_LOW_MIL_SUPPORT", "Even MIL support is weak, suggesting poor transfer for this anomaly pattern"


def build_dataset(
    feature_dir: str,
    list_file: str,
    num_segments: int,
    val_split: bool,
    val_ratio: float,
    seed: int,
) -> tuple[torch.utils.data.Dataset, str]:
    if val_split:
        _, samples = make_train_val_split(
            list_file,
            feature_dir,
            val_ratio=val_ratio,
            seed=seed,
        )
        dataset = InMemoryDataset(samples, num_segments=num_segments)
        split_name = "val_split"
    else:
        dataset = UCFCrimeDataset(
            feature_dir,
            list_file,
            mode="test",
            num_segments=num_segments,
            violence_only=False,
        )
        split_name = f"direct:{Path(list_file).name if list_file != 'auto' else 'auto'}"
    return dataset, split_name


def load_model(checkpoint_path: str, device: torch.device) -> tuple[ViolenceDetector, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_args = ckpt.get("args", {})
    model = ViolenceDetector(
        input_dim=infer_input_dim_from_checkpoint(ckpt),
        num_classes=7,
        d_model=model_args.get("d_model", 512),
        nhead=model_args.get("nhead", 8),
        trn_layers=model_args.get("trn_layers", 2),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def run_inference(model: ViolenceDetector, dataset, device: torch.device, threshold: float, log_every: int) -> List[dict]:
    results: List[dict] = []
    model.eval()

    for idx in range(len(dataset)):
        item = dataset[idx]
        feat = item["features"].unsqueeze(0).to(device)
        label = int(item["label"].item())
        cat_id = int(item["cat_id"].item())
        filepath = item["filepath"]

        out = model(feat)
        trn_scores = out["trn_scores"].squeeze(0).detach().cpu().numpy().astype(np.float32)
        mil_scores = out["mil_scores"].squeeze(0).detach().cpu().numpy().astype(np.float32)
        cls_logits = out["cls_logits"].squeeze(0).detach().cpu().numpy().astype(np.float32)

        video_score = float(np.max(trn_scores))
        predicted = 1 if video_score >= threshold else 0
        pred_cls = int(np.argmax(cls_logits.mean(axis=0)))

        results.append({
            "idx": idx,
            "filepath": filepath,
            "video_name": feature_path_to_video_name(filepath),
            "cat_id": cat_id,
            "cat_name": CAT_ID_TO_NAME.get(cat_id, f"cat_{cat_id}"),
            "label": label,
            "predicted": predicted,
            "vid_score": video_score,
            "trn_scores": trn_scores,
            "mil_scores": mil_scores,
            "pred_cls": CAT_ID_TO_NAME.get(pred_cls, f"cls_{pred_cls}"),
        })

        if log_every > 0 and (idx + 1) % log_every == 0:
            print(f"  Processed {idx + 1}/{len(dataset)}")

    return results


def plot_error_cases(cases: Sequence[dict], error_type: str, output_path: str, max_cases: int) -> None:
    if not cases:
        print(f"  No {error_type} cases found")
        return

    if error_type == "FP":
        chosen = sorted(cases, key=lambda row: -row["vid_score"])[:max_cases]
        title = f"Top-{len(chosen)} False Positives"
    else:
        chosen = sorted(cases, key=lambda row: row["vid_score"])[:max_cases]
        title = f"Top-{len(chosen)} False Negatives"

    ncols = 5
    nrows = (len(chosen) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.6, max(1, nrows) * 2.5))
    axes = np.array(axes).reshape(-1)

    for idx, row in enumerate(chosen):
        ax = axes[idx]
        t = np.arange(len(row["trn_scores"]))
        ax.plot(t, row["trn_scores"], color="#CC2200", linewidth=1.5, label="TRN")
        ax.plot(t, row["mil_scores"], color="#2266CC", linewidth=1.0, linestyle="--", alpha=0.7, label="MIL")
        ax.axhline(0.5, color="gray", linewidth=0.8, linestyle=":", alpha=0.8)
        ax.fill_between(t, row["trn_scores"], alpha=0.1, color="#CC2200")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, max(0, len(t) - 1))
        ax.set_title(
            f"{row['cat_name']} [{row['mode_code']}]\nscore={row['vid_score']:.3f}",
            fontsize=7,
            fontweight="bold",
        )
        ax.tick_params(labelsize=6)
        if idx == 0:
            ax.legend(fontsize=5, loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for idx in range(len(chosen), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def save_csv(fp_cases: Sequence[dict], fn_cases: Sequence[dict], output_path: str) -> None:
    fieldnames = [
        "error_type",
        "video_name",
        "filepath",
        "cat_name",
        "label",
        "predicted",
        "vid_score",
        "mode_code",
        "mode_desc",
        "pred_cls",
        "idx",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in list(fp_cases) + list(fn_cases):
            writer.writerow({key: row.get(key) for key in fieldnames})
    print(f"  Saved: {output_path}")


def write_summary(
    fp_cases: Sequence[dict],
    fn_cases: Sequence[dict],
    output_path: str,
    threshold: float,
    total_videos: int,
    split_name: str,
) -> None:
    fp_by_mode: Dict[str, List[dict]] = defaultdict(list)
    fn_by_mode: Dict[str, List[dict]] = defaultdict(list)
    fp_by_cat: Dict[str, int] = defaultdict(int)
    fn_by_cat: Dict[str, int] = defaultdict(int)
    fp_by_pred_cls: Dict[str, int] = defaultdict(int)
    fn_by_pred_cls: Dict[str, int] = defaultdict(int)

    for row in fp_cases:
        fp_by_mode[row["mode_code"]].append(row)
        fp_by_cat[row["cat_name"]] += 1
        fp_by_pred_cls[row["pred_cls"]] += 1
    for row in fn_cases:
        fn_by_mode[row["mode_code"]].append(row)
        fn_by_cat[row["cat_name"]] += 1
        fn_by_pred_cls[row["pred_cls"]] += 1

    lines = [
        "=" * 70,
        "ERROR TAXONOMY REPORT",
        "=" * 70,
        f"Split            : {split_name}",
        f"Threshold        : {threshold}",
        f"Total videos     : {total_videos}",
        f"False Positives  : {len(fp_cases)}",
        f"False Negatives  : {len(fn_cases)}",
        "",
        "-" * 70,
        "FALSE POSITIVE BREAKDOWN",
        "-" * 70,
    ]

    for mode, rows in sorted(fp_by_mode.items(), key=lambda item: (-len(item[1]), item[0])):
        lines.append(f"  {mode:<22s} n={len(rows):3d}  {rows[0]['mode_desc']}")
    if not fp_by_mode:
        lines.append("  none")

    lines.extend([
        "",
        "FP by category:",
    ])
    for cat_name, count in sorted(fp_by_cat.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"  {cat_name:<15s} {count}")
    if not fp_by_cat:
        lines.append("  none")

    lines.extend([
        "",
        "FP predicted class breakdown:",
    ])
    for pred_cls, count in sorted(fp_by_pred_cls.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"  {pred_cls:<15s} {count}")
    if not fp_by_pred_cls:
        lines.append("  none")

    lines.extend([
        "",
        "-" * 70,
        "FALSE NEGATIVE BREAKDOWN",
        "-" * 70,
    ])
    for mode, rows in sorted(fn_by_mode.items(), key=lambda item: (-len(item[1]), item[0])):
        lines.append(f"  {mode:<22s} n={len(rows):3d}  {rows[0]['mode_desc']}")
    if not fn_by_mode:
        lines.append("  none")

    lines.extend([
        "",
        "FN by category:",
    ])
    for cat_name, count in sorted(fn_by_cat.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"  {cat_name:<15s} {count}")
    if not fn_by_cat:
        lines.append("  none")

    lines.extend([
        "",
        "FN predicted class breakdown:",
    ])
    for pred_cls, count in sorted(fn_by_pred_cls.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"  {pred_cls:<15s} {count}")
    if not fn_by_pred_cls:
        lines.append("  none")

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"  Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run error taxonomy analysis")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature_dir", required=True)
    parser.add_argument("--list_file", default="auto")
    parser.add_argument("--output_dir", default="./visualizations/error_taxonomy")
    parser.add_argument("--num_segments", type=int, default=None,
                        help="Override num_segments; otherwise infer from checkpoint args")
    parser.add_argument("--threshold", type=float, default=0.6606)
    parser.add_argument("--val_split", action="store_true",
                        help="Regenerate the validation split from the full training list")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_cases", type=int, default=100,
                        help="How many worst FP/FN timelines to visualize")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--log_every", type=int, default=50)
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Device: {device}")

    checkpoint = os.path.abspath(os.path.expanduser(args.checkpoint))
    feature_dir = os.path.abspath(os.path.expanduser(args.feature_dir))
    list_file = os.path.expanduser(args.list_file) if args.list_file != "auto" else "auto"
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))
    os.makedirs(output_dir, exist_ok=True)

    model, ckpt = load_model(checkpoint, device)
    ckpt_args = ckpt.get("args", {})
    num_segments = int(args.num_segments or ckpt_args.get("num_segments", 32))

    dataset, split_name = build_dataset(
        feature_dir=feature_dir,
        list_file=list_file,
        num_segments=num_segments,
        val_split=args.val_split,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print(
        f"Loaded checkpoint: epoch={ckpt.get('epoch', '?')} "
        f"vAUC={ckpt.get('metrics', {}).get('video_auc', float('nan'))} "
        f"T={num_segments}"
    )
    print(f"Dataset split: {split_name}")
    print(f"Videos: {len(dataset)}")
    print(f"Running inference with threshold={args.threshold:.6f}")

    results = run_inference(model, dataset, device, args.threshold, args.log_every)

    tp_cases: List[dict] = []
    tn_cases: List[dict] = []
    fp_cases: List[dict] = []
    fn_cases: List[dict] = []

    for row in results:
        gt = row["label"]
        pred = row["predicted"]
        if gt == 0 and pred == 1:
            mode_code, mode_desc = classify_fp_mode(
                row["trn_scores"],
                row["mil_scores"],
                row["cat_name"],
                row["video_name"],
                row["pred_cls"],
                args.threshold,
            )
            row["error_type"] = "FP"
            row["mode_code"] = mode_code
            row["mode_desc"] = mode_desc
            fp_cases.append(row)
        elif gt == 1 and pred == 0:
            mode_code, mode_desc = classify_fn_mode(
                row["trn_scores"],
                row["mil_scores"],
                row["cat_name"],
                row["video_name"],
                row["pred_cls"],
                args.threshold,
            )
            row["error_type"] = "FN"
            row["mode_code"] = mode_code
            row["mode_desc"] = mode_desc
            fn_cases.append(row)
        elif gt == 1 and pred == 1:
            row["error_type"] = "TP"
            row["mode_code"] = "TP"
            row["mode_desc"] = "Correct detection"
            tp_cases.append(row)
        else:
            row["error_type"] = "TN"
            row["mode_code"] = "TN"
            row["mode_desc"] = "Correct rejection"
            tn_cases.append(row)

    print(f"\nConfusion matrix (threshold={args.threshold:.6f}):")
    print(f"  TP={len(tp_cases)}  TN={len(tn_cases)}  FP={len(fp_cases)}  FN={len(fn_cases)}")
    if tp_cases or fp_cases:
        precision = len(tp_cases) / (len(tp_cases) + len(fp_cases))
        print(f"  Precision = {precision:.4f}")
    if tp_cases or fn_cases:
        recall = len(tp_cases) / (len(tp_cases) + len(fn_cases))
        print(f"  Recall    = {recall:.4f}")

    csv_path = os.path.join(output_dir, "error_taxonomy.csv")
    summary_path = os.path.join(output_dir, "error_summary.txt")
    save_csv(fp_cases, fn_cases, csv_path)
    write_summary(fp_cases, fn_cases, summary_path, args.threshold, len(results), split_name)

    print("\nPlotting worst FP/FN timelines...")
    plot_error_cases(fp_cases, "FP", os.path.join(output_dir, "fp_cases.png"), args.max_cases)
    plot_error_cases(fn_cases, "FN", os.path.join(output_dir, "fn_cases.png"), args.max_cases)

    print("\nDone")
    print(f"  CSV     : {csv_path}")
    print(f"  Summary : {summary_path}")


if __name__ == "__main__":
    main()
