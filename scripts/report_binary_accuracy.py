#!/usr/bin/env python3
"""
Threshold-based binary classification report for a trained checkpoint.

This complements AUC/AP with a concrete operating-point analysis:
  - accuracy
  - precision / recall / F1
  - specificity
  - false positive rate / false negative rate
  - confusion matrix

Scores are computed in the same refined-score space as `predict.py`
(temporal smoothing + boundary refinement) so that the reported thresholded
metrics match the actual prediction pipeline.

Typical usage on the UCF val split:
    python scripts/report_binary_accuracy.py \
        --checkpoint runs/step1_progressive_s42/checkpoints/best.pt \
        --feature_dir /path/to/UCF_train_i3d \
        --list_file auto \
        --val_split \
        --num_segments 32 \
        --threshold 0.6843 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from dataset import InMemoryDataset, UCFCrimeDataset, make_train_val_split  # noqa: E402
from evaluate import infer_input_dim_from_checkpoint, resolve_device  # noqa: E402
from model import ViolenceDetector  # noqa: E402
from predict import (  # noqa: E402
    DEFAULT_BOUNDARY_W,
    DEFAULT_SMOOTH_K,
    boundary_refine_scores,
    smooth_scores,
)


def _pad_boundary(boundary: np.ndarray, target_len: int) -> np.ndarray:
    if boundary.size == 0:
        return np.zeros(target_len, dtype=np.float32)
    if len(boundary) >= target_len:
        return boundary[:target_len].astype(np.float32, copy=False)
    return np.append(boundary, np.repeat(boundary[-1], target_len - len(boundary))).astype(
        np.float32
    )


@torch.no_grad()
def collect_video_scores(
    model,
    dataset,
    device,
    smooth_k: int,
    boundary_weight: float,
) -> List[Dict[str, object]]:
    model.eval()
    rows = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        feat = item["features"].unsqueeze(0).to(device)
        out = model(feat)
        trn_scores = out["trn_scores"].squeeze(0).detach().cpu().numpy().astype(np.float32)
        boundary = _pad_boundary(
            out["boundary"].squeeze(0).detach().cpu().numpy().astype(np.float32),
            len(trn_scores),
        )
        trn_smooth = smooth_scores(trn_scores, k=smooth_k).astype(np.float32, copy=False)
        trn_refined = boundary_refine_scores(
            trn_smooth,
            boundary,
            weight=boundary_weight,
        ).astype(np.float32, copy=False)
        video_score = float(trn_refined.max())
        rows.append({
            "filepath": item["filepath"],
            "label": int(item["label"].item()),
            "cat_id": int(item["cat_id"].item()),
            "video_score": video_score,
        })
    return rows


def compute_binary_report(rows: List[Dict[str, object]], threshold: float) -> Dict[str, object]:
    y_true = np.array([int(r["label"]) for r in rows], dtype=np.int64)
    scores = np.array([float(r["video_score"]) for r in rows], dtype=np.float32)
    y_pred = (scores >= threshold).astype(np.int64)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    n = int(len(rows))

    accuracy = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    balanced_acc = 0.5 * (recall + specificity)

    return {
        "threshold": float(threshold),
        "n_videos": n,
        "n_anomalous": int(np.sum(y_true == 1)),
        "n_normal": int(np.sum(y_true == 0)),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_acc),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "confusion_matrix": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Report threshold-based binary accuracy")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature_dir", required=True)
    parser.add_argument("--list_file", default="auto")
    parser.add_argument("--num_segments", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.6843)
    parser.add_argument("--smooth_k", type=int, default=DEFAULT_SMOOTH_K)
    parser.add_argument("--boundary_weight", type=float, default=DEFAULT_BOUNDARY_W)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--val_split", action="store_true",
                        help="Evaluate on a stratified validation split created from feature_dir/list_file")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Device: {device}")

    ckpt = torch.load(os.path.expanduser(args.checkpoint), map_location=device)
    input_dim = infer_input_dim_from_checkpoint(ckpt)
    model_args = ckpt.get("args", {})
    model = ViolenceDetector(
        input_dim=input_dim,
        num_classes=7,
        d_model=model_args.get("d_model", 512),
        nhead=model_args.get("nhead", 8),
        trn_layers=model_args.get("trn_layers", 2),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    feature_dir = os.path.expanduser(args.feature_dir)
    list_file = os.path.expanduser(args.list_file) if args.list_file != "auto" else "auto"

    if args.val_split:
        _, val_samples = make_train_val_split(
            list_file,
            feature_dir,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        dataset = InMemoryDataset(val_samples, num_segments=args.num_segments)
        split_name = "val_split"
    else:
        dataset = UCFCrimeDataset(
            feature_dir,
            list_file,
            mode="test",
            num_segments=args.num_segments,
            violence_only=False,
        )
        split_name = "dataset"

    print(f"Running binary evaluation on {len(dataset)} videos ({split_name})...")
    rows = collect_video_scores(
        model,
        dataset,
        device,
        smooth_k=args.smooth_k,
        boundary_weight=args.boundary_weight,
    )
    report = compute_binary_report(rows, args.threshold)
    report.update({
        "checkpoint": os.path.abspath(os.path.expanduser(args.checkpoint)),
        "feature_dir": os.path.abspath(feature_dir),
        "list_file": list_file,
        "split": split_name,
        "num_segments": int(args.num_segments),
        "smooth_k": int(args.smooth_k),
        "boundary_weight": float(args.boundary_weight),
        "score_space": "smoothed_boundary_refined_trn",
    })

    cm = report["confusion_matrix"]
    print("\n=== Binary Accuracy Report ===")
    print(f"Split                   : {split_name}")
    print(f"Threshold               : {report['threshold']:.3f}")
    print(f"Videos                  : {report['n_videos']} "
          f"({report['n_anomalous']} anomalous, {report['n_normal']} normal)")
    print(f"Accuracy                : {report['accuracy']:.4f}")
    print(f"Balanced accuracy       : {report['balanced_accuracy']:.4f}")
    print(f"Precision / Recall / F1 : {report['precision']:.4f} / "
          f"{report['recall']:.4f} / {report['f1']:.4f}")
    print(f"Specificity             : {report['specificity']:.4f}")
    print(f"FPR / FNR               : {report['false_positive_rate']:.4f} / "
          f"{report['false_negative_rate']:.4f}")
    print(f"Confusion matrix        : TP={cm['tp']} TN={cm['tn']} FP={cm['fp']} FN={cm['fn']}")

    if args.output_json:
        output_json = os.path.abspath(os.path.expanduser(args.output_json))
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved JSON report to {output_json}")


if __name__ == "__main__":
    main()
