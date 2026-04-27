#!/usr/bin/env python3
"""
Calibrate a binary anomaly threshold on an evaluation split.

The script sweeps all observed video scores and reports the best threshold under
three criteria:
  - accuracy
  - balanced accuracy
  - F1

Scores are computed in the same space as `predict.py`:
raw TRN scores -> temporal smoothing -> boundary refinement -> video max.
This keeps threshold calibration consistent with downstream prediction and
boundary analysis.
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
def collect_scores(
    model,
    dataset,
    device,
    smooth_k: int,
    boundary_weight: float,
) -> List[Dict[str, object]]:
    rows = []
    model.eval()
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
        rows.append({
            "filepath": item["filepath"],
            "label": int(item["label"].item()),
            "video_score": float(trn_refined.max()),
        })
    return rows


def compute_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (scores >= threshold).astype(np.int64)
    tp = int(np.sum((pred == 1) & (labels == 1)))
    tn = int(np.sum((pred == 0) & (labels == 0)))
    fp = int(np.sum((pred == 1) & (labels == 0)))
    fn = int(np.sum((pred == 0) & (labels == 1)))

    accuracy = float(np.mean(pred == labels))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    balanced_accuracy = 0.5 * (recall + specificity)
    neg_precision = tn / (tn + fn) if (tn + fn) else 0.0
    neg_recall = tn / (tn + fp) if (tn + fp) else 0.0
    neg_f1 = (
        2 * neg_precision * neg_recall / (neg_precision + neg_recall)
        if (neg_precision + neg_recall)
        else 0.0
    )
    macro_f1 = 0.5 * (f1 + neg_f1)

    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "balanced_accuracy": float(balanced_accuracy),
        "f1": float(f1),
        "macro_f1": float(macro_f1),
        "neg_f1": float(neg_f1),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def select_best(rows: List[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    labels = np.array([int(r["label"]) for r in rows], dtype=np.int64)
    scores = np.array([float(r["video_score"]) for r in rows], dtype=np.float32)
    thresholds = sorted(set(np.round(scores, 6).tolist()))
    thresholds = [thresholds[0] - 1e-6] + thresholds + [thresholds[-1] + 1e-6]

    best_acc = None
    best_bal = None
    best_f1 = None
    best_macro_f1 = None

    for threshold in thresholds:
        metrics = compute_metrics(labels, scores, float(threshold))
        if best_acc is None or (
            metrics["accuracy"], metrics["balanced_accuracy"], metrics["macro_f1"], metrics["f1"]
        ) > (
            best_acc["accuracy"], best_acc["balanced_accuracy"], best_acc["macro_f1"], best_acc["f1"]
        ):
            best_acc = metrics
        if best_bal is None or (
            metrics["balanced_accuracy"], metrics["macro_f1"], metrics["f1"], metrics["accuracy"]
        ) > (
            best_bal["balanced_accuracy"], best_bal["macro_f1"], best_bal["f1"], best_bal["accuracy"]
        ):
            best_bal = metrics
        if best_f1 is None or (
            metrics["f1"], metrics["macro_f1"], metrics["balanced_accuracy"], metrics["accuracy"]
        ) > (
            best_f1["f1"], best_f1["macro_f1"], best_f1["balanced_accuracy"], best_f1["accuracy"]
        ):
            best_f1 = metrics
        if best_macro_f1 is None or (
            metrics["macro_f1"], metrics["balanced_accuracy"], metrics["f1"], metrics["accuracy"]
        ) > (
            best_macro_f1["macro_f1"], best_macro_f1["balanced_accuracy"], best_macro_f1["f1"], best_macro_f1["accuracy"]
        ):
            best_macro_f1 = metrics

    return {
        "best_accuracy": best_acc,
        "best_balanced_accuracy": best_bal,
        "best_f1": best_f1,
        "best_macro_f1": best_macro_f1,
        "score_range": {
            "min": float(scores.min()),
            "max": float(scores.max()),
            "mean": float(scores.mean()),
            "median": float(np.median(scores)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate anomaly threshold on a split")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature_dir", required=True)
    parser.add_argument("--list_file", default="auto")
    parser.add_argument("--num_segments", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--val_split", action="store_true")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smooth_k", type=int, default=DEFAULT_SMOOTH_K)
    parser.add_argument("--boundary_weight", type=float, default=DEFAULT_BOUNDARY_W)
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Device: {device}")

    checkpoint = os.path.expanduser(args.checkpoint)
    feature_dir = os.path.expanduser(args.feature_dir)
    list_file = os.path.expanduser(args.list_file) if args.list_file != "auto" else "auto"

    ckpt = torch.load(checkpoint, map_location=device)
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

    if args.val_split:
        _, samples = make_train_val_split(
            list_file,
            feature_dir,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
        dataset = InMemoryDataset(samples, num_segments=args.num_segments)
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

    print(f"Running threshold sweep on {len(dataset)} videos ({split_name})...")
    rows = collect_scores(
        model,
        dataset,
        device,
        smooth_k=args.smooth_k,
        boundary_weight=args.boundary_weight,
    )
    report = select_best(rows)
    report.update({
        "checkpoint": os.path.abspath(checkpoint),
        "feature_dir": os.path.abspath(feature_dir),
        "split": split_name,
        "num_segments": int(args.num_segments),
        "smooth_k": int(args.smooth_k),
        "boundary_weight": float(args.boundary_weight),
        "score_space": "smoothed_boundary_refined_trn",
        "n_videos": len(rows),
    })

    print("\n=== Threshold Calibration ===")
    for key in ("best_accuracy", "best_balanced_accuracy", "best_f1", "best_macro_f1"):
        metrics = report[key]
        print(
            f"{key:24s}: threshold={metrics['threshold']:.6f} | "
            f"acc={metrics['accuracy']:.4f} | "
            f"bal_acc={metrics['balanced_accuracy']:.4f} | "
            f"f1={metrics['f1']:.4f} | "
            f"macro_f1={metrics['macro_f1']:.4f} | "
            f"TP={metrics['tp']} TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']}"
        )

    if args.output_json:
        output_json = os.path.abspath(os.path.expanduser(args.output_json))
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved JSON report to {output_json}")


if __name__ == "__main__":
    main()
