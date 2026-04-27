#!/usr/bin/env python3
"""
Analyze false-alarm tendency on a directory of normal-only feature files.

This is intended as a supplementary diagnostic:
  - How high are anomaly scores on normal videos?
  - How often would the model falsely flag a normal video as anomalous?
  - Which normal videos are the strongest false alarms?

Example:
    python scripts/analyze_normal_false_alarms.py \
        --checkpoint runs/step1_progressive_s42/checkpoints/best.pt \
        --feature_dir /path/to/normal/features \
        --num_segments 32 \
        --device mps \
        --output_json runs/shanghai_false_alarm_summary.json \
        --output_csv runs/shanghai_false_alarm_scores.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from feature_utils import scan_feature_files  # noqa: E402
from predict import (  # noqa: E402
    DEFAULT_BOUNDARY_W,
    DEFAULT_MERGE_GAP,
    DEFAULT_MIN_DUR,
    DEFAULT_SMOOTH_K,
    DEFAULT_THRESHOLD,
    build_model,
    predict,
)


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

    raise ValueError(f"Unsupported device: {requested}")


def summarize_scores(scores: np.ndarray) -> Dict[str, float]:
    return {
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "std": float(np.std(scores)),
        "p90": float(np.percentile(scores, 90)),
        "p95": float(np.percentile(scores, 95)),
        "p99": float(np.percentile(scores, 99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze normal-only false alarms from a trained checkpoint"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature_dir", required=True)
    parser.add_argument("--num_segments", type=int, default=32)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--frames_per_segment", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--smooth_k", type=int, default=DEFAULT_SMOOTH_K)
    parser.add_argument("--min_duration", type=float, default=DEFAULT_MIN_DUR)
    parser.add_argument("--merge_gap", type=float, default=DEFAULT_MERGE_GAP)
    parser.add_argument("--boundary_weight", type=float, default=DEFAULT_BOUNDARY_W)
    parser.add_argument("--no_snap", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional cap for quick smoke tests")
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--output_csv", default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"Device: {device}")

    feature_paths = scan_feature_files(os.path.expanduser(args.feature_dir))
    if args.limit is not None:
        feature_paths = feature_paths[: args.limit]
    if not feature_paths:
        raise FileNotFoundError(f"No .npy files found under {args.feature_dir}")

    print(f"Found {len(feature_paths)} normal feature files")
    model = build_model(os.path.expanduser(args.checkpoint), device)

    results: List[Dict[str, Any]] = []
    for idx, path in enumerate(feature_paths, start=1):
        rel = os.path.relpath(path, os.path.expanduser(args.feature_dir))
        print(f"[{idx}/{len(feature_paths)}] {rel}", flush=True)
        pred = predict(
            model,
            path,
            device,
            num_segments=args.num_segments,
            fps=args.fps,
            frames_per_seg=args.frames_per_segment,
            threshold=args.threshold,
            smooth_k=args.smooth_k,
            min_duration=args.min_duration,
            merge_gap=args.merge_gap,
            boundary_weight=args.boundary_weight,
            snap_boundaries=not args.no_snap,
        )
        events = pred.get("events", [])
        results.append({
            "filepath": path,
            "relative_path": rel,
            "video_score": float(pred["video_score"]),
            "video_label": int(pred["video_label"]),
            "n_events": len(events),
            "pred_categories": sorted({ev["category"] for ev in events}),
            "max_event_confidence": float(max((ev["confidence"] for ev in events), default=0.0)),
        })

    scores = np.array([r["video_score"] for r in results], dtype=np.float32)
    false_alarm_count = int(sum(r["video_label"] == 1 for r in results))
    event_count = int(sum(r["n_events"] > 0 for r in results))
    top_results = sorted(results, key=lambda x: x["video_score"], reverse=True)[: args.top_k]

    summary = {
        "checkpoint": os.path.abspath(os.path.expanduser(args.checkpoint)),
        "feature_dir": os.path.abspath(os.path.expanduser(args.feature_dir)),
        "n_videos": len(results),
        "threshold": float(args.threshold),
        "num_segments": int(args.num_segments),
        "score_stats": summarize_scores(scores),
        "false_alarm_count": false_alarm_count,
        "false_alarm_rate": float(false_alarm_count / len(results)),
        "videos_with_detected_events": event_count,
        "event_detection_rate": float(event_count / len(results)),
        "top_false_alarms": top_results,
    }

    print("\n=== Normal-Only False Alarm Summary ===")
    print(f"Videos                  : {summary['n_videos']}")
    print(f"Threshold               : {summary['threshold']:.3f}")
    print(f"False alarms            : {summary['false_alarm_count']} "
          f"({summary['false_alarm_rate']:.2%})")
    print(f"Videos with events      : {summary['videos_with_detected_events']} "
          f"({summary['event_detection_rate']:.2%})")
    stats = summary["score_stats"]
    print(f"Score mean / median     : {stats['mean']:.4f} / {stats['median']:.4f}")
    print(f"Score p95 / p99 / max   : {stats['p95']:.4f} / {stats['p99']:.4f} / {stats['max']:.4f}")

    print(f"\nTop {len(top_results)} highest-scoring normal videos:")
    for rank, item in enumerate(top_results, start=1):
        cats = ",".join(item["pred_categories"]) if item["pred_categories"] else "-"
        print(
            f"  {rank:2d}. {item['relative_path']} | "
            f"score={item['video_score']:.4f} | "
            f"pred={'ANOM' if item['video_label'] else 'norm'} | "
            f"events={item['n_events']} | cats={cats}"
        )

    if args.output_json:
        output_json = os.path.abspath(os.path.expanduser(args.output_json))
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary JSON saved to {output_json}")

    if args.output_csv:
        output_csv = os.path.abspath(os.path.expanduser(args.output_csv))
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "relative_path",
                    "video_score",
                    "video_label",
                    "n_events",
                    "pred_categories",
                    "max_event_confidence",
                    "filepath",
                ],
            )
            writer.writeheader()
            for row in sorted(results, key=lambda x: x["video_score"], reverse=True):
                writer.writerow({
                    **row,
                    "pred_categories": ",".join(row["pred_categories"]),
                })
        print(f"Per-video CSV saved to {output_csv}")


if __name__ == "__main__":
    main()
