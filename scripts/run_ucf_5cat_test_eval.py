#!/usr/bin/env python3
"""
Run evaluation on the local UCF 5-category + Normal test subset.

This script is strict on purpose:
  - verifies the expected local test-feature counts before evaluation
  - accepts either checkpoint files or run directories
  - resolves device automatically (cuda -> mps -> cpu)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from evaluate import evaluate_checkpoint  # noqa: E402
from feature_utils import scan_feature_files  # noqa: E402
from dataset import get_category_from_filename  # noqa: E402


EXPECTED_TEST_COUNTS: Dict[str, int] = {
    "Abuse": 2,
    "Explosion": 21,
    "Fighting": 5,
    "Robbery": 5,
    "Shooting": 23,
    "Normal": 150,
}


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


def resolve_checkpoint(path_str: str) -> Path:
    path = Path(path_str).expanduser().resolve()
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"checkpoint path not found: {path}")

    candidates = [
        path / "checkpoints" / "best.pt",
        path / "best.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Could not find best checkpoint under {path}. "
        "Expected either <run_dir>/checkpoints/best.pt or <run_dir>/best.pt."
    )


def count_categories(feature_dir: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for feature_path in scan_feature_files(str(feature_dir)):
        category = get_category_from_filename(feature_path)
        counts[category] = counts.get(category, 0) + 1
    return counts


def validate_test_subset(feature_dir: Path) -> Dict[str, int]:
    counts = count_categories(feature_dir)
    problems = []

    for category, expected in EXPECTED_TEST_COUNTS.items():
        actual = counts.get(category, 0)
        if actual != expected:
            problems.append(f"{category}: expected {expected}, found {actual}")

    unexpected = sorted(cat for cat in counts if cat not in EXPECTED_TEST_COUNTS)
    if unexpected:
        problems.append(f"unexpected categories present: {unexpected}")

    if problems:
        details = "\n".join(f"  - {item}" for item in problems)
        raise RuntimeError(
            "UCF 5cat+Normal test subset is incomplete or inconsistent.\n"
            f"{details}\n"
            f"Checked feature dir: {feature_dir}"
        )

    return counts


def compact_metrics(metrics: Dict[str, object]) -> Dict[str, float]:
    return {
        "video_auc": float(metrics["video_auc"]),
        "video_ap": float(metrics["video_ap"]),
        "frame_auc": float(metrics["frame_auc"]),
        "frame_ap": float(metrics["frame_ap"]),
    }


def run_one(
    label: str,
    checkpoint: Path,
    feature_dir: Path,
    device: torch.device,
    num_segments: int,
    frames_per_segment: int,
) -> Dict[str, object]:
    print(f"\n[{label}] checkpoint: {checkpoint}")
    metrics = evaluate_checkpoint(
        checkpoint_path=str(checkpoint),
        feature_dir=str(feature_dir),
        list_file="auto",
        device=device,
        frames_per_segment=frames_per_segment,
        num_segments=num_segments,
        use_val_split=False,
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate progressive/joint checkpoints on the local UCF 5cat+Normal test subset"
    )
    parser.add_argument("--progressive", required=True, help="Path to progressive checkpoint or run dir")
    parser.add_argument("--joint", required=True, help="Path to joint checkpoint or run dir")
    parser.add_argument(
        "--feature_dir",
        default="/Users/liuyuxiang/Documents/USA/cs585/project/UCF/UCF_split_test_i3d_5cat",
        help="Root directory that contains the 5-category test features plus features/Normal",
    )
    parser.add_argument("--num_segments", type=int, default=32)
    parser.add_argument("--frames_per_segment", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--output_json", default=None, help="Optional path to save metrics as JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_dir = Path(args.feature_dir).expanduser().resolve()
    if not feature_dir.exists():
        raise FileNotFoundError(f"feature_dir not found: {feature_dir}")

    counts = validate_test_subset(feature_dir)
    progressive_ckpt = resolve_checkpoint(args.progressive)
    joint_ckpt = resolve_checkpoint(args.joint)
    device = resolve_device(args.device)

    print("Verified test subset counts:")
    for category in sorted(counts):
        print(f"  {category}: {counts[category]}")
    print(f"Device: {device}")

    progressive_metrics = run_one(
        "progressive",
        progressive_ckpt,
        feature_dir,
        device,
        num_segments=args.num_segments,
        frames_per_segment=args.frames_per_segment,
    )
    joint_metrics = run_one(
        "joint",
        joint_ckpt,
        feature_dir,
        device,
        num_segments=args.num_segments,
        frames_per_segment=args.frames_per_segment,
    )

    summary = {
        "feature_dir": str(feature_dir),
        "counts": counts,
        "progressive": {
            "checkpoint": str(progressive_ckpt),
            **compact_metrics(progressive_metrics),
        },
        "joint": {
            "checkpoint": str(joint_ckpt),
            **compact_metrics(joint_metrics),
        },
    }

    print("\nSummary:")
    print(
        f"  progressive  vAUC={summary['progressive']['video_auc']:.4f} "
        f"vAP={summary['progressive']['video_ap']:.4f} "
        f"fAUC={summary['progressive']['frame_auc']:.4f} "
        f"fAP={summary['progressive']['frame_ap']:.4f}"
    )
    print(
        f"  joint        vAUC={summary['joint']['video_auc']:.4f} "
        f"vAP={summary['joint']['video_ap']:.4f} "
        f"fAUC={summary['joint']['frame_auc']:.4f} "
        f"fAP={summary['joint']['frame_ap']:.4f}"
    )

    if args.output_json:
        out_path = Path(args.output_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved JSON: {out_path}")


if __name__ == "__main__":
    main()
