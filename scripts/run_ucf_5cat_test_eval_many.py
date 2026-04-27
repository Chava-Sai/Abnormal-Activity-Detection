#!/usr/bin/env python3
"""
Evaluate multiple checkpoints on the local UCF 5-category + Normal test subset.

Usage:
  python run_ucf_5cat_test_eval_many.py \
    --item seq8 /path/to/run_or_best.pt 8 \
    --item seq16 /path/to/run_or_best.pt 16 \
    --item seq32 /path/to/run_or_best.pt 32
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate multiple checkpoints on the local UCF 5cat+Normal test subset"
    )
    parser.add_argument(
        "--item",
        nargs=3,
        action="append",
        metavar=("LABEL", "CHECKPOINT_OR_RUN", "NUM_SEGMENTS"),
        required=True,
        help="Repeatable triplet, for example: --item seq8 /path/to/run 8",
    )
    parser.add_argument(
        "--feature_dir",
        default="/Users/liuyuxiang/Documents/USA/cs585/project/UCF/UCF_split_test_i3d_5cat",
        help="Root directory that contains the 5-category test features plus features/Normal",
    )
    parser.add_argument("--frames_per_segment", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_dir = Path(args.feature_dir).expanduser().resolve()
    if not feature_dir.exists():
        raise FileNotFoundError(f"feature_dir not found: {feature_dir}")

    counts = validate_test_subset(feature_dir)
    device = resolve_device(args.device)

    print("Verified test subset counts:")
    for category in sorted(counts):
        print(f"  {category}: {counts[category]}")
    print(f"Device: {device}")

    summary_items: List[dict] = []
    for label, raw_path, raw_segments in args.item:
        checkpoint = resolve_checkpoint(raw_path)
        num_segments = int(raw_segments)
        print(f"\n[{label}] checkpoint: {checkpoint}")
        metrics = evaluate_checkpoint(
            checkpoint_path=str(checkpoint),
            feature_dir=str(feature_dir),
            list_file="auto",
            device=device,
            frames_per_segment=args.frames_per_segment,
            num_segments=num_segments,
            use_val_split=False,
        )
        summary_items.append(
            {
                "label": label,
                "checkpoint": str(checkpoint),
                "num_segments": num_segments,
                **compact_metrics(metrics),
            }
        )

    print("\nSummary:")
    for item in summary_items:
        print(
            f"  {item['label']:12s} T={item['num_segments']:>2d} "
            f"vAUC={item['video_auc']:.4f} "
            f"vAP={item['video_ap']:.4f} "
            f"fAUC={item['frame_auc']:.4f} "
            f"fAP={item['frame_ap']:.4f}"
        )

    payload = {
        "feature_dir": str(feature_dir),
        "counts": counts,
        "results": summary_items,
    }
    if args.output_json:
        out_path = Path(args.output_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved JSON: {out_path}")


if __name__ == "__main__":
    main()
