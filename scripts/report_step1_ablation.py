#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_best_pt(path: Path) -> dict:
    try:
        import pickle
        import torch
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "best.pt exists but could not be read. Install torch in the current Python environment, "
            "or use --csv results_ablation.csv instead."
        ) from exc

    checkpoint = torch.load(path, map_location="cpu", pickle_module=pickle)
    args = checkpoint.get("args", {}) or {}
    metrics = checkpoint.get("metrics", {}) or {}
    training_schedule = checkpoint.get("training_schedule")
    if not training_schedule:
        training_schedule = "joint" if args.get("joint") else "progressive"
    return {
        "training_schedule": training_schedule,
        "joint": bool(args.get("joint", False)),
        "seed": int(args.get("seed", -1)),
        "num_segments": int(args.get("num_segments", -1)),
        "best_epoch": int(checkpoint.get("epoch", -1)),
        "best_metrics": metrics,
        "checkpoint_dir": str(path.parent.resolve()),
        "_summary_path": str(path.resolve()),
    }


def load_summary(path_str: str) -> dict:
    path = Path(path_str).expanduser().resolve()
    raw = path_str.strip()
    if raw.startswith("/path/to/"):
        raise FileNotFoundError(
            "You passed the example placeholder path literally. Replace it with a real run directory, "
            "or use --csv results_ablation.csv to compare the repo's recorded results."
        )
    if path.is_dir():
        summary_path = path / "run_summary.json"
        best_ckpt_path = path / "checkpoints" / "best.pt"
        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            payload["_summary_path"] = str(summary_path)
            return payload
        if best_ckpt_path.exists():
            return _read_best_pt(best_ckpt_path)
        raise FileNotFoundError(
            f"Neither run_summary.json nor checkpoints/best.pt was found under: {path}"
        )
    if not path.exists():
        raise FileNotFoundError(
            f"Path not found: {path}\n"
            "Pass a real run directory, a run_summary.json file, a checkpoints/best.pt file, "
            "or use --csv results_ablation.csv."
        )
    if path.name == "run_summary.json":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        payload["_summary_path"] = str(path)
        return payload
    if path.name == "best.pt":
        return _read_best_pt(path)
    raise FileNotFoundError(
        f"Unsupported input path: {path}\n"
        "Expected a run directory, run_summary.json, or checkpoints/best.pt."
    )


def _match_csv_row(rows: list[dict], schedule: str, num_segments: int, seed: int) -> dict:
    for row in rows:
        if row.get("training_schedule") != schedule:
            continue
        if int(row.get("num_segments", -1)) != num_segments:
            continue
        if int(row.get("seed", -1)) != seed:
            continue
        return row
    raise FileNotFoundError(
        f"No row found in CSV for training_schedule={schedule}, num_segments={num_segments}, seed={seed}"
    )


def load_csv_pair(csv_path: str, num_segments: int, seed: int) -> tuple[dict, dict]:
    path = Path(csv_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    progressive_row = _match_csv_row(rows, "progressive", num_segments, seed)
    joint_row = _match_csv_row(rows, "joint", num_segments, seed)

    def _to_summary(row: dict) -> dict:
        return {
            "training_schedule": row["training_schedule"],
            "num_segments": int(row["num_segments"]),
            "seed": int(row["seed"]),
            "best_metrics": {
                "video_auc": float(row["best_video_auc"]),
            },
            "_summary_path": str(path),
            "_run_dir": row.get("run_dir", ""),
        }

    return _to_summary(progressive_row), _to_summary(joint_row)


def format_row(summary: dict) -> tuple[str, int, int, float]:
    schedule = str(summary.get("training_schedule", "unknown"))
    num_segments = int(summary.get("num_segments", -1))
    seed = int(summary.get("seed", -1))
    best_auc = float(summary.get("best_metrics", {}).get("video_auc", 0.0))
    return schedule, num_segments, seed, best_auc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare progressive vs joint Step 1 runs from run_summary.json files."
    )
    parser.add_argument("--progressive", help="Path to progressive run dir, run_summary.json, or checkpoints/best.pt")
    parser.add_argument("--joint", help="Path to joint run dir, run_summary.json, or checkpoints/best.pt")
    parser.add_argument("--csv", help="Path to results_ablation.csv for comparing recorded results without run dirs")
    parser.add_argument("--num-segments", type=int, default=32, help="Used with --csv mode")
    parser.add_argument("--seed", type=int, default=42, help="Used with --csv mode")
    args = parser.parse_args()

    if args.csv:
        progressive, joint = load_csv_pair(args.csv, num_segments=args.num_segments, seed=args.seed)
    else:
        if not args.progressive or not args.joint:
            default_csv = DEFAULT_REPO_ROOT / "results_ablation.csv"
            raise SystemExit(
                "Either provide both --progressive and --joint, or use --csv.\n"
                f"Example:\n"
                f"  python3 {Path(__file__).resolve()} --csv {default_csv} --num-segments 32 --seed 42"
            )
        progressive = load_summary(args.progressive)
        joint = load_summary(args.joint)

    p_sched, p_seg, p_seed, p_auc = format_row(progressive)
    j_sched, j_seg, j_seed, j_auc = format_row(joint)

    print("# Step 1: Progressive vs Joint Training")
    print("| training_schedule | num_segments | seed | best_video_auc | summary_path |")
    print("|---|---:|---:|---:|---|")
    print(f"| {p_sched} | {p_seg} | {p_seed} | {p_auc:.4f} | {progressive['_summary_path']} |")
    print(f"| {j_sched} | {j_seg} | {j_seed} | {j_auc:.4f} | {joint['_summary_path']} |")
    print()

    delta = p_auc - j_auc
    if delta > 0:
        verdict = f"progressive outperformed joint by {delta:.4f} video AUC"
    elif delta < 0:
        verdict = f"joint outperformed progressive by {abs(delta):.4f} video AUC"
    else:
        verdict = "progressive and joint tied on video AUC"
    print(f"Conclusion: {verdict}.")


if __name__ == "__main__":
    main()
