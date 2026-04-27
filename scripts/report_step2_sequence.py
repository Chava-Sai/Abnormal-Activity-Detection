#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_summary(run_path: str) -> dict:
    path = Path(run_path).expanduser().resolve()
    if path.is_dir():
        path = path / "run_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"run summary not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    payload["_summary_path"] = str(path)
    return payload


def load_csv_rows(csv_path: str, schedule: str, seed: int, experiment: str | None) -> list[dict]:
    path = Path(csv_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    matched: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if experiment and row.get("experiment") != experiment:
                continue
            if row.get("training_schedule") != schedule:
                continue
            if int(row.get("seed", -1)) != seed:
                continue
            matched.append(
                {
                    "training_schedule": row["training_schedule"],
                    "seed": int(row["seed"]),
                    "num_segments": int(row["num_segments"]),
                    "best_metrics": {
                        "video_auc": float(row["best_video_auc"]),
                    },
                    "_summary_path": str(path),
                    "_run_dir": row.get("run_dir", ""),
                }
            )
    if not matched:
        raise FileNotFoundError(
            f"No rows found in {path} for training_schedule={schedule}, seed={seed}"
        )
    matched.sort(key=lambda item: item["num_segments"])
    return matched


def format_row(summary: dict) -> tuple[int, float, str]:
    num_segments = int(summary["num_segments"])
    video_auc = float(summary.get("best_metrics", {}).get("video_auc", 0.0))
    summary_path = summary.get("_summary_path", "")
    return num_segments, video_auc, summary_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize Step 2 temporal sequence-length ablation (num_segments=8/16/32)"
    )
    parser.add_argument("--runs", nargs="*", help="Run dirs or run_summary.json files")
    parser.add_argument("--csv", help="Optional CSV source such as results_ablation.csv")
    parser.add_argument("--schedule", default="progressive", help="Training schedule label")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment", default="sequence_length_ablation")
    args = parser.parse_args()

    if args.csv:
        rows = load_csv_rows(
            args.csv,
            schedule=args.schedule,
            seed=args.seed,
            experiment=args.experiment,
        )
    else:
        if not args.runs:
            raise SystemExit("Pass --runs <run1> <run2> <run3>, or use --csv.")
        rows = [load_summary(path) for path in args.runs]
        rows.sort(key=lambda item: int(item["num_segments"]))

    print("# Step 2: Temporal Sequence Length Ablation")
    print("| training_schedule | num_segments | seed | best_video_auc | summary_path |")
    print("|---|---:|---:|---:|---|")
    for row in rows:
        num_segments, video_auc, summary_path = format_row(row)
        print(
            f"| {row.get('training_schedule', args.schedule)} | {num_segments} | "
            f"{int(row.get('seed', args.seed))} | {video_auc:.4f} | {summary_path} |"
        )

    best = max(rows, key=lambda item: float(item.get("best_metrics", {}).get("video_auc", 0.0)))
    print()
    print(
        "Conclusion: "
        f"T={int(best['num_segments'])} achieved the highest val video AUC "
        f"({float(best['best_metrics']['video_auc']):.4f})."
    )


if __name__ == "__main__":
    main()
