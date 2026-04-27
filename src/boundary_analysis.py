#!/usr/bin/env python3
"""
Boundary Precision Analysis for Violence Event Detection.

This script supports two modes:

1. Strict GT temporal-boundary evaluation
   Use an explicit split list plus `--annotation_file` to compare predicted
   event boundaries against frame-level UCF annotations. This mode produces:
     - `start_error_histogram.png`
     - `end_error_histogram.png`
     - `iou_distribution.png`
     - `early_late_bias.png`
     - `boundary_eval.csv`
     - `boundary_summary.txt`

2. Proxy-only analysis
   When no temporal annotation file is provided, the script falls back to a
   proxy analysis on score/boundary behavior, which is still useful on the
   train/val splits that do not expose frame-level event onset/offset labels.

Recommended usage for the current repo state:
  - Strict Step 4: run on `UCF_split_test_i3d_5cat` with
    `Temporal_Anomaly_Annotation_for_Testing_Videos.txt`
  - Proxy-only val diagnostics: run on `ucf-i3d-val-split.list`
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import InMemoryDataset, UCFCrimeDataset, make_train_val_split
from feature_utils import load_feature_array, strip_feature_suffix
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


def pad_boundary(boundary: np.ndarray, target_len: int) -> np.ndarray:
    if boundary.size == 0:
        return np.zeros(target_len, dtype=np.float32)
    if len(boundary) >= target_len:
        return boundary[:target_len].astype(np.float32, copy=False)
    return np.append(boundary, np.repeat(boundary[-1], target_len - len(boundary))).astype(np.float32)


def smooth_scores(scores: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return scores.astype(np.float32, copy=True)

    sigma = max(k / 2.0, 1e-6)
    half = k // 2
    xs = np.arange(-half, half + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (xs / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(scores, half, mode="reflect")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[:len(scores)].astype(np.float32, copy=False)


def boundary_refine_scores(scores: np.ndarray, boundary: np.ndarray, weight: float) -> np.ndarray:
    spread = np.zeros_like(boundary, dtype=np.float32)
    spread[:-1] += boundary[1:]
    spread[1:] += boundary[:-1]
    spread = np.clip(spread / 2.0, 0.0, 1.0)
    refined = scores + weight * spread * (1.0 - scores)
    return np.clip(refined, 0.0, 1.0)


def threshold_to_intervals(scores: np.ndarray, threshold: float) -> List[dict]:
    binary = (scores >= threshold).astype(np.int32)
    intervals: List[dict] = []
    in_event = False
    start_seg = 0

    for idx, flag in enumerate(binary):
        if flag and not in_event:
            in_event = True
            start_seg = idx
        elif not flag and in_event:
            in_event = False
            intervals.append({"seg_start": start_seg, "seg_end": idx})

    if in_event:
        intervals.append({"seg_start": start_seg, "seg_end": len(binary)})

    return intervals


def merge_intervals(intervals: Sequence[dict], merge_gap_segments: int) -> List[dict]:
    if not intervals:
        return []

    merged = [dict(intervals[0])]
    for interval in intervals[1:]:
        prev = merged[-1]
        gap = int(interval["seg_start"]) - int(prev["seg_end"])
        if gap <= merge_gap_segments:
            prev["seg_end"] = int(interval["seg_end"])
        else:
            merged.append(dict(interval))
    return merged


def filter_short(intervals: Sequence[dict], min_duration_segments: int) -> List[dict]:
    return [
        dict(interval)
        for interval in intervals
        if (int(interval["seg_end"]) - int(interval["seg_start"])) >= min_duration_segments
    ]


def snap_to_boundary(intervals: Sequence[dict], boundary: np.ndarray, snap_window: int) -> List[dict]:
    if not intervals:
        return []

    n = len(boundary)
    snapped: List[dict] = []
    for interval in intervals:
        seg_start = int(interval["seg_start"])
        seg_end = int(interval["seg_end"])
        new_interval = dict(interval)

        lo = max(0, seg_start - snap_window)
        hi = min(n - 1, seg_start + snap_window)
        best_start = lo + int(np.argmax(boundary[lo:hi + 1]))
        new_interval["seg_start"] = best_start

        lo = max(0, seg_end - snap_window)
        hi = min(n - 1, seg_end + snap_window)
        best_end = lo + int(np.argmax(boundary[lo:hi + 1])) + 1
        new_interval["seg_end"] = min(best_end, n)

        if int(new_interval["seg_start"]) >= int(new_interval["seg_end"]):
            new_interval = dict(interval)
        snapped.append(new_interval)

    return snapped


def interval_confidence(interval: dict, scores: np.ndarray) -> float:
    lo = int(interval["seg_start"])
    hi = int(interval["seg_end"])
    if hi <= lo:
        return 0.0
    return float(scores[lo:hi].mean())


def choose_primary_interval(intervals: Sequence[dict], scores: np.ndarray) -> Optional[dict]:
    if not intervals:
        return None
    return max(intervals, key=lambda interval: (interval_confidence(interval, scores), interval["seg_end"] - interval["seg_start"]))


def build_dataset(
    feature_dir: str,
    list_file: str,
    num_segments: int,
    direct_list: bool,
    val_ratio: float,
    seed: int,
) -> tuple[torch.utils.data.Dataset, str]:
    if direct_list:
        dataset = UCFCrimeDataset(
            feature_dir,
            list_file,
            mode="test",
            num_segments=num_segments,
            violence_only=False,
        )
        split_name = f"direct:{Path(list_file).name if list_file != 'auto' else 'auto'}"
    else:
        _, samples = make_train_val_split(
            list_file,
            feature_dir,
            val_ratio=val_ratio,
            seed=seed,
        )
        dataset = InMemoryDataset(samples, num_segments=num_segments)
        split_name = "val_split"
    return dataset, split_name


def sample_index_map(raw_segments: int, num_segments: int) -> np.ndarray:
    if raw_segments <= 0:
        return np.zeros(num_segments, dtype=np.int32)
    if raw_segments >= num_segments:
        return np.linspace(0, raw_segments - 1, num_segments, dtype=int).astype(np.int32)

    pad = np.repeat(raw_segments - 1, num_segments - raw_segments)
    return np.concatenate([np.arange(raw_segments, dtype=np.int32), pad])


def resized_interval_to_raw_segments(
    interval: dict,
    raw_segments: int,
    num_segments: int,
) -> Tuple[int, int]:
    seg_start = int(interval["seg_start"])
    seg_end = int(interval["seg_end"])

    if raw_segments <= 0:
        return 0, 0

    if raw_segments >= num_segments:
        idx = sample_index_map(raw_segments, num_segments)
        start_raw = int(idx[min(seg_start, num_segments - 1)])
        if seg_end <= 0:
            end_raw = start_raw + 1
        else:
            end_raw = int(idx[min(seg_end - 1, num_segments - 1)]) + 1
    else:
        start_raw = min(seg_start, raw_segments - 1)
        end_raw = min(seg_end, raw_segments)

    start_raw = max(0, min(start_raw, raw_segments - 1))
    end_raw = max(start_raw + 1, min(end_raw, raw_segments))
    return start_raw, end_raw


def raw_segments_to_frames(start_seg: int, end_seg: int, frames_per_segment: int) -> Tuple[int, int]:
    start_frame = int(start_seg * frames_per_segment)
    end_frame = int(end_seg * frames_per_segment)
    return start_frame, end_frame


def iou_1d(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    left = max(int(a[0]), int(b[0]))
    right = min(int(a[1]), int(b[1]))
    inter = max(0, right - left)
    union = max(int(a[1]), int(b[1])) - min(int(a[0]), int(b[0]))
    if union <= 0:
        return 0.0
    return float(inter / union)


def parse_temporal_annotations(annotation_file: str) -> Dict[str, dict]:
    annotations: Dict[str, dict] = {}

    with open(annotation_file, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 6:
                continue

            video_name = parts[0]
            category = parts[1]
            values = [int(float(token)) for token in parts[2:6]]
            intervals = []
            for idx in range(0, len(values), 2):
                start = values[idx]
                end = values[idx + 1]
                if start >= 0 and end >= 0 and end >= start:
                    intervals.append((start, end + 1))

            annotations[video_name] = {
                "video_name": video_name,
                "category": category,
                "intervals": intervals,
            }

    return annotations


def feature_path_to_video_name(filepath: str) -> str:
    basename = os.path.basename(filepath)
    if basename.endswith("_i3d.npy"):
        return basename[:-8] + ".mp4"
    if basename.endswith(".npy"):
        return basename[:-4] + ".mp4"
    return basename


def gt_frames_to_raw_segments(
    intervals: Sequence[Tuple[int, int]],
    raw_segments: int,
    frames_per_segment: int,
) -> List[Tuple[int, int]]:
    gt_segments: List[Tuple[int, int]] = []
    for start_frame, end_frame in intervals:
        start_seg = int(math.floor(start_frame / frames_per_segment))
        end_seg = int(math.ceil(end_frame / frames_per_segment))
        if raw_segments > 0:
            start_seg = max(0, min(start_seg, raw_segments - 1))
            end_seg = max(start_seg + 1, min(end_seg, raw_segments))
        gt_segments.append((start_seg, end_seg))
    return gt_segments


def local_maxima_indices(values: np.ndarray, positive_only: bool = True) -> List[int]:
    if values.size == 0:
        return []

    peaks: List[int] = []
    for idx, value in enumerate(values):
        if positive_only and value <= 0:
            continue
        left = values[idx - 1] if idx > 0 else -float("inf")
        right = values[idx + 1] if idx + 1 < len(values) else -float("inf")
        if value >= left and value >= right:
            peaks.append(idx)
    return peaks


def directional_boundary_candidates(
    refined_scores: np.ndarray,
    boundary: np.ndarray,
    raw_segments: int,
    num_segments: int,
    frames_per_segment: int,
) -> tuple[List[dict], List[dict]]:
    """
    Build onset/offset boundary peak candidates.

    The boundary head is non-directional, so we inject direction using the local
    TRN score change:
      onset signal  = boundary * positive score delta
      offset signal = boundary * negative score delta
    """
    if len(refined_scores) == 0:
        return [], []

    idx_map = sample_index_map(raw_segments, num_segments)
    delta = np.diff(refined_scores, append=refined_scores[-1]).astype(np.float32, copy=False)
    onset_signal = boundary * np.clip(delta, 0.0, None)
    offset_signal = boundary * np.clip(-delta, 0.0, None)

    def edge_frame(edge_idx: int) -> int:
        edge_idx = int(min(max(edge_idx, 0), len(idx_map) - 1))
        return int(idx_map[edge_idx] * frames_per_segment)

    onset_peaks = local_maxima_indices(onset_signal, positive_only=True)
    offset_peaks = local_maxima_indices(offset_signal, positive_only=True)
    if not onset_peaks:
        onset_peaks = [int(np.argmax(onset_signal))]
    if not offset_peaks:
        offset_peaks = [int(np.argmax(offset_signal))]

    onset_candidates = [
        {
            "edge_idx": peak,
            "frame": edge_frame(peak),
            "strength": float(onset_signal[peak]),
        }
        for peak in onset_peaks
    ]
    offset_candidates = [
        {
            "edge_idx": peak,
            "frame": edge_frame(peak + 1),
            "strength": float(offset_signal[peak]),
        }
        for peak in offset_peaks
    ]
    return onset_candidates, offset_candidates


def nearest_candidate_frame(candidates: Sequence[dict], target_frame: int) -> tuple[Optional[int], float]:
    if not candidates:
        return None, float("nan")
    best = min(candidates, key=lambda row: abs(int(row["frame"]) - int(target_frame)))
    return int(best["frame"]), float(best["strength"])


@torch.no_grad()
def run_inference(
    model: ViolenceDetector,
    dataset,
    device: torch.device,
    threshold: float,
    smooth_k: int,
    boundary_weight: float,
    merge_gap_segments: int,
    min_duration_segments: int,
    snap_window: int,
    disable_snap: bool,
    frames_per_segment: int,
    log_every: int,
) -> List[dict]:
    model.eval()
    results: List[dict] = []

    num_segments = int(dataset.num_segments) if getattr(dataset, "num_segments", None) is not None else None
    if num_segments is None:
        raise ValueError("dataset.num_segments is required for boundary analysis")

    for idx in range(len(dataset)):
        item = dataset[idx]
        filepath = item["filepath"]
        raw_feat = load_feature_array(filepath)
        raw_segments = int(raw_feat.shape[0])
        feat = item["features"].unsqueeze(0).to(device)
        label = int(item["label"].item())
        cat_id = int(item["cat_id"].item())

        out = model(feat)
        trn_scores = out["trn_scores"].squeeze(0).detach().cpu().numpy().astype(np.float32)
        mil_scores = out["mil_scores"].squeeze(0).detach().cpu().numpy().astype(np.float32)
        boundary = pad_boundary(
            out["boundary"].squeeze(0).detach().cpu().numpy().astype(np.float32),
            len(trn_scores),
        )

        trn_smooth = smooth_scores(trn_scores, smooth_k)
        trn_refined = boundary_refine_scores(trn_smooth, boundary, boundary_weight)

        intervals = threshold_to_intervals(trn_refined, threshold)
        intervals = merge_intervals(intervals, merge_gap_segments=merge_gap_segments)
        intervals = filter_short(intervals, min_duration_segments=min_duration_segments)
        if intervals and not disable_snap:
            intervals = snap_to_boundary(intervals, boundary, snap_window=snap_window)

        primary_interval = choose_primary_interval(intervals, trn_refined)
        primary_raw_interval = None
        primary_frame_interval = None
        if primary_interval is not None:
            raw_start, raw_end = resized_interval_to_raw_segments(primary_interval, raw_segments, num_segments)
            primary_raw_interval = (raw_start, raw_end)
            primary_frame_interval = raw_segments_to_frames(raw_start, raw_end, frames_per_segment)

        results.append({
            "idx": idx,
            "filepath": filepath,
            "video_name": feature_path_to_video_name(filepath),
            "label": label,
            "cat_id": cat_id,
            "cat_name": CAT_ID_TO_NAME.get(cat_id, f"cat_{cat_id}"),
            "raw_segments": raw_segments,
            "trn_scores": trn_scores,
            "mil_scores": mil_scores,
            "boundary": boundary,
            "refined_scores": trn_refined,
            "video_score": float(trn_refined.max()),
            "intervals": intervals,
            "primary_interval": primary_interval,
            "primary_raw_interval": primary_raw_interval,
            "primary_frame_interval": primary_frame_interval,
        })

        if log_every > 0 and (idx + 1) % log_every == 0:
            print(f"  Processed {idx + 1}/{len(dataset)} videos")

    return results


def proxy_boundary_summary(results: Sequence[dict], output_dir: str) -> str:
    anomalous_bt, normal_bt, peak_positions = [], [], []
    rel_starts, rel_ends = [], []

    for result in results:
        boundary = result["boundary"]
        if result["label"] == 1:
            anomalous_bt.extend(boundary.tolist())
        else:
            normal_bt.extend(boundary.tolist())

        peaks = np.flatnonzero(boundary >= np.percentile(boundary, 90)) if len(boundary) else np.array([])
        for peak in peaks:
            peak_positions.append(float(peak / max(len(boundary), 1)))

        primary = result["primary_interval"]
        if result["label"] == 1 and primary is not None and len(result["refined_scores"]) > 0:
            denom = float(len(result["refined_scores"]))
            rel_starts.append(float(primary["seg_start"] / denom))
            rel_ends.append(float(primary["seg_end"] / denom))

    if anomalous_bt and normal_bt:
        bins = np.linspace(0, 1, 40)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(anomalous_bt, bins=bins, alpha=0.6, density=True, color="tomato", label=f"Anomalous (n={len(anomalous_bt)})")
        ax.hist(normal_bt, bins=bins, alpha=0.6, density=True, color="steelblue", label=f"Normal (n={len(normal_bt)})")
        ax.set_xlabel("Boundary confidence")
        ax.set_ylabel("Density")
        ax.set_title("Boundary Confidence Distribution")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "boundary_conf_distribution.png"), dpi=150)
        plt.close(fig)

    if peak_positions:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(peak_positions, bins=25, density=True, color="darkorange", edgecolor="white")
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.7, label="Uniform baseline")
        ax.set_xlabel("Relative boundary-peak position")
        ax.set_ylabel("Density")
        ax.set_title("Boundary Peak Distribution")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "boundary_peak_distribution.png"), dpi=150)
        plt.close(fig)

    if rel_starts and rel_ends:
        bins = np.linspace(0, 1, 20)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(rel_starts, bins=bins, color="tomato", edgecolor="white")
        axes[0].axvline(np.mean(rel_starts), color="black", linestyle="--", label=f"mean={np.mean(rel_starts):.2f}")
        axes[0].set_title("Predicted Event Start Position")
        axes[0].set_xlabel("Relative video position")
        axes[0].legend(fontsize=9)

        axes[1].hist(rel_ends, bins=bins, color="steelblue", edgecolor="white")
        axes[1].axvline(np.mean(rel_ends), color="black", linestyle="--", label=f"mean={np.mean(rel_ends):.2f}")
        axes[1].set_title("Predicted Event End Position")
        axes[1].set_xlabel("Relative video position")
        axes[1].legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "event_position_bias.png"), dpi=150)
        plt.close(fig)

    lines = [
        "=" * 70,
        "BOUNDARY PROXY ANALYSIS",
        "=" * 70,
        f"Videos           : {len(results)}",
        f"Anomalous videos : {sum(1 for result in results if result['label'] == 1)}",
        f"Normal videos    : {sum(1 for result in results if result['label'] == 0)}",
    ]
    if anomalous_bt and normal_bt:
        lines.extend([
            f"Mean boundary confidence (anom): {float(np.mean(anomalous_bt)):.4f}",
            f"Mean boundary confidence (norm): {float(np.mean(normal_bt)):.4f}",
        ])
    if rel_starts and rel_ends:
        lines.extend([
            f"Mean predicted start position : {float(np.mean(rel_starts)):.3f}",
            f"Mean predicted end position   : {float(np.mean(rel_ends)):.3f}",
        ])

    summary_path = os.path.join(output_dir, "boundary_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return summary_path


def write_boundary_eval_csv(rows: Sequence[dict], output_path: str) -> None:
    fieldnames = [
        "video_name",
        "filepath",
        "label",
        "cat_name",
        "gt_category",
        "has_gt",
        "has_prediction",
        "gt_start_frame",
        "gt_end_frame",
        "pred_start_frame",
        "pred_end_frame",
        "pred_start_strength",
        "pred_end_strength",
        "start_error_frames",
        "end_error_frames",
        "iou",
        "gt_match_index",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def plot_signed_error_hist(errors: Sequence[float], title: str, xlabel: str, color: str, output_path: str) -> None:
    if not errors:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(min(errors), max(errors), 25) if min(errors) != max(errors) else 20
    ax.hist(errors, bins=bins, color=color, edgecolor="white", alpha=0.9)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2)
    ax.axvline(float(np.mean(errors)), color="gray", linestyle=":", linewidth=1.4, label=f"mean={np.mean(errors):.1f}")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_iou_distribution(ious: Sequence[float], output_path: str) -> None:
    if not ious:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ious, bins=np.linspace(0, 1, 21), color="seagreen", edgecolor="white", alpha=0.9)
    ax.axvline(float(np.mean(ious)), color="black", linestyle="--", label=f"mean={np.mean(ious):.3f}")
    ax.set_title("Predicted vs GT IoU Distribution")
    ax.set_xlabel("IoU")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_early_late_bias(start_errors: Sequence[float], end_errors: Sequence[float], output_path: str) -> None:
    if not start_errors and not end_errors:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    data = []
    labels = []
    if start_errors:
        data.append(list(start_errors))
        labels.append("Start")
    if end_errors:
        data.append(list(end_errors))
        labels.append("End")

    ax.boxplot(
        data,
        vert=False,
        tick_labels=labels,
        showmeans=True,
        patch_artist=True,
        boxprops={"facecolor": "#E8EEF9", "edgecolor": "#355C7D"},
        medianprops={"color": "#C06C84", "linewidth": 1.5},
        meanprops={"marker": "o", "markerfacecolor": "#355C7D", "markeredgecolor": "#355C7D"},
    )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Signed Error (frames)   negative=early, positive=late")
    ax.set_title("Early/Late Boundary Bias")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def strict_boundary_eval(
    results: Sequence[dict],
    annotations: Dict[str, dict],
    num_segments: int,
    frames_per_segment: int,
    output_dir: str,
) -> str:
    rows: List[dict] = []
    start_errors: List[float] = []
    end_errors: List[float] = []
    ious: List[float] = []
    false_alarms = 0
    gt_anomaly_videos = 0
    detected_gt_videos = 0
    missing_annotations = 0

    for result in results:
        ann = annotations.get(result["video_name"])
        if ann is None:
            missing_annotations += 1
            continue

        gt_intervals = ann["intervals"]
        gt_exists = len(gt_intervals) > 0
        if gt_exists:
            gt_anomaly_videos += 1

        row = {
            "video_name": result["video_name"],
            "filepath": result["filepath"],
            "label": result["label"],
            "cat_name": result["cat_name"],
            "gt_category": ann["category"],
            "has_gt": int(gt_exists),
            "has_prediction": 0,
            "gt_start_frame": None,
            "gt_end_frame": None,
            "pred_start_frame": None,
            "pred_end_frame": None,
            "start_error_frames": None,
            "end_error_frames": None,
            "iou": None,
            "gt_match_index": None,
            "pred_start_strength": None,
            "pred_end_strength": None,
        }

        # A valid temporal prediction must first survive score thresholding.
        # Otherwise every video would always yield at least one argmax-based
        # onset/offset pair, which massively overcounts false alarms on normal videos.
        predicted_intervals = result.get("intervals") or []
        row["has_prediction"] = int(bool(predicted_intervals))

        if not row["has_prediction"]:
            if not gt_exists:
                rows.append(row)
                continue
            row["iou"] = 0.0
            rows.append(row)
            if gt_exists:
                ious.append(0.0)
            continue

        onset_candidates, offset_candidates = directional_boundary_candidates(
            refined_scores=result["refined_scores"],
            boundary=result["boundary"],
            raw_segments=result["raw_segments"],
            num_segments=num_segments,
            frames_per_segment=frames_per_segment,
        )
        row["has_prediction"] = int(bool(onset_candidates) and bool(offset_candidates))

        if not gt_exists:
            if row["has_prediction"]:
                false_alarms += 1
                row["iou"] = 0.0
            rows.append(row)
            continue

        if not row["has_prediction"]:
            row["iou"] = 0.0
            rows.append(row)
            ious.append(0.0)
            continue

        interval_candidates = []
        for gt_idx, (gt_start_frame, gt_end_frame) in enumerate(gt_intervals):
            pred_start_frame, pred_start_strength = nearest_candidate_frame(onset_candidates, gt_start_frame)
            pred_end_frame, pred_end_strength = nearest_candidate_frame(offset_candidates, gt_end_frame)

            if pred_start_frame is None or pred_end_frame is None:
                continue
            if pred_end_frame <= pred_start_frame:
                later_offsets = [candidate for candidate in offset_candidates if int(candidate["frame"]) > pred_start_frame]
                if later_offsets:
                    best_later = min(later_offsets, key=lambda candidate: abs(int(candidate["frame"]) - gt_end_frame))
                    pred_end_frame = int(best_later["frame"])
                    pred_end_strength = float(best_later["strength"])
                else:
                    pred_end_frame = pred_start_frame + frames_per_segment

            start_error = float(pred_start_frame - gt_start_frame)
            end_error = float(pred_end_frame - gt_end_frame)
            pred_frame_interval = (pred_start_frame, pred_end_frame)
            gt_frame_interval = (gt_start_frame, gt_end_frame)
            inter = max(0, min(pred_frame_interval[1], gt_frame_interval[1]) - max(pred_frame_interval[0], gt_frame_interval[0]))
            union = max(pred_frame_interval[1], gt_frame_interval[1]) - min(pred_frame_interval[0], gt_frame_interval[0])
            iou = float(inter / union) if union > 0 else 0.0
            interval_candidates.append({
                "gt_idx": gt_idx,
                "gt_start_frame": gt_start_frame,
                "gt_end_frame": gt_end_frame,
                "pred_start_frame": pred_start_frame,
                "pred_end_frame": pred_end_frame,
                "pred_start_strength": pred_start_strength,
                "pred_end_strength": pred_end_strength,
                "start_error": start_error,
                "end_error": end_error,
                "iou": iou,
                "abs_error_sum": abs(start_error) + abs(end_error),
            })

        if not interval_candidates:
            row["iou"] = 0.0
            rows.append(row)
            ious.append(0.0)
            continue

        best = max(interval_candidates, key=lambda item: (item["iou"], -item["abs_error_sum"]))
        detected_gt_videos += 1

        row.update({
            "gt_start_frame": best["gt_start_frame"],
            "gt_end_frame": best["gt_end_frame"],
            "pred_start_frame": best["pred_start_frame"],
            "pred_end_frame": best["pred_end_frame"],
            "pred_start_strength": best["pred_start_strength"],
            "pred_end_strength": best["pred_end_strength"],
            "start_error_frames": best["start_error"],
            "end_error_frames": best["end_error"],
            "iou": best["iou"],
            "gt_match_index": best["gt_idx"],
        })
        rows.append(row)

        start_errors.append(best["start_error"])
        end_errors.append(best["end_error"])
        ious.append(best["iou"])

    csv_path = os.path.join(output_dir, "boundary_eval.csv")
    write_boundary_eval_csv(rows, csv_path)

    plot_signed_error_hist(
        start_errors,
        title="Start Boundary Error Histogram",
        xlabel="Predicted start - GT onset (frames)",
        color="tomato",
        output_path=os.path.join(output_dir, "start_error_histogram.png"),
    )
    plot_signed_error_hist(
        end_errors,
        title="End Boundary Error Histogram",
        xlabel="Predicted end - GT offset (frames)",
        color="steelblue",
        output_path=os.path.join(output_dir, "end_error_histogram.png"),
    )
    plot_iou_distribution(ious, os.path.join(output_dir, "iou_distribution.png"))
    plot_early_late_bias(start_errors, end_errors, os.path.join(output_dir, "early_late_bias.png"))

    mean_abs_start = float(np.mean(np.abs(start_errors))) if start_errors else float("nan")
    mean_abs_end = float(np.mean(np.abs(end_errors))) if end_errors else float("nan")
    mean_signed_start = float(np.mean(start_errors)) if start_errors else float("nan")
    mean_signed_end = float(np.mean(end_errors)) if end_errors else float("nan")
    mean_iou = float(np.mean(ious)) if ious else float("nan")
    median_iou = float(np.median(ious)) if ious else float("nan")

    start_bias = "early" if mean_signed_start < 0 else "late"
    end_bias = "early" if mean_signed_end < 0 else "late"

    lines = [
        "=" * 70,
        "BOUNDARY PRECISION ANALYSIS",
        "=" * 70,
        "Method                                : directional boundary peaks matched to GT onset/offset",
        f"Total videos with annotations parsed : {len(annotations)}",
        f"Results evaluated                    : {len(rows)}",
        f"GT anomalous videos                  : {gt_anomaly_videos}",
        f"Videos with matched peak pairs       : {detected_gt_videos}",
        f"Normal-video false alarms            : {false_alarms}",
        f"Missing annotation matches           : {missing_annotations}",
        "",
        f"Mean absolute start error (frames)   : {mean_abs_start:.2f}",
        f"Mean absolute end error (frames)     : {mean_abs_end:.2f}",
        f"Mean signed start error (frames)     : {mean_signed_start:.2f} ({start_bias})",
        f"Mean signed end error (frames)       : {mean_signed_end:.2f} ({end_bias})",
        f"Mean IoU                             : {mean_iou:.4f}",
        f"Median IoU                           : {median_iou:.4f}",
        f"IoU >= 0.1                           : {sum(iou >= 0.1 for iou in ious)}/{len(ious)}",
        f"IoU >= 0.3                           : {sum(iou >= 0.3 for iou in ious)}/{len(ious)}",
        f"IoU >= 0.5                           : {sum(iou >= 0.5 for iou in ious)}/{len(ious)}",
        "",
        "Outputs:",
        "  boundary_eval.csv",
        "  start_error_histogram.png",
        "  end_error_histogram.png",
        "  iou_distribution.png",
        "  early_late_bias.png",
    ]

    summary_path = os.path.join(output_dir, "boundary_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Boundary precision analysis")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature_dir", required=True)
    parser.add_argument("--list_file", default="auto")
    parser.add_argument("--output_dir", default="./visualizations/boundary_analysis")
    parser.add_argument("--annotation_file", default=None,
                        help="Optional UCF temporal annotation txt for strict GT evaluation")
    parser.add_argument("--num_segments", type=int, default=None,
                        help="Override num_segments; otherwise infer from checkpoint args")
    parser.add_argument("--frames_per_segment", type=int, default=None,
                        help="Override frames_per_segment; otherwise infer from checkpoint args")
    parser.add_argument("--threshold", type=float, default=0.6843)
    parser.add_argument("--smooth_k", type=int, default=5)
    parser.add_argument("--boundary_weight", type=float, default=0.25)
    parser.add_argument("--merge_gap_segments", type=int, default=1)
    parser.add_argument("--min_duration_segments", type=int, default=1)
    parser.add_argument("--snap_window", type=int, default=2)
    parser.add_argument("--disable_snap", action="store_true")
    parser.add_argument("--direct_list", action="store_true",
                        help="Use list_file as an explicit split list instead of regenerating a val split")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
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

    ckpt = torch.load(checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})
    num_segments = int(args.num_segments or ckpt_args.get("num_segments", 32))
    frames_per_segment = int(args.frames_per_segment or ckpt_args.get("frames_per_segment", 16))

    model = ViolenceDetector(
        input_dim=infer_input_dim_from_checkpoint(ckpt),
        num_classes=7,
        d_model=ckpt_args.get("d_model", 512),
        nhead=ckpt_args.get("nhead", 8),
        trn_layers=ckpt_args.get("trn_layers", 2),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    print(
        f"Loaded checkpoint: epoch={ckpt.get('epoch', '?')} "
        f"T={num_segments} frames_per_segment={frames_per_segment}"
    )

    dataset, split_name = build_dataset(
        feature_dir=feature_dir,
        list_file=list_file,
        num_segments=num_segments,
        direct_list=args.direct_list,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"Dataset split: {split_name}")
    print(f"Videos: {len(dataset)}")

    results = run_inference(
        model=model,
        dataset=dataset,
        device=device,
        threshold=args.threshold,
        smooth_k=args.smooth_k,
        boundary_weight=args.boundary_weight,
        merge_gap_segments=args.merge_gap_segments,
        min_duration_segments=args.min_duration_segments,
        snap_window=args.snap_window,
        disable_snap=args.disable_snap,
        frames_per_segment=frames_per_segment,
        log_every=args.log_every,
    )

    if args.annotation_file:
        annotation_file = os.path.abspath(os.path.expanduser(args.annotation_file))
        annotations = parse_temporal_annotations(annotation_file)
        summary_path = strict_boundary_eval(
            results=results,
            annotations=annotations,
            num_segments=num_segments,
            frames_per_segment=frames_per_segment,
            output_dir=output_dir,
        )
        print(f"Strict GT boundary analysis saved to {output_dir}")
        print(f"Summary: {summary_path}")
        return

    summary_path = proxy_boundary_summary(results, output_dir)
    print(f"Proxy boundary analysis saved to {output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
