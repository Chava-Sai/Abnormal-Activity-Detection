"""
Temporal event prediction for Violence Event Detection.

Given one or more video feature files (.npy), runs full model inference,
applies temporal smoothing + boundary refinement, then outputs detected
violence event intervals: (t_start, t_end, category, confidence).

Usage:
    # Single video
    python src/predict.py \
        --checkpoint /path/to/best.pt \
        --feature    /path/to/Fighting001_x264_i3d.npy \
        --fps 25 --frames_per_segment 16

    # Batch over a directory
    python src/predict.py \
        --checkpoint /path/to/best.pt \
        --feature_dir /path/to/features \
        --output_json predictions.json \
        --fps 25

    # With custom thresholds
    python src/predict.py \
        --checkpoint /path/to/best.pt \
        --feature    /path/to/Fighting001_x264_i3d.npy \
        --threshold 0.45 --min_duration 2.0 --smooth_k 3
"""

import os
import sys
import json
import argparse
import re
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ViolenceDetector

# ── Constants ──────────────────────────────────────────────────────────────────
CAT_ID_TO_NAME = {
    0: 'Normal',
    1: 'Abuse',
    2: 'Fighting',
    3: 'Shooting',
    4: 'Explosion',
    5: 'Robbery',
    6: 'Riot',
}

DEFAULT_THRESHOLD   = 0.45   # anomaly score threshold
DEFAULT_SMOOTH_K    = 5      # Gaussian smooth window (segments)
DEFAULT_MIN_DUR     = 1.0    # minimum event duration (seconds)
DEFAULT_MERGE_GAP   = 1.0    # merge intervals closer than this (seconds)
DEFAULT_BOUNDARY_W  = 0.25   # weight for boundary-guided boundary snapping


# ── Feature loading ────────────────────────────────────────────────────────────
def load_features(feature_path: str, num_segments: int) -> np.ndarray:
    """Load and temporally resize features to num_segments × 2048."""
    feat = np.load(feature_path)
    if feat.ndim == 3:
        feat = feat.mean(axis=1)         # (T, 10, 2048) → (T, 2048)
    elif feat.ndim != 2:
        raise ValueError(f"Unexpected shape {feat.shape} in {feature_path}")

    T, D = feat.shape
    N    = num_segments

    if T == N:
        return feat.astype(np.float32)
    elif T > N:
        idx  = np.linspace(0, T - 1, N, dtype=int)
        return feat[idx].astype(np.float32)
    else:
        pad  = np.zeros((N - T, D), dtype=np.float32)
        return np.concatenate([feat, pad], axis=0)


# ── Inference ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def infer(model: ViolenceDetector, feature_path: str, num_segments: int,
          device: torch.device) -> dict:
    """Run full model forward pass; return raw score arrays."""
    feat = load_features(feature_path, num_segments)
    x    = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)

    out = model(x)

    trn_scores = out['trn_scores'].squeeze(0).cpu().numpy()   # (N,)
    mil_scores = out['mil_scores'].squeeze(0).cpu().numpy()   # (N,)
    cls_logits = out['cls_logits'].squeeze(0).cpu().numpy()   # (N, 7)
    boundary   = out['boundary'].squeeze(0).cpu().numpy()     # (N,) or (N-1,)

    # Pad boundary to length N if necessary (boundary head outputs N-1 transitions)
    if len(boundary) < num_segments:
        boundary = np.append(boundary, boundary[-1])

    return {
        'trn_scores': trn_scores,
        'mil_scores': mil_scores,
        'cls_logits': cls_logits,
        'boundary':   boundary,
        'n_segments': num_segments,
    }


# ── Temporal smoothing ─────────────────────────────────────────────────────────
def smooth_scores(scores: np.ndarray, k: int) -> np.ndarray:
    """
    Gaussian-weighted smoothing over k-segment window.
    Falls back to uniform filter when k is small.
    """
    if k <= 1:
        return scores.copy()

    # Build Gaussian kernel
    sigma   = k / 2.0
    half    = k // 2
    xs      = np.arange(-half, half + 1)
    kernel  = np.exp(-0.5 * (xs / sigma) ** 2)
    kernel /= kernel.sum()

    # Reflect padding to avoid edge artifacts
    padded  = np.pad(scores, half, mode='reflect')
    return np.convolve(padded, kernel, mode='valid')[:len(scores)]


# ── Boundary-guided score shaping ─────────────────────────────────────────────
def boundary_refine_scores(scores: np.ndarray, boundary: np.ndarray,
                           weight: float = DEFAULT_BOUNDARY_W) -> np.ndarray:
    """
    Amplify scores near high-boundary-confidence positions.

    The boundary head predicts confidence of a normal↔anomalous transition at
    each segment edge.  We boost the anomaly score in a 1-segment window
    around each high-confidence boundary, which sharpens detected intervals.

    Args:
        scores:   (N,) smoothed anomaly scores
        boundary: (N,) boundary confidence per segment
        weight:   mixing coefficient for boundary boost

    Returns:
        refined scores (N,), clipped to [0, 1]
    """
    # Boundary signal: spread each edge confidence onto neighbouring segments
    spread = np.zeros_like(boundary)
    spread[:-1] += boundary[1:]   # edge i influences segment i
    spread[1:]  += boundary[:-1]  # edge i-1 influences segment i
    spread = np.clip(spread / 2.0, 0, 1)

    # Refined score = original + weight * boundary_spread * (1 - score)
    # (only boosts scores that are already moderate, prevents clamping at 1)
    refined = scores + weight * spread * (1.0 - scores)
    return np.clip(refined, 0.0, 1.0)


# ── Per-segment category from cls_logits ──────────────────────────────────────
def segment_category(cls_logits: np.ndarray, trn_scores: np.ndarray,
                     seg_start: int, seg_end: int) -> tuple:
    """
    Return (cat_id, cat_name, prob) for segments [seg_start, seg_end).
    Uses top-scoring segments for robustness.
    """
    window_logits = cls_logits[seg_start:seg_end]       # (w, 7)
    window_scores = trn_scores[seg_start:seg_end]       # (w,)

    # Weighted average of class probabilities, weighted by TRN scores
    probs  = torch.softmax(
        torch.tensor(window_logits, dtype=torch.float32), dim=-1
    ).numpy()                                            # (w, 7)

    weights = window_scores / (window_scores.sum() + 1e-9)
    avg_prob = (probs * weights[:, None]).sum(axis=0)   # (7,)

    cat_id   = int(avg_prob.argmax())
    cat_name = CAT_ID_TO_NAME.get(cat_id, f'cat_{cat_id}')
    return cat_id, cat_name, float(avg_prob[cat_id])


# ── Thresholding → raw intervals ───────────────────────────────────────────────
def threshold_to_intervals(scores: np.ndarray, threshold: float,
                            fps: int, frames_per_seg: int) -> list:
    """
    Convert a binary-thresholded score array to time intervals (seconds).

    Returns list of dicts: {seg_start, seg_end, t_start, t_end}
    """
    binary     = (scores >= threshold).astype(int)
    intervals  = []
    in_event   = False
    start_seg  = 0
    seg_dur    = frames_per_seg / fps           # seconds per segment

    for i, b in enumerate(binary):
        if b and not in_event:
            in_event  = True
            start_seg = i
        elif not b and in_event:
            in_event = False
            intervals.append({
                'seg_start': start_seg,
                'seg_end':   i,
                't_start':   round(start_seg * seg_dur, 3),
                't_end':     round(i          * seg_dur, 3),
            })

    if in_event:  # handle event running to end of video
        i = len(binary)
        intervals.append({
            'seg_start': start_seg,
            'seg_end':   i,
            't_start':   round(start_seg * seg_dur, 3),
            't_end':     round(i          * seg_dur, 3),
        })

    return intervals


# ── Merge close intervals ──────────────────────────────────────────────────────
def merge_intervals(intervals: list, merge_gap_s: float) -> list:
    """
    Merge consecutive intervals separated by less than merge_gap_s seconds.
    """
    if not intervals:
        return []

    merged = [dict(intervals[0])]
    for iv in intervals[1:]:
        prev = merged[-1]
        gap  = iv['t_start'] - prev['t_end']
        if gap <= merge_gap_s:
            prev['t_end']   = iv['t_end']
            prev['seg_end'] = iv['seg_end']
        else:
            merged.append(dict(iv))

    return merged


# ── Filter short events ────────────────────────────────────────────────────────
def filter_short(intervals: list, min_duration_s: float) -> list:
    """Remove intervals shorter than min_duration_s seconds."""
    return [iv for iv in intervals if (iv['t_end'] - iv['t_start']) >= min_duration_s]


# ── Snap boundaries to boundary-confidence peaks ──────────────────────────────
def snap_to_boundary(intervals: list, boundary: np.ndarray,
                     fps: int, frames_per_seg: int,
                     snap_window: int = 2) -> list:
    """
    Fine-tune t_start / t_end of each interval to align with the nearest
    boundary-confidence peak within ±snap_window segments.

    This uses the boundary head output to achieve sub-segment precision.
    """
    seg_dur = frames_per_seg / fps
    N       = len(boundary)
    snapped = []

    for iv in intervals:
        new_iv = dict(iv)

        # Snap start
        lo = max(0,     iv['seg_start'] - snap_window)
        hi = min(N - 1, iv['seg_start'] + snap_window)
        best_start = lo + int(boundary[lo:hi + 1].argmax())
        new_iv['seg_start'] = best_start
        new_iv['t_start']   = round(best_start * seg_dur, 3)

        # Snap end
        lo = max(0,     iv['seg_end'] - snap_window)
        hi = min(N - 1, iv['seg_end'] + snap_window)
        best_end = lo + int(boundary[lo:hi + 1].argmax()) + 1  # exclusive
        best_end = min(best_end, N)
        new_iv['seg_end'] = best_end
        new_iv['t_end']   = round(best_end * seg_dur, 3)

        # Ensure start < end after snapping
        if new_iv['t_start'] >= new_iv['t_end']:
            new_iv = iv  # revert to un-snapped if snap collapsed the interval

        snapped.append(new_iv)

    return snapped


# ── Confidence score per interval ─────────────────────────────────────────────
def interval_confidence(iv: dict, scores: np.ndarray) -> float:
    """Mean TRN score over the interval's segments."""
    seg_scores = scores[iv['seg_start']:iv['seg_end']]
    if len(seg_scores) == 0:
        return 0.0
    return float(seg_scores.mean())


# ── Main prediction pipeline ───────────────────────────────────────────────────
def predict(
    model:             ViolenceDetector,
    feature_path:      str,
    device:            torch.device,
    num_segments:      int   = 32,
    fps:               int   = 25,
    frames_per_seg:    int   = 16,
    threshold:         float = DEFAULT_THRESHOLD,
    smooth_k:          int   = DEFAULT_SMOOTH_K,
    min_duration:      float = DEFAULT_MIN_DUR,
    merge_gap:         float = DEFAULT_MERGE_GAP,
    boundary_weight:   float = DEFAULT_BOUNDARY_W,
    snap_boundaries:   bool  = True,
) -> dict:
    """
    End-to-end prediction pipeline for a single video feature file.

    Returns:
        {
            'filepath':   str,
            'video_label': int,   # 0=Normal, 1=Anomalous (model decision)
            'video_score': float, # max smoothed score
            'events': [
                {
                  't_start':    float,  # seconds
                  't_end':      float,
                  'duration':   float,
                  'category':   str,
                  'cat_id':     int,
                  'confidence': float,
                },
                ...
            ],
            'scores': {           # raw/smoothed arrays for downstream use
                'trn_raw':    list,
                'trn_smooth': list,
                'boundary':   list,
            }
        }
    """
    # 1. Inference
    raw = infer(model, feature_path, num_segments, device)
    trn_raw  = raw['trn_scores']   # (N,)
    boundary = raw['boundary']     # (N,)
    cls_log  = raw['cls_logits']   # (N, 7)

    # 2. Smooth TRN scores
    trn_smooth = smooth_scores(trn_raw, k=smooth_k)

    # 3. Boundary-guided score refinement
    trn_refined = boundary_refine_scores(trn_smooth, boundary, weight=boundary_weight)

    # 4. Threshold → raw intervals
    intervals = threshold_to_intervals(trn_refined, threshold, fps, frames_per_seg)

    # 5. Merge gaps
    intervals = merge_intervals(intervals, merge_gap_s=merge_gap)

    # 6. Filter short events
    intervals = filter_short(intervals, min_duration_s=min_duration)

    # 7. Snap to boundary peaks
    if snap_boundaries and intervals:
        intervals = snap_to_boundary(intervals, boundary, fps, frames_per_seg)

    # 8. Annotate each interval with category + confidence
    events = []
    for iv in intervals:
        cat_id, cat_name, cat_conf = segment_category(
            cls_log, trn_refined, iv['seg_start'], iv['seg_end']
        )
        # Skip predicted-Normal intervals (model is confident it's background)
        if cat_id == 0:
            continue

        conf = interval_confidence(iv, trn_refined)
        events.append({
            't_start':    iv['t_start'],
            't_end':      iv['t_end'],
            'duration':   round(iv['t_end'] - iv['t_start'], 3),
            'category':   cat_name,
            'cat_id':     cat_id,
            'confidence': round(conf, 4),
        })

    # 9. Video-level decision
    video_score  = float(trn_refined.max())
    video_label  = 1 if video_score >= threshold else 0

    return {
        'filepath':    feature_path,
        'video_label': video_label,
        'video_score': round(video_score, 4),
        'events':      events,
        'scores': {
            'trn_raw':    trn_raw.tolist(),
            'trn_smooth': trn_refined.tolist(),
            'boundary':   boundary.tolist(),
        },
    }


# ── Batch prediction ───────────────────────────────────────────────────────────
def predict_batch(model, feature_paths: list, device, **kwargs) -> list:
    """Run predict() on a list of feature paths."""
    results = []
    for i, fp in enumerate(feature_paths):
        print(f"  [{i+1}/{len(feature_paths)}] {os.path.basename(fp)}", end='', flush=True)
        try:
            r = predict(model, fp, device, **kwargs)
            n_events = len(r['events'])
            label    = 'ANOMALOUS' if r['video_label'] else 'normal'
            print(f"  →  {label}  score={r['video_score']:.3f}  events={n_events}")
            results.append(r)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({'filepath': fp, 'error': str(e)})
    return results


# ── Pretty print for a single result ─────────────────────────────────────────
def print_result(result: dict):
    """Human-readable summary of a prediction result."""
    fp    = os.path.basename(result['filepath'])
    label = 'ANOMALOUS' if result['video_label'] else 'Normal'
    score = result['video_score']
    print(f"\n{'─'*60}")
    print(f"  File   : {fp}")
    print(f"  Label  : {label}  (score={score:.4f})")

    events = result.get('events', [])
    if not events:
        print(f"  Events : none detected")
    else:
        print(f"  Events : {len(events)} detected")
        for i, ev in enumerate(events, 1):
            print(f"    [{i}] {ev['t_start']:6.1f}s – {ev['t_end']:6.1f}s"
                  f"  ({ev['duration']:.1f}s)"
                  f"  cat={ev['category']:10s}"
                  f"  conf={ev['confidence']:.3f}")
    print(f"{'─'*60}")


# ── CLI ────────────────────────────────────────────────────────────────────────
def build_model(checkpoint_path: str, device: torch.device) -> ViolenceDetector:
    """Load checkpoint and reconstruct model."""
    ckpt       = torch.load(checkpoint_path, map_location=device)
    model_args = ckpt.get('args', {})
    model      = ViolenceDetector(
        input_dim=2048, num_classes=7,
        d_model=model_args.get('d_model', 512),
        nhead=model_args.get('nhead', 8),
        trn_layers=model_args.get('trn_layers', 2),
    ).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    epoch = ckpt.get('epoch', '?')
    auc   = ckpt.get('metrics', {}).get('video_auc', '?')
    if isinstance(auc, float):
        auc = f"{auc:.4f}"
    print(f"Loaded checkpoint: epoch={epoch}  val_auc={auc}")
    return model


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = build_model(args.checkpoint, device)

    # Collect feature paths
    feature_paths = []
    if args.feature:
        feature_paths = [os.path.expanduser(f) for f in args.feature]
    elif args.feature_dir:
        d = os.path.expanduser(args.feature_dir)
        feature_paths = sorted([
            os.path.join(d, f) for f in os.listdir(d)
            if f.endswith('.npy')
        ])

    if not feature_paths:
        print("No feature files specified. Use --feature or --feature_dir.")
        return

    print(f"\nProcessing {len(feature_paths)} video(s)...")

    predict_kwargs = dict(
        num_segments    = args.num_segments,
        fps             = args.fps,
        frames_per_seg  = args.frames_per_segment,
        threshold       = args.threshold,
        smooth_k        = args.smooth_k,
        min_duration    = args.min_duration,
        merge_gap       = args.merge_gap,
        boundary_weight = args.boundary_weight,
        snap_boundaries = not args.no_snap,
    )

    if len(feature_paths) == 1:
        result = predict(model, feature_paths[0], device, **predict_kwargs)
        print_result(result)
        all_results = [result]
    else:
        all_results = predict_batch(model, feature_paths, device, **predict_kwargs)

        # Summary stats
        n_anom   = sum(1 for r in all_results if r.get('video_label', 0) == 1)
        n_events = sum(len(r.get('events', [])) for r in all_results)
        print(f"\nSummary: {len(all_results)} videos | "
              f"{n_anom} anomalous | {n_events} events detected")

    # Save JSON
    if args.output_json:
        # Strip raw score arrays from JSON to keep it readable (optional)
        serialisable = []
        for r in all_results:
            r2 = {k: v for k, v in r.items() if k != 'scores'}
            serialisable.append(r2)
        with open(args.output_json, 'w') as f:
            json.dump(serialisable, f, indent=2)
        print(f"\nPredictions saved to {args.output_json}")

    # Save CSV (compact, easy to import into Excel / LaTeX)
    if args.output_csv:
        rows = []
        for r in all_results:
            fname = os.path.basename(r.get('filepath', ''))
            # Infer ground-truth label from filename
            base = re.sub(r'_x264_i3d\.npy$', '', fname)
            gt_cat = re.sub(r'_?\d+$', '', base)
            if gt_cat.startswith('Normal'):
                gt_cat = 'Normal'
            for ev in r.get('events', []):
                rows.append(
                    f"{fname},{gt_cat},{ev['t_start']},{ev['t_end']},"
                    f"{ev['duration']},{ev['category']},{ev['confidence']:.4f}"
                )
            if not r.get('events'):
                rows.append(f"{fname},{gt_cat},,,,,")

        with open(args.output_csv, 'w') as f:
            f.write("filename,gt_category,t_start,t_end,duration,pred_category,confidence\n")
            f.write('\n'.join(rows) + '\n')
        print(f"CSV saved to {args.output_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Violence event temporal prediction from I3D features'
    )

    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--feature',     nargs='+',
                             help='One or more .npy feature file paths')
    input_group.add_argument('--feature_dir', type=str,
                             help='Directory of .npy feature files (batch mode)')

    # Model
    parser.add_argument('--checkpoint',         required=True)
    parser.add_argument('--num_segments',       type=int,   default=32)

    # Video timing
    parser.add_argument('--fps',                type=int,   default=25)
    parser.add_argument('--frames_per_segment', type=int,   default=16)

    # Detection parameters
    parser.add_argument('--threshold',          type=float, default=DEFAULT_THRESHOLD,
                        help='Anomaly score threshold (default: 0.45)')
    parser.add_argument('--smooth_k',           type=int,   default=DEFAULT_SMOOTH_K,
                        help='Gaussian smoothing window in segments (default: 5)')
    parser.add_argument('--min_duration',       type=float, default=DEFAULT_MIN_DUR,
                        help='Min event duration in seconds (default: 1.0)')
    parser.add_argument('--merge_gap',          type=float, default=DEFAULT_MERGE_GAP,
                        help='Merge events closer than this (seconds, default: 1.0)')
    parser.add_argument('--boundary_weight',    type=float, default=DEFAULT_BOUNDARY_W,
                        help='Boundary head score boost weight (default: 0.25)')
    parser.add_argument('--no_snap',            action='store_true',
                        help='Disable boundary snapping')

    # Output
    parser.add_argument('--output_json',        default=None,
                        help='Save predictions to JSON file')
    parser.add_argument('--output_csv',         default=None,
                        help='Save predictions to CSV file (one row per event)')

    args = parser.parse_args()
    main(args)
