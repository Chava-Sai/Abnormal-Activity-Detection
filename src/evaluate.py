"""
Comprehensive evaluation for Violence Event Detection.

Supports two evaluation modes:
  1. Video-level AUC  — uses max segment score per video as video-level predictor
  2. Frame-level AUC  — expands segment scores to frame predictions (×16 frames/segment)
                        and computes ROC-AUC over frames (matches RTFM protocol)

For the frame-level eval we use a "proxy" ground truth:
  - All frames in an anomalous video → gt = 1
  - All frames in a normal video     → gt = 0
This is a valid lower bound for temporal localization performance. For exact
RTFM-style evaluation you need the full 290-video test set + gt-ucf.npy.

Per-category AP is also computed to understand which categories the model
handles well / poorly.

Usage (standalone):
    python src/evaluate.py \
        --checkpoint /path/to/best.pt \
        --feature_dir /projectnb/cs585/students/saichava/datasets/ucf_train/UCF_Train_ten_crop_i3d \
        --list_file   ~/violence_detection/list/ucf-i3d-val-split.list \
        --frames_per_segment 16 \
        --output_json eval_results.json
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, auc
)
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ViolenceDetector
from dataset import UCFCrimeDataset, InMemoryDataset, make_train_val_split


def infer_input_dim_from_checkpoint(ckpt: dict, fallback: int = 2048) -> int:
    model_args = ckpt.get('args', {})
    if model_args.get('input_dim') is not None:
        return int(model_args['input_dim'])

    weight = ckpt.get('state_dict', {}).get('mil_scorer.net.0.weight')
    if weight is not None:
        return int(weight.shape[1])
    return fallback

CAT_ID_TO_NAME = {
    0: 'Normal',
    1: 'Abuse',
    2: 'Fighting',
    3: 'Shooting',
    4: 'Explosion',
    5: 'Robbery',
    6: 'Riot',
}


def resolve_device(requested: str) -> torch.device:
    if requested == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')

    if requested == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError("Requested --device cuda but CUDA is not available")
        return torch.device('cuda')

    if requested == 'mps':
        if not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available():
            raise ValueError("Requested --device mps but MPS is not available")
        return torch.device('mps')

    if requested == 'cpu':
        return torch.device('cpu')

    raise ValueError(f"unsupported device: {requested}")


@torch.no_grad()
def run_inference(model, dataset, device, frames_per_segment: int = 16):
    """
    Run model inference on every video in the dataset.

    Returns:
        results: list of dicts with keys:
          - filepath: str
          - label: int (0=normal, 1=anomalous)
          - cat_id: int
          - video_score: float (max TRN score across segments)
          - mil_score:   float (max MIL score across segments)
          - segment_scores: np.ndarray (T,) — TRN scores per segment
          - frame_scores:   np.ndarray (T*fps,) — scores expanded to frames
          - n_frames: int
    """
    model.eval()
    results = []

    for i in range(len(dataset)):
        item = dataset[i]
        feat   = item['features'].unsqueeze(0).to(device)  # (1, T, 2048)
        label  = item['label'].item()
        cat_id = item['cat_id'].item()
        fpath  = item['filepath']

        out = model(feat)

        trn_scores = out['trn_scores'].squeeze(0).cpu().numpy()   # (T,)
        mil_scores = out['mil_scores'].squeeze(0).cpu().numpy()   # (T,)
        T = len(trn_scores)

        frame_scores = np.repeat(trn_scores, frames_per_segment)  # (T*16,)
        n_frames     = len(frame_scores)

        results.append({
            'filepath':       fpath,
            'label':          label,
            'cat_id':         cat_id,
            'video_score':    float(trn_scores.max()),
            'mil_score':      float(mil_scores.max()),
            'segment_scores': trn_scores,
            'frame_scores':   frame_scores,
            'n_frames':       n_frames,
        })

    return results


def compute_video_level_metrics(results: list) -> dict:
    """Video-level AUC and AP (max segment score per video)."""
    scores = np.array([r['video_score'] for r in results])
    labels = np.array([r['label']       for r in results])

    if len(np.unique(labels)) < 2:
        return {'video_auc': 0.0, 'video_ap': 0.0}

    return {
        'video_auc': float(roc_auc_score(labels, scores)),
        'video_ap':  float(average_precision_score(labels, scores)),
    }


def compute_frame_level_metrics(results: list) -> dict:
    """
    Frame-level AUC using proxy GT (all frames in anomalous video = 1).
    Matches RTFM evaluation protocol structure.
    """
    all_pred, all_gt = [], []

    for r in results:
        n = r['n_frames']
        all_pred.extend(r['frame_scores'].tolist())
        all_gt.extend([r['label']] * n)  # proxy: label applies to all frames

    pred = np.array(all_pred)
    gt   = np.array(all_gt)

    if len(np.unique(gt)) < 2:
        return {'frame_auc': 0.0, 'frame_ap': 0.0}

    fpr, tpr, _ = roc_curve(gt, pred)
    frame_auc   = float(auc(fpr, tpr))
    frame_ap    = float(average_precision_score(gt, pred))

    return {
        'frame_auc': frame_auc,
        'frame_ap':  frame_ap,
        'n_frames':  len(gt),
        'n_anomalous_frames': int(gt.sum()),
    }


def compute_per_category_metrics(results: list) -> dict:
    """
    Per-category Average Precision (one-vs-rest).
    Useful to diagnose which violence types the model struggles with.
    """
    cat_metrics = {}

    # All anomalous categories present
    cat_ids = set(r['cat_id'] for r in results if r['label'] == 1)

    normal_scores = np.array([r['video_score'] for r in results if r['label'] == 0])

    for cid in sorted(cat_ids):
        cat_name  = CAT_ID_TO_NAME.get(cid, f'cat_{cid}')
        cat_res   = [r for r in results if r['cat_id'] == cid]
        cat_scores = np.array([r['video_score'] for r in cat_res])

        # Binary: this category vs normal
        scores = np.concatenate([cat_scores, normal_scores])
        labels = np.concatenate([np.ones(len(cat_scores)), np.zeros(len(normal_scores))])

        if len(cat_scores) == 0:
            continue

        ap = float(average_precision_score(labels, scores)) if len(np.unique(labels)) > 1 else 0.0
        cat_metrics[cat_name] = {
            'n_videos': len(cat_scores),
            'ap':       ap,
            'mean_score': float(cat_scores.mean()),
        }

    return cat_metrics


def evaluate_checkpoint(
    checkpoint_path: str,
    feature_dir: str,
    list_file: str,
    device: torch.device,
    frames_per_segment: int = 16,
    num_segments: int = 32,
    d_model: int = 512,
    nhead: int = 8,
    trn_layers: int = 2,
    use_val_split: bool = False,
    val_ratio: float = 0.2,
) -> dict:
    """Load checkpoint and run full evaluation."""

    # Load model
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_args = ckpt.get('args', {})
    input_dim = infer_input_dim_from_checkpoint(ckpt)

    model = ViolenceDetector(
        input_dim=input_dim,
        num_classes=7,
        d_model=model_args.get('d_model', d_model),
        nhead=model_args.get('nhead', nhead),
        trn_layers=model_args.get('trn_layers', trn_layers),
    ).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    epoch   = ckpt.get('epoch', '?')
    tr_auc  = ckpt.get('metrics', {}).get('auc', '?')
    print(f"Loaded checkpoint: epoch={epoch}, reported_auc={tr_auc}")

    # Build dataset
    if use_val_split:
        _, val_samples = make_train_val_split(list_file, feature_dir, val_ratio=val_ratio)
        dataset = InMemoryDataset(val_samples, num_segments=num_segments)
    else:
        dataset = UCFCrimeDataset(
            feature_dir, list_file or 'auto',
            mode='test', num_segments=num_segments, violence_only=False
        )

    print(f"Running inference on {len(dataset)} videos...")
    results = run_inference(model, dataset, device, frames_per_segment)

    # Compute metrics
    vid_metrics  = compute_video_level_metrics(results)
    frm_metrics  = compute_frame_level_metrics(results)
    cat_metrics  = compute_per_category_metrics(results)

    metrics = {
        'checkpoint':    checkpoint_path,
        'epoch':         epoch,
        'n_videos':      len(results),
        **vid_metrics,
        **frm_metrics,
        'per_category':  cat_metrics,
    }

    # Pretty print
    print(f"\n{'='*55}")
    print(f"  Video-level AUC : {vid_metrics['video_auc']:.4f}")
    print(f"  Video-level AP  : {vid_metrics['video_ap']:.4f}")
    print(f"  Frame-level AUC : {frm_metrics['frame_auc']:.4f}  (proxy GT)")
    print(f"  Frame-level AP  : {frm_metrics['frame_ap']:.4f}  (proxy GT)")
    print(f"{'='*55}")
    print(f"\nPer-category AP (video-level, one-vs-normal):")
    for cat, m in sorted(cat_metrics.items(), key=lambda x: -x[1]['ap']):
        bar = '█' * int(m['ap'] * 20)
        print(f"  {cat:12s}: AP={m['ap']:.3f}  n={m['n_videos']:3d}  {bar}")

    return metrics


def main(args):
    device = resolve_device(args.device)
    print(f"Device: {device}")

    metrics = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        feature_dir=os.path.expanduser(args.feature_dir),
        list_file=(os.path.expanduser(args.list_file) if args.list_file != 'auto' else 'auto'),
        device=device,
        frames_per_segment=args.frames_per_segment,
        num_segments=args.num_segments,
        use_val_split=args.val_split,
        val_ratio=args.val_ratio,
    )

    if args.output_json:
        # Convert numpy arrays to lists for JSON serialization
        for r in metrics.get('per_category', {}).values():
            pass  # already serializable
        with open(args.output_json, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"\nResults saved to {args.output_json}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',        required=True)
    parser.add_argument('--feature_dir',       required=True)
    parser.add_argument('--list_file',         default='auto',
                        help='List file (val split or test list)')
    parser.add_argument('--frames_per_segment', type=int, default=16)
    parser.add_argument('--num_segments',       type=int, default=32)
    parser.add_argument('--val_split',          action='store_true',
                        help='Create val split from list_file (training list)')
    parser.add_argument('--val_ratio',          type=float, default=0.2)
    parser.add_argument('--device',             choices=('auto', 'cpu', 'cuda', 'mps'), default='auto')
    parser.add_argument('--output_json',        default=None)
    args = parser.parse_args()
    main(args)
