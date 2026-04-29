"""
ShanghaiTech Robustness Evaluation
===================================
Zero-shot binary anomaly detection eval using a model trained on UCF-Crime.
Reports frame-level AUC and AP (as per paper Table 2 / Table 3 scope).

Usage:
    # Single GPU
    python evaluate_shanghaitech.py \
        --checkpoint /path/to/best.pt \
        --feat_dir   /path/to/test_features \
        --mask_dir   /path/to/test_frame_mask

    # Two GPUs (DataParallel)
    python evaluate_shanghaitech.py \
        --checkpoint /path/to/best.pt \
        --feat_dir   /path/to/test_features \
        --mask_dir   /path/to/test_frame_mask \
        --gpus 0 1

    # Override frames-per-segment if your features used a different stride
    python evaluate_shanghaitech.py ... --frames_per_segment 16
"""

import os
import sys
import json
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

# ── allow running from any working directory ──────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ViolenceDetector


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device,
               gpu_ids: list) -> nn.Module:
    """
    Load ViolenceDetector from checkpoint.
    Infers input_dim from the saved weights so the eval always matches
    whatever the checkpoint was trained with (1024 or 2048).
    Wraps in DataParallel when multiple GPUs are requested.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get('state_dict', ckpt)          # tolerate bare state dicts

    # Infer input_dim from first MIL layer weight shape
    mil_w = state.get('mil_scorer.net.0.weight')
    if mil_w is None:
        # try without DataParallel 'module.' prefix
        mil_w = state.get('module.mil_scorer.net.0.weight')
    input_dim = int(mil_w.shape[1]) if mil_w is not None else 2048

    # Build from saved args if present, else use sensible defaults
    saved_args = ckpt.get('args', {}) if isinstance(ckpt, dict) else {}
    model = ViolenceDetector(
        input_dim  = input_dim,
        num_classes= saved_args.get('num_classes', 7),
        d_model    = saved_args.get('d_model',     512),
        nhead      = saved_args.get('nhead',       8),
        trn_layers = saved_args.get('trn_layers',  2),
    )

    # Strip 'module.' prefix from DataParallel-saved checkpoints
    clean_state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(clean_state, strict=True)

    # Multi-GPU
    if len(gpu_ids) > 1:
        print(f"Using DataParallel on GPUs: {gpu_ids}")
        model = nn.DataParallel(model, device_ids=gpu_ids)

    model.to(device)
    model.eval()

    epoch = ckpt.get('epoch', '?') if isinstance(ckpt, dict) else '?'
    val_auc = ckpt.get('metrics', {}).get('video_auc', '?') \
              if isinstance(ckpt, dict) else '?'
    print(f"Checkpoint loaded  |  input_dim={input_dim}  "
          f"epoch={epoch}  val_auc={val_auc}")
    return model, input_dim


# ── feature loading ───────────────────────────────────────────────────────────

def load_npy_feature(path: str) -> np.ndarray:
    """
    Load a .npy feature file → float32 (T, D).
    Handles both plain (T, D) and 10-crop (T, 10, D) layouts.
    """
    feat = np.load(path)
    if feat.ndim == 3:
        feat = feat.mean(axis=1)          # 10-crop → mean
    elif feat.ndim != 2:
        raise ValueError(f"Unexpected shape {feat.shape} in {path}")
    return feat.astype(np.float32)


def temporal_resize(feat: np.ndarray, n: int) -> np.ndarray:
    """Uniformly sample to n segments (or zero-pad if shorter)."""
    t, d = feat.shape
    if t == n:
        return feat
    if t > n:
        idx = np.linspace(0, t - 1, n, dtype=int)
        return feat[idx]
    pad = np.zeros((n - t, d), dtype=np.float32)
    return np.concatenate([feat, pad], axis=0)


# ── GT mask loading ───────────────────────────────────────────────────────────

def find_mask(mask_dir: str, video_id: str) -> str | None:
    """
    Locate the ground-truth frame mask for a video.
    ShanghaiTech naming: <scene>_<clip>.npy  e.g. 01_001.npy
    Tries exact match and common variants.
    """
    candidates = [
        os.path.join(mask_dir, f"{video_id}.npy"),
        os.path.join(mask_dir, f"{video_id}_label.npy"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


# ── inference for one video ───────────────────────────────────────────────────

@torch.no_grad()
def infer_video(model: nn.Module, feat_path: str,
                num_segments: int, device: torch.device,
                input_dim: int = 1024) -> np.ndarray:
    feat = temporal_resize(load_npy_feature(feat_path), num_segments)
    x    = torch.tensor(feat).unsqueeze(0).to(device)   # (1, T, D)

    if x.shape[-1] == 2048 and input_dim == 1024:
        x = x.reshape(x.shape[0], x.shape[1], 1024, 2).mean(-1)

    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    out  = raw_model(x)
    trn  = out['trn_scores'].squeeze(0).cpu().numpy()
    return trn

# ── evaluation loop ───────────────────────────────────────────────────────────

def evaluate(args):
    # ── device setup ──────────────────────────────────────────────────────────
    if args.gpus and torch.cuda.is_available():
        gpu_ids = [int(g) for g in args.gpus]
        device  = torch.device(f'cuda:{gpu_ids[0]}')
    elif torch.cuda.is_available():
        gpu_ids = [0]
        device  = torch.device('cuda:0')
    else:
        gpu_ids = []
        device  = torch.device('cpu')

    print(f"Device : {device}  |  GPUs : {gpu_ids if gpu_ids else 'none (CPU)'}")

    # ── load model ────────────────────────────────────────────────────────────
    model, input_dim = load_model(args.checkpoint, device, gpu_ids)

    # ── collect test feature files ────────────────────────────────────────────
    feat_files = sorted(
        glob.glob(os.path.join(args.feat_dir, '**', '*.npy'), recursive=True) +
        glob.glob(os.path.join(args.feat_dir, '*.npy'))
    )
    # deduplicate (glob may return duplicates across patterns)
    feat_files = sorted(set(feat_files))

    if not feat_files:
        raise FileNotFoundError(
            f"No .npy files found under {args.feat_dir}\n"
            "Check --feat_dir points to the TEST feature directory."
        )
    print(f"Found {len(feat_files)} feature files in {args.feat_dir}")

    # ── main eval loop ────────────────────────────────────────────────────────
    all_frame_scores  = []
    all_frame_labels  = []
    skipped           = []
    per_video_results = []

    for i, fp in enumerate(feat_files):
        video_id = os.path.splitext(os.path.basename(fp))[0]

        # locate GT mask
        mask_path = find_mask(args.mask_dir, video_id)
        if mask_path is None:
            skipped.append(video_id)
            continue

        # model forward
        try:
            seg_scores = infer_video(model, fp, args.num_segments, device, input_dim)
        except Exception as e:
            print(f"  [WARN] {video_id}: inference failed — {e}")
            skipped.append(video_id)
            continue

        # load GT mask (frame-level binary: 0=normal, 1=anomaly)
        gt = np.load(mask_path).astype(np.int32).flatten()
        total_frames = len(gt)

        # expand segment scores → frame scores
        # each segment covers args.frames_per_segment frames
        frame_scores = np.repeat(seg_scores, args.frames_per_segment)

        # align lengths: truncate or pad to match GT
        if len(frame_scores) >= total_frames:
            frame_scores = frame_scores[:total_frames]
        else:
            # pad last score to cover remaining frames
            pad_len = total_frames - len(frame_scores)
            frame_scores = np.concatenate(
                [frame_scores, np.full(pad_len, frame_scores[-1])]
            )

        all_frame_scores.extend(frame_scores.tolist())
        all_frame_labels.extend(gt.tolist())

        per_video_results.append({
            'video_id':    video_id,
            'n_frames':    total_frames,
            'n_anomalous': int(gt.sum()),
            'max_score':   float(seg_scores.max()),
            'mean_score':  float(seg_scores.mean()),
        })

        if (i + 1) % 20 == 0 or (i + 1) == len(feat_files):
            print(f"  Processed {i+1}/{len(feat_files)} videos "
                  f"(skipped {len(skipped)})")

    if skipped:
        print(f"\n[WARN] Skipped {len(skipped)} videos (no GT mask found):")
        for v in skipped[:10]:
            print(f"  {v}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped)-10} more")

    # ── compute metrics ───────────────────────────────────────────────────────
    all_scores = np.array(all_frame_scores)
    all_labels = np.array(all_frame_labels)

    if all_labels.sum() == 0:
        raise ValueError(
            "No anomalous frames found in GT masks. "
            "Check --mask_dir is the TEST mask directory (not train)."
        )

    auc = roc_auc_score(all_labels, all_scores)
    ap  = average_precision_score(all_labels, all_scores)

    # ── print results ─────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  ShanghaiTech Robustness Evaluation (Zero-Shot)")
    print(f"{'='*55}")
    print(f"  Videos evaluated  : {len(per_video_results)}")
    print(f"  Videos skipped    : {len(skipped)}")
    print(f"  Total frames      : {len(all_labels):,}")
    print(f"  Anomalous frames  : {int(all_labels.sum()):,}  "
          f"({100*all_labels.mean():.1f}%)")
    print(f"  Frame-level AUC   : {auc:.4f}")
    print(f"  Frame-level AP    : {ap:.4f}")
    print(f"{'='*55}\n")

    # ── save results ──────────────────────────────────────────────────────────
    results = {
        'dataset':          'ShanghaiTech',
        'eval_type':        'zero_shot_binary',
        'checkpoint':       args.checkpoint,
        'feat_dir':         args.feat_dir,
        'num_segments':     args.num_segments,
        'frames_per_seg':   args.frames_per_segment,
        'videos_evaluated': len(per_video_results),
        'videos_skipped':   len(skipped),
        'total_frames':     int(len(all_labels)),
        'anomalous_frames': int(all_labels.sum()),
        'anomalous_pct':    float(100 * all_labels.mean()),
        'frame_auc':        float(auc),
        'frame_ap':         float(ap),
        'per_video':        per_video_results,
    }

    out_path = args.output_json
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved → {out_path}")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Zero-shot ShanghaiTech evaluation for violence detector'
    )
    p.add_argument('--checkpoint',        required=True,
                   help='Path to best.pt checkpoint')
    p.add_argument('--feat_dir',          required=True,
                   help='Directory of TEST .npy feature files')
    p.add_argument('--mask_dir',          required=True,
                   help='Directory of frame-level GT masks (.npy, 0/1 per frame)')
    p.add_argument('--num_segments',      type=int, default=32,
                   help='Segments per video fed to model (default: 32)')
    p.add_argument('--frames_per_segment',type=int, default=16,
                   help='Frames per segment used during extraction (default: 16)')
    p.add_argument('--gpus',              nargs='+', default=None,
                   help='GPU ids to use, e.g. --gpus 0 1  (default: auto)')
    p.add_argument('--output_json',       default='shanghai_eval_results.json',
                   help='Where to save the JSON results')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
