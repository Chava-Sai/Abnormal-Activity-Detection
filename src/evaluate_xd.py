"""
Zero-shot evaluation of our UCF-Crime trained model on XD-Violence features.

XD-Violence I3D features:
  - RGB/  : (T, 1024) per clip-crop file
  - Flow/ : (T, 1024) per clip-crop file
  - Concatenated RGB+Flow → (T, 2048) matches our model's input_dim

Labels are embedded in filenames:
  - *_label_A_*  → Normal  (label=0)
  - *_label_B*_* → Violent (label=1)  [B1..B6 = violence subtypes]

Evaluation:
  - For each clip that has BOTH RGB and Flow features:
      * Load 10 crops (or however many exist), average features
      * Run through our model (zero-shot — no fine-tuning)
      * Video score = max TRN segment score
  - Compute AUROC over all clips

Usage:
    python src/evaluate_xd.py \
        --checkpoint /path/to/best.pt \
        --feature_dir /path/to/i3d-features \
        --output_dir  /path/to/output \
        --split train
"""

import os
import sys
import argparse
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ViolenceDetector


# ── XD-Violence label / category maps ─────────────────────────────────────────
#  label_A                 → Normal
#  label_B1 / B1-*        → Fighting
#  label_B2 / B2-*        → Car Accident
#  label_B3 / B3-*        → Explosion
#  label_B4 / B4-*        → Riot
#  label_B5 / B5-*        → Shooting
#  label_B6 / B6-*        → Abuse
SUBTYPE_MAP = {
    'B1': 'Fighting',
    'B2': 'CarAccident',
    'B3': 'Explosion',
    'B4': 'Riot',
    'B5': 'Shooting',
    'B6': 'Abuse',
}

# Categories that overlap with our UCF-Crime training
OVERLAP_CATS = {'Fighting', 'Explosion', 'Shooting', 'Abuse'}


def parse_label(filename):
    """
    Extract binary label and subtype from XD-Violence filename.
    Returns (label, subtype_name)
      label = 0 (Normal) or 1 (Violent)
    """
    name = os.path.basename(filename)
    if '_label_A' in name:
        return 0, 'Normal'
    for key, cat in SUBTYPE_MAP.items():
        if f'_label_{key}' in name or f'_label_B{key[1]}' in name:
            return 1, cat
    if '_label_B' in name:
        return 1, 'Violent'
    return -1, 'Unknown'


def get_clip_stem(filename):
    """
    Strip the crop index suffix (__0, __1, ...) from filename to get clip stem.
    E.g.: 'Movie__#ts_label_A__3.npy' → 'Movie__#ts_label_A'
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    # Remove trailing __N
    parts = base.rsplit('__', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return base


def load_clip_features(feature_dir, split, clip_stem, stream='RGB'):
    """
    Load all crop files for a clip and average them.
    Returns averaged feature array of shape (T, 1024), or None if missing.
    """
    if split == 'train':
        subdir = stream          # 'RGB' or 'Flow'
    else:
        subdir = stream + 'Test' # 'RGBTest' or 'FlowTest'

    dir_path = os.path.join(feature_dir, subdir)
    crops = []
    for crop_idx in range(10):
        fname = f"{clip_stem}__{crop_idx}.npy"
        fpath = os.path.join(dir_path, fname)
        if os.path.exists(fpath):
            crops.append(np.load(fpath))

    if not crops:
        return None

    # Stack and average → (T, 1024)
    # Crops may have slightly different T — use minimum
    min_T = min(c.shape[0] for c in crops)
    crops  = np.stack([c[:min_T] for c in crops], axis=0)  # (N_crops, T, 1024)
    return crops.mean(axis=0)                                # (T, 1024)


def resample(feat, num_segments=32):
    """Resample (T, D) to (num_segments, D)."""
    T = feat.shape[0]
    if T == num_segments:
        return feat
    idx = np.linspace(0, T - 1, num_segments).astype(int)
    return feat[idx]


# ── Model loading ──────────────────────────────────────────────────────────────
def load_model(checkpoint_path, device):
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
    epoch   = ckpt.get('epoch', '?')
    metrics = ckpt.get('metrics', {})
    v_auc   = metrics.get('video_auc', float('nan'))
    print(f"Loaded checkpoint: epoch={epoch}  UCF-Crime vAUC={v_auc:.4f}")
    return model


# ── Main evaluation ────────────────────────────────────────────────────────────
@torch.no_grad()
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = load_model(args.checkpoint, device)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Discover clip stems from RGB directory ──────────────────────────────
    rgb_subdir = 'RGB' if args.split == 'train' else 'RGBTest'
    rgb_dir    = os.path.join(args.feature_dir, rgb_subdir)

    if not os.path.isdir(rgb_dir) or len(os.listdir(rgb_dir)) == 0:
        print(f"[ERROR] {rgb_dir} is empty or missing. "
              f"Cannot evaluate {args.split} split without RGB features.")
        return

    # Collect unique clip stems from RGB directory
    rgb_files  = [f for f in os.listdir(rgb_dir) if f.endswith('.npy')]
    clip_stems = sorted(set(get_clip_stem(f) for f in rgb_files))
    print(f"Found {len(rgb_files)} RGB crop files → {len(clip_stems)} unique clips")

    # ── Run inference ───────────────────────────────────────────────────────
    scores_all  = []
    labels_all  = []
    subtypes_all = []
    skipped     = 0

    for i, stem in enumerate(clip_stems):
        label, subtype = parse_label(stem)
        if label == -1:
            skipped += 1
            continue

        # Load RGB and Flow, concatenate → (T, 2048)
        rgb_feat  = load_clip_features(args.feature_dir, args.split, stem, 'RGB')
        flow_feat = load_clip_features(args.feature_dir, args.split, stem, 'Flow')

        if rgb_feat is None or flow_feat is None:
            skipped += 1
            continue

        # Align T dimension
        min_T = min(rgb_feat.shape[0], flow_feat.shape[0])
        feat  = np.concatenate([rgb_feat[:min_T], flow_feat[:min_T]], axis=1)  # (T, 2048)

        # Resample to num_segments
        feat = resample(feat, args.num_segments)                                 # (32, 2048)
        feat_t = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device) # (1, 32, 2048)

        out       = model(feat_t)
        trn_score = out['trn_scores'].squeeze(0).cpu().numpy()
        vid_score = float(np.max(trn_score))

        scores_all.append(vid_score)
        labels_all.append(label)
        subtypes_all.append(subtype)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(clip_stems)}")

    print(f"\nSkipped: {skipped} clips (missing RGB+Flow or unknown label)")
    print(f"Evaluated: {len(scores_all)} clips")

    # ── Overall AUC ─────────────────────────────────────────────────────────
    scores_np = np.array(scores_all)
    labels_np = np.array(labels_all)

    n_normal  = int((labels_np == 0).sum())
    n_violent = int((labels_np == 1).sum())
    print(f"Normal: {n_normal}  |  Violent: {n_violent}")

    if len(set(labels_all)) < 2:
        print("[ERROR] Only one class present — cannot compute AUC")
        return

    auc_overall = roc_auc_score(labels_np, scores_np)
    print(f"\nOverall AUC (zero-shot XD-Violence {args.split}): {auc_overall:.4f}")

    # ── Per-subtype AUC ─────────────────────────────────────────────────────
    print("\nPer-subtype AUC (anomalous subtype vs ALL Normal clips):")
    subtype_results = {}
    normal_mask = labels_np == 0

    subtypes_seen = sorted(set(s for s, l in zip(subtypes_all, labels_all) if l == 1))
    for sub in subtypes_seen:
        sub_mask  = np.array([s == sub for s in subtypes_all])
        sel_mask  = normal_mask | sub_mask
        if sel_mask.sum() == 0:
            continue
        sel_labels = labels_np[sel_mask]
        sel_scores = scores_np[sel_mask]
        if len(set(sel_labels.tolist())) < 2:
            continue
        sub_auc = roc_auc_score(sel_labels, sel_scores)
        n_sub   = int(sub_mask.sum())
        overlap = '★ overlap' if sub in OVERLAP_CATS else ''
        print(f"  {sub:<15s}: AUC={sub_auc:.4f}  n={n_sub}  {overlap}")
        subtype_results[sub] = (sub_auc, n_sub)

    # ── Overlap categories only ──────────────────────────────────────────────
    overlap_labels, overlap_scores = [], []
    for score, label, sub in zip(scores_all, labels_all, subtypes_all):
        if label == 0 or sub in OVERLAP_CATS:
            overlap_labels.append(label)
            overlap_scores.append(score)
    if len(set(overlap_labels)) == 2:
        auc_overlap = roc_auc_score(overlap_labels, overlap_scores)
        print(f"\nOverlap categories AUC (Fighting+Explosion+Shooting+Abuse vs Normal): "
              f"{auc_overlap:.4f}")
    else:
        auc_overlap = None

    # ── Score distribution ───────────────────────────────────────────────────
    print(f"\nScore stats:")
    print(f"  Normal  — mean={scores_np[labels_np==0].mean():.3f}  "
          f"max={scores_np[labels_np==0].max():.3f}")
    print(f"  Violent — mean={scores_np[labels_np==1].mean():.3f}  "
          f"max={scores_np[labels_np==1].max():.3f}")

    # ── Save results ─────────────────────────────────────────────────────────
    out_path = os.path.join(args.output_dir, f'xd_results_{args.split}.txt')
    with open(out_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("XD-Violence Zero-Shot Evaluation\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Split     : {args.split}\n")
        f.write(f"Clips     : {len(scores_all)} evaluated, {skipped} skipped\n")
        f.write(f"Normal    : {n_normal}  |  Violent: {n_violent}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Overall AUC: {auc_overall:.4f}\n")
        if auc_overlap is not None:
            f.write(f"Overlap categories AUC: {auc_overlap:.4f}\n")
        f.write("\nPer-subtype:\n")
        for sub, (auc, n) in subtype_results.items():
            overlap = '(overlap)' if sub in OVERLAP_CATS else ''
            f.write(f"  {sub:<15s}: {auc:.4f}  n={n}  {overlap}\n")
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',    required=True)
    parser.add_argument('--feature_dir',   required=True,
                        help='Path to i3d-features/ containing RGB/, Flow/, etc.')
    parser.add_argument('--output_dir',    default='./visualizations/xd_eval')
    parser.add_argument('--split',         default='train',
                        choices=['train', 'test'],
                        help='train uses RGB+Flow, test uses RGBTest+FlowTest')
    parser.add_argument('--num_segments',  type=int, default=32)
    args = parser.parse_args()
    main(args)
