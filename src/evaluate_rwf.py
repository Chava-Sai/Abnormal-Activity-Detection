"""
RWF-2000 Evaluation Script.

Loads I3D features extracted by extract_i3d_features.ipynb,
runs inference with the UCF-Crime trained model, and reports:
  - Binary AUC (Fight vs NonFight)
  - F1, Precision, Recall for Fight class
  - Accuracy

List file format (one line per video):
    {filename}_i3d.npy {label}
    label: 1 = Fight, 0 = NonFight

Usage:
    python src/evaluate_rwf.py \
        --checkpoint $PROJ/experiments/run_v4/checkpoints/best.pt \
        --feature_dir $PROJ/datasets/rwf2000_i3d \
        --list_file   $PROJ/datasets/rwf2000_i3d/rwf2000.list \
        --num_segments 32
"""

import os
import sys
import argparse
import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, accuracy_score,
    classification_report
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ViolenceDetector


def load_and_resample(npy_path, num_segments):
    """Load .npy features and resample to num_segments via linear interpolation."""
    feat = np.load(npy_path).astype(np.float32)  # (T, 2048)
    T = feat.shape[0]
    if T == num_segments:
        return feat
    # Resample along time axis
    idx = np.linspace(0, T - 1, num_segments)
    lo  = np.floor(idx).astype(int)
    hi  = np.minimum(lo + 1, T - 1)
    w   = (idx - lo)[:, None]
    return feat[lo] * (1 - w) + feat[hi] * w  # (num_segments, 2048)


@torch.no_grad()
def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    ckpt_args = ckpt['args']

    model = ViolenceDetector(
        input_dim=getattr(ckpt_args, 'input_dim', 2048),
        d_model=getattr(ckpt_args, 'd_model', 512),
        nhead=getattr(ckpt_args, 'nhead', 8),
        trn_layers=getattr(ckpt_args, 'trn_layers', 2),
        num_classes=getattr(ckpt_args, 'num_classes', 7),
        dropout_mil=getattr(ckpt_args, 'dropout_mil', 0.5),
        dropout_cls=getattr(ckpt_args, 'dropout_cls', 0.3),
        dropout_trn=getattr(ckpt_args, 'dropout_trn', 0.1),
    )
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)
    model.eval()
    print(f'Loaded checkpoint: epoch={ckpt["epoch"]}, '
          f'UCF vAUC={ckpt["metrics"].get("video_auc", "?"):.4f}')

    # Load list file
    lines = open(args.list_file).read().strip().splitlines()
    print(f'Videos in list: {len(lines)}')

    scores, labels, skipped = [], [], 0

    for line in lines:
        parts = line.strip().split()
        fname, label = parts[0], int(parts[1])
        fpath = os.path.join(args.feature_dir, fname)

        if not os.path.exists(fpath):
            skipped += 1
            continue

        feat = load_and_resample(fpath, args.num_segments)  # (T, 2048)
        x = torch.from_numpy(feat).unsqueeze(0).to(device)  # (1, T, 2048)

        out = model(x)
        # Use max TRN score as video-level anomaly score
        trn = out['trn_scores'].squeeze(0).cpu().numpy()  # (T,)
        mil = out['mil_scores'].squeeze(0).cpu().numpy()  # (T,)
        score = float(max(trn.max(), mil.max()))

        scores.append(score)
        labels.append(label)

    print(f'Evaluated: {len(scores)} videos  |  Skipped: {skipped}')

    scores = np.array(scores)
    labels = np.array(labels)

    # AUC
    auc = roc_auc_score(labels, scores)

    # Binary predictions at threshold 0.5
    preds = (scores >= 0.5).astype(int)
    acc  = accuracy_score(labels, preds)
    f1   = f1_score(labels, preds, pos_label=1)
    prec = precision_score(labels, preds, pos_label=1, zero_division=0)
    rec  = recall_score(labels, preds, pos_label=1, zero_division=0)

    print('\n' + '='*55)
    print('RWF-2000 Evaluation Results')
    print('='*55)
    print(f'  AUC (Fight vs NonFight) : {auc:.4f}')
    print(f'  Accuracy                : {acc:.4f}')
    print(f'  Fight F1                : {f1:.4f}')
    print(f'  Fight Precision         : {prec:.4f}')
    print(f'  Fight Recall            : {rec:.4f}')
    print()
    print('Per-class report:')
    print(classification_report(labels, preds,
                                target_names=['NonFight', 'Fight'],
                                digits=4))

    # Label distribution
    n_fight    = labels.sum()
    n_nonfight = len(labels) - n_fight
    print(f'Label distribution: Fight={n_fight}, NonFight={n_nonfight}')
    print('='*55)

    return auc, f1, acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',   required=True)
    parser.add_argument('--feature_dir',  required=True)
    parser.add_argument('--list_file',    required=True)
    parser.add_argument('--num_segments', type=int, default=32)
    args = parser.parse_args()
    evaluate(args)
