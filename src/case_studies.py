"""
Case Studies: 5 Success + 5 Failure temporal score plots.

Runs inference on val set, selects:
  - Top-5 True Positives  (highest-scoring correct detections)
  - Top-5 False Positives (highest-scoring incorrect alarms)
  - Top-5 False Negatives (anomalous videos with lowest max score)
  - Top-5 True Negatives  (correct normal rejections, most confident)

Produces a multi-panel figure for the paper's case study section.

Usage:
    python src/case_studies.py \
        --checkpoint /path/to/best.pt \
        --feature_dir /path/to/UCF_Train_ten_crop_i3d \
        --list_file   /path/to/ucf-i3d-train.list \
        --output_dir  /path/to/output \
        --threshold   0.5
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ViolenceDetector
from dataset import make_train_val_split, InMemoryDataset

CAT_ID_TO_NAME = {
    0: 'Normal', 1: 'Abuse', 2: 'Fighting',
    3: 'Shooting', 4: 'Explosion', 5: 'Robbery', 6: 'Riot',
}

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
    print(f"Loaded checkpoint: epoch={ckpt.get('epoch','?')}  "
          f"vAUC={ckpt.get('metrics',{}).get('video_auc','?'):.4f}")
    return model


# ── Inference ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, dataset, device, threshold):
    results = []
    for i in range(len(dataset)):
        item       = dataset[i]
        feat       = item['features'].unsqueeze(0).to(device)
        label      = int(item['label'].item())
        cat_id     = int(item['cat_id'].item())

        out        = model(feat)
        trn_scores = out['trn_scores'].squeeze(0).cpu().numpy()
        mil_scores = out['mil_scores'].squeeze(0).cpu().numpy()
        boundary   = out['boundary'].squeeze(0).cpu().numpy()
        vid_score  = float(np.max(trn_scores))
        predicted  = 1 if vid_score >= threshold else 0
        cat_name   = CAT_ID_TO_NAME.get(cat_id, f'cat_{cat_id}')

        results.append({
            'idx':        i,
            'label':      label,
            'predicted':  predicted,
            'cat_name':   cat_name,
            'vid_score':  vid_score,
            'trn_scores': trn_scores,
            'mil_scores': mil_scores,
            'boundary':   boundary,
        })
    return results


# ── Panel plot ─────────────────────────────────────────────────────────────────
def plot_case_panel(ax, r, title_prefix, fps=25.0, frames_per_seg=16):
    T    = len(r['trn_scores'])
    secs = np.arange(T) * frames_per_seg / fps

    ax.plot(secs, r['trn_scores'], color='#CC2200', linewidth=1.8,
            label='TRN', zorder=3)
    ax.plot(secs, r['mil_scores'], color='#2266CC', linewidth=1.2,
            linestyle='--', alpha=0.65, label='MIL', zorder=2)
    ax.fill_between(secs, r['trn_scores'], alpha=0.12, color='#CC2200')
    ax.axhline(0.5, color='gray', linewidth=0.7, linestyle=':', alpha=0.6)

    # Shade high-score regions
    high = r['trn_scores'] >= 0.5
    if high.any():
        ax.fill_between(secs, 0, r['trn_scores'],
                        where=high, alpha=0.20, color='#CC2200',
                        label='Score≥0.5')

    ax.set_xlim(secs[0], secs[-1])
    ax.set_ylim(-0.05, 1.10)

    gt_str   = 'Anomalous' if r['label'] == 1 else 'Normal'
    pred_str = 'ANOM' if r['predicted'] == 1 else 'NORM'
    correct  = '✓' if r['label'] == r['predicted'] else '✗'
    ax.set_title(
        f"{title_prefix}: {r['cat_name']}\n"
        f"GT={gt_str}  Pred={pred_str}  Score={r['vid_score']:.3f}  {correct}",
        fontsize=8, fontweight='bold', pad=3
    )
    ax.set_xlabel('Time (s)', fontsize=7)
    ax.set_ylabel('Score', fontsize=7)
    ax.tick_params(labelsize=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.2)


# ── Main figure ────────────────────────────────────────────────────────────────
def plot_case_studies(tp5, fp5, fn5, tn5, output_path, fps=25.0, frames_per_seg=16):
    """
    4-row × 5-col figure:
      Row 0: True Positives  (correct detections — green border)
      Row 1: False Positives (false alarms — orange border)
      Row 2: False Negatives (missed violence — red border)
      Row 3: True Negatives  (correct normal — blue border)
    """
    groups = [
        ('TP — Correct Detections',    tp5, '#2DC653'),
        ('FP — False Alarms',          fp5, '#F4A261'),
        ('FN — Missed Violence',       fn5, '#CC2200'),
        ('TN — Correct Normal',        tn5, '#4E9AF1'),
    ]

    n_cols   = 5
    n_rows   = len(groups)
    fig_h    = n_rows * 3.2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.2, fig_h))

    row_labels = []
    for row_idx, (group_label, cases, color) in enumerate(groups):
        row_labels.append((row_idx, group_label, color))
        for col_idx in range(n_cols):
            ax = axes[row_idx][col_idx]
            if col_idx < len(cases):
                r = cases[col_idx]
                prefix = f"{group_label.split('—')[0].strip()}"
                plot_case_panel(ax, r, prefix, fps, frames_per_seg)
                # Color the spine
                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(1.5)
                if col_idx == 0:
                    ax.legend(fontsize=5, loc='upper right')
            else:
                ax.set_visible(False)

    # Row labels on left side
    for row_idx, group_label, color in row_labels:
        axes[row_idx][0].set_ylabel(
            group_label.split('—')[1].strip() + '\n\nScore',
            fontsize=7, color=color, fontweight='bold'
        )

    fig.suptitle('Case Studies: Success and Failure Analysis',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout(h_pad=1.5, w_pad=0.8)
    fig.savefig(output_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Separate success/failure panels ───────────────────────────────────────────
def plot_success_failure(tp5, fn5, fp5, output_dir, fps=25.0, frames_per_seg=16):
    """
    Two separate 1-row × 5-col figures for cleaner paper inclusion.
    """
    for cases, label, color, fname in [
        (tp5, 'Success Cases (True Positives)',  '#2DC653', 'success_cases.png'),
        (fn5, 'Failure Cases (False Negatives)', '#CC2200', 'failure_fn_cases.png'),
        (fp5, 'False Alarm Cases (False Positives)', '#F4A261', 'failure_fp_cases.png'),
    ]:
        n = min(5, len(cases))
        if n == 0:
            continue
        fig, axes = plt.subplots(1, n, figsize=(n * 3.5, 3.5))
        if n == 1:
            axes = [axes]

        for col, r in enumerate(cases[:n]):
            plot_case_panel(axes[col], r, '', fps, frames_per_seg)
            for spine in axes[col].spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(1.5)
            if col == 0:
                axes[col].legend(fontsize=6, loc='upper right')

        # Hide empty
        for col in range(n, len(axes)):
            axes[col].set_visible(False)

        fig.suptitle(label, fontsize=11, fontweight='bold', color=color, y=1.02)
        fig.tight_layout()
        out_path = os.path.join(output_dir, fname)
        fig.savefig(out_path, dpi=130, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = load_model(args.checkpoint, device)
    os.makedirs(args.output_dir, exist_ok=True)

    print("\nBuilding val dataset...")
    _, val_samples = make_train_val_split(
        args.list_file,
        os.path.expanduser(args.feature_dir),
        val_ratio=args.val_ratio,
        seed=42,
    )
    dataset = InMemoryDataset(
        val_samples,
        os.path.expanduser(args.feature_dir),
        num_segments=args.num_segments,
    )
    print(f"Val set: {len(dataset)} videos")

    print(f"\nRunning inference (threshold={args.threshold})...")
    results = run_inference(model, dataset, device, args.threshold)

    # Separate into groups
    tp = [r for r in results if r['label'] == 1 and r['predicted'] == 1]
    fp = [r for r in results if r['label'] == 0 and r['predicted'] == 1]
    fn = [r for r in results if r['label'] == 1 and r['predicted'] == 0]
    tn = [r for r in results if r['label'] == 0 and r['predicted'] == 0]

    print(f"TP={len(tp)}  FP={len(fp)}  FN={len(fn)}  TN={len(tn)}")

    # Select top-5 most illustrative cases
    tp5 = sorted(tp, key=lambda r: -r['vid_score'])[:5]            # highest confident TP
    fp5 = sorted(fp, key=lambda r: -r['vid_score'])[:5]            # worst FP (highest false score)
    fn5 = sorted(fn, key=lambda r:  r['vid_score'])[:5]            # worst FN (lowest missed score)
    tn5 = sorted(tn, key=lambda r:  r['vid_score'])[:5]            # most confident TN

    print("\nTop-5 True Positives:")
    for r in tp5:
        print(f"  [{r['cat_name']:12s}] score={r['vid_score']:.3f}")

    print("\nTop-5 False Positives (false alarms):")
    for r in fp5:
        print(f"  [{r['cat_name']:12s}] score={r['vid_score']:.3f}")

    print("\nTop-5 False Negatives (missed violence):")
    for r in fn5:
        print(f"  [{r['cat_name']:12s}] score={r['vid_score']:.3f}")

    print("\nTop-5 True Negatives:")
    for r in tn5:
        print(f"  [{r['cat_name']:12s}] score={r['vid_score']:.3f}")

    print("\nGenerating combined figure...")
    plot_case_studies(
        tp5, fp5, fn5, tn5,
        os.path.join(args.output_dir, 'case_studies_combined.png'),
        args.fps, args.frames_per_segment
    )

    print("Generating separate panels...")
    plot_success_failure(
        tp5, fn5, fp5,
        args.output_dir,
        args.fps, args.frames_per_segment
    )

    print("\nDone. Saved to:", args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',         required=True)
    parser.add_argument('--feature_dir',        required=True)
    parser.add_argument('--list_file',          required=True)
    parser.add_argument('--output_dir',         default='./visualizations/case_studies')
    parser.add_argument('--num_segments',       type=int,   default=32)
    parser.add_argument('--val_ratio',          type=float, default=0.2)
    parser.add_argument('--threshold',          type=float, default=0.5)
    parser.add_argument('--fps',                type=float, default=25.0)
    parser.add_argument('--frames_per_segment', type=int,   default=16)
    args = parser.parse_args()
    main(args)
