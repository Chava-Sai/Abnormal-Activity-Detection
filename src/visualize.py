"""
Temporal anomaly score visualization for Violence Event Detection.

Plots MIL score, TRN score, and boundary confidence over time for
selected videos. Produces a publication-quality figure for the paper.

Usage:
    python src/visualize.py \
        --checkpoint /path/to/best.pt \
        --feature_dir /path/to/features \
        --output_dir  /path/to/output \
        --fps 25 \
        --frames_per_segment 16
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ViolenceDetector

# ── Category config ────────────────────────────────────────────────────────────
CAT_ID_TO_NAME = {0: 'Normal', 1: 'Abuse', 2: 'Fighting',
                  3: 'Shooting', 4: 'Explosion', 5: 'Robbery', 6: 'Riot'}

# Videos to visualize: one per violence category + one normal
DEFAULT_VIDEOS = [
    ('Fighting',   'Fighting002_x264_i3d.npy'),
    ('Robbery',    'Robbery001_x264_i3d.npy'),
    ('Shooting',   'Shooting001_x264_i3d.npy'),
    ('Explosion',  'Explosion001_x264_i3d.npy'),
    ('Abuse',      'Abuse001_x264_i3d.npy'),
    ('Normal',     'Normal_Videos001_x264_i3d.npy'),
]

COLORS = {
    'trn':      '#E63946',   # red   — TRN (primary)
    'mil':      '#457B9D',   # blue  — MIL baseline
    'boundary': '#2DC653',   # green — boundary confidence
    'anomaly':  '#FFBA08',   # gold  — anomaly region shade
    'normal':   '#A8DADC',   # light blue — normal region shade
}


# ── Inference ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def infer_video(model, feature_path: str, num_segments: int, device) -> dict:
    """Load one video's features and run full model inference."""
    feat = np.load(feature_path)
    if feat.ndim == 3:
        feat = feat.mean(axis=1)          # (T, 10, 2048) → (T, 2048)

    T = feat.shape[0]
    N = num_segments

    # Temporal resize
    if T > N:
        idx  = np.linspace(0, T - 1, N, dtype=int)
        feat = feat[idx]
    elif T < N:
        pad  = np.zeros((N - T, feat.shape[1]), dtype=np.float32)
        feat = np.concatenate([feat, pad], axis=0)

    x   = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)
    out = model(x)

    trn_scores = out['trn_scores'].squeeze(0).cpu().numpy()       # (N,)
    mil_scores = out['mil_scores'].squeeze(0).cpu().numpy()       # (N,)
    cls_logits = out['cls_logits'].squeeze(0).cpu().numpy()       # (N, 7)
    boundary   = out['boundary'].squeeze(0).cpu().numpy()         # (N,)

    # Predicted category = argmax of top-k segment mean
    k         = max(1, int(N * 0.1))
    top_idx   = np.argsort(trn_scores)[-k:]
    pred_cat  = int(cls_logits[top_idx].mean(axis=0).argmax())

    return {
        'trn_scores': trn_scores,
        'mil_scores': mil_scores,
        'boundary':   boundary,
        'pred_cat':   pred_cat,
        'n_segments': N,
    }


# ── Per-video plot ─────────────────────────────────────────────────────────────
def plot_single(ax, result: dict, title: str, is_anomalous: bool,
                fps: int = 25, frames_per_seg: int = 16):
    """Draw temporal score curves on a single axes."""
    N    = result['n_segments']
    secs = np.arange(N) * frames_per_seg / fps   # segment start time in seconds

    # Background shading
    bg_color = COLORS['anomaly'] if is_anomalous else COLORS['normal']
    ax.axhspan(0, 1, alpha=0.08, color=bg_color, zorder=0)

    # Decision threshold line
    ax.axhline(0.5, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)

    # Score curves
    ax.plot(secs, result['mil_scores'], color=COLORS['mil'],
            linewidth=1.5, label='MIL score', alpha=0.85)
    ax.plot(secs, result['trn_scores'], color=COLORS['trn'],
            linewidth=2.0, label='TRN score', alpha=0.95)
    # Boundary may be N-1 (between-segment transitions) — pad to match N
    boundary = result['boundary']
    if len(boundary) < len(secs):
        boundary = np.append(boundary, boundary[-1])
    ax.plot(secs, boundary,  color=COLORS['boundary'],
            linewidth=1.2, label='Boundary', alpha=0.80, linestyle=':')

    # Predicted category annotation
    pred_name = CAT_ID_TO_NAME.get(result['pred_cat'], '?')
    max_score = result['trn_scores'].max()
    ax.text(0.98, 0.95, f"pred: {pred_name}\nscore: {max_score:.2f}",
            transform=ax.transAxes, fontsize=7, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='gray', alpha=0.8))

    ax.set_title(title, fontsize=9, fontweight='bold', pad=4)
    ax.set_xlim(secs[0], secs[-1])
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('Time (s)', fontsize=7)
    ax.set_ylabel('Score', fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ── Main figure ────────────────────────────────────────────────────────────────
def make_figure(model, feature_dir: str, video_list: list,
                output_dir: str, num_segments: int,
                fps: int, frames_per_seg: int, device):
    """Create a 2×3 grid figure with one panel per video."""
    os.makedirs(output_dir, exist_ok=True)

    # Filter to files that actually exist
    valid = []
    for cat, fname in video_list:
        fpath = os.path.join(feature_dir, fname)
        if os.path.exists(fpath):
            valid.append((cat, fname, fpath))
        else:
            print(f"  [skip] not found: {fname}")

    if not valid:
        print("No valid video files found. Exiting.")
        return

    n     = len(valid)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig = plt.figure(figsize=(14, 4.5 * nrows))
    fig.suptitle('Temporal Anomaly Score Profiles — Violence Event Detection',
                 fontsize=12, fontweight='bold', y=1.01)
    gs  = GridSpec(nrows, ncols, figure=fig, hspace=0.55, wspace=0.35)

    for i, (cat, fname, fpath) in enumerate(valid):
        print(f"  Running inference: {fname}")
        result      = infer_video(model, fpath, num_segments, device)
        is_anomalous = (cat != 'Normal')
        title        = f"{cat}\n({fname.replace('_x264_i3d.npy','')})"

        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        plot_single(ax, result, title, is_anomalous, fps, frames_per_seg)

    # Legend (shared)
    legend_elements = [
        mpatches.Patch(color=COLORS['trn'],      label='TRN score (primary)'),
        mpatches.Patch(color=COLORS['mil'],      label='MIL score (baseline)'),
        mpatches.Patch(color=COLORS['boundary'], label='Boundary confidence'),
        mpatches.Patch(color=COLORS['anomaly'],  alpha=0.3, label='Anomalous video'),
        mpatches.Patch(color=COLORS['normal'],   alpha=0.3, label='Normal video'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5,
               fontsize=8, bbox_to_anchor=(0.5, -0.03),
               frameon=True, framealpha=0.9)

    out_path = os.path.join(output_dir, 'temporal_scores.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {out_path}")
    plt.close(fig)

    # Also save per-video individual plots
    for cat, fname, fpath in valid:
        result      = infer_video(model, fpath, num_segments, device)
        is_anomalous = (cat != 'Normal')

        fig2, ax2 = plt.subplots(figsize=(8, 3))
        title      = f"{cat} — {fname.replace('_x264_i3d.npy','')}"
        plot_single(ax2, result, title, is_anomalous, fps, frames_per_seg)
        ax2.legend(loc='upper left', fontsize=7, framealpha=0.8)

        out2 = os.path.join(output_dir, fname.replace('_x264_i3d.npy', '_scores.png'))
        fig2.savefig(out2, dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print(f"  Saved: {out2}")


# ── CLI ────────────────────────────────────────────────────────────────────────
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    ckpt       = torch.load(args.checkpoint, map_location=device)
    model_args = ckpt.get('args', {})
    model      = ViolenceDetector(
        input_dim=2048, num_classes=7,
        d_model=model_args.get('d_model', 512),
        nhead=model_args.get('nhead', 8),
        trn_layers=model_args.get('trn_layers', 2),
    ).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print(f"Loaded checkpoint: epoch={ckpt.get('epoch','?')}")

    # Build video list
    if args.videos:
        video_list = []
        for v in args.videos:
            fname = os.path.basename(v)
            # Infer category from filename
            import re
            base = re.sub(r'_x264_i3d\.npy$', '', fname)
            cat  = re.sub(r'_?\d+$', '', base)
            if cat.startswith('Normal'):
                cat = 'Normal'
            video_list.append((cat, fname))
    else:
        video_list = DEFAULT_VIDEOS

    print(f"\nVisualizing {len(video_list)} videos...")
    make_figure(
        model       = model,
        feature_dir = os.path.expanduser(args.feature_dir),
        video_list  = video_list,
        output_dir  = os.path.expanduser(args.output_dir),
        num_segments= args.num_segments,
        fps         = args.fps,
        frames_per_seg = args.frames_per_segment,
        device      = device,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',         required=True)
    parser.add_argument('--feature_dir',        required=True)
    parser.add_argument('--output_dir',         default='./visualizations')
    parser.add_argument('--videos',             nargs='+', default=None,
                        help='Specific feature filenames (default: one per category)')
    parser.add_argument('--num_segments',       type=int, default=32)
    parser.add_argument('--fps',                type=int, default=25)
    parser.add_argument('--frames_per_segment', type=int, default=16)
    args = parser.parse_args()
    main(args)
