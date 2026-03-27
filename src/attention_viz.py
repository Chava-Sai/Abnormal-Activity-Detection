"""
TRN Self-Attention Heatmap Visualization for Violence Event Detection.

Extracts self-attention weights from each transformer layer of the TRN,
plots T×T heatmaps showing which segments attend to which, alongside
TRN anomaly scores.

Usage:
    python src/attention_viz.py \
        --checkpoint /path/to/best.pt \
        --feature_dir /path/to/UCF_Train_ten_crop_i3d \
        --output_dir  /path/to/output \
        --fps 25 --frames_per_segment 16
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
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ViolenceDetector

# ── Default videos (one per category) ─────────────────────────────────────────
DEFAULT_VIDEOS = [
    ('Fighting',   'Fighting002_x264_i3d.npy'),
    ('Robbery',    'Robbery001_x264_i3d.npy'),
    ('Shooting',   'Shooting001_x264_i3d.npy'),
    ('Explosion',  'Explosion001_x264_i3d.npy'),
    ('Abuse',      'Abuse001_x264_i3d.npy'),
    ('Normal',     'Normal_Videos001_x264_i3d.npy'),
]

# Custom colormap: white → deep red
ATTN_CMAP = LinearSegmentedColormap.from_list(
    'attn', ['#FFFFFF', '#FFF0F0', '#FF9999', '#CC0000', '#660000']
)


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


# ── Feature loading ────────────────────────────────────────────────────────────
def load_feature(path, num_segments=32):
    """Load .npy feature file → (1, T, 2048) tensor."""
    feat = np.load(path)                          # (T, 10, 2048) or (T, 2048)
    if feat.ndim == 3:
        feat = feat.mean(axis=1)                  # (T, 2048)
    # Resample to num_segments
    T = feat.shape[0]
    if T != num_segments:
        idx  = np.linspace(0, T - 1, num_segments).astype(int)
        feat = feat[idx]
    return torch.tensor(feat, dtype=torch.float32).unsqueeze(0)  # (1, T, 2048)


# ── Attention extraction ───────────────────────────────────────────────────────
@torch.no_grad()
def extract_attention_and_scores(model, feat_tensor, device):
    """
    Extract per-layer TRN self-attention weights and TRN scores.

    Returns:
        layer_attns : list of (T, T) arrays — one per transformer layer
        trn_scores  : (T,) array — refined anomaly scores
        mil_scores  : (T,) array — raw MIL scores
    """
    x   = feat_tensor.to(device)   # (1, T, 2048)
    trn = model.trn

    # Project input + positional encoding
    h = trn.pos_enc(trn.input_proj(x))   # (1, T, d_model)

    layer_attns = []
    for layer in trn.transformer.layers:
        # Extract attention weights from this layer's self-attention
        # need_weights=True, average_attn_weights=True → (1, T, T)
        _, attn_w = layer.self_attn(
            h, h, h,
            need_weights=True,
            average_attn_weights=True,
        )
        if attn_w is not None:
            layer_attns.append(attn_w.squeeze(0).cpu().numpy())  # (T, T)

        # Pass through the full layer to update h for next layer
        h = layer(h)

    # TRN scores from final hidden state
    trn_scores = trn.score_head(h).squeeze().cpu().numpy()     # (T,)

    # MIL scores
    mil_scores = model.mil_scorer(x).squeeze().cpu().numpy()   # (T,)

    return layer_attns, trn_scores, mil_scores


# ── Single video plot ──────────────────────────────────────────────────────────
def plot_video_attention(category, filename, layer_attns, trn_scores, mil_scores,
                         fps, frames_per_segment, output_dir):
    """
    Plot for one video:
    - Row 1: TRN scores + MIL scores over time
    - Row 2+: T×T attention heatmap per layer
    """
    n_layers = len(layer_attns)
    T        = len(trn_scores)
    secs     = np.arange(T) * frames_per_segment / fps

    fig = plt.figure(figsize=(14, 4 + 4 * n_layers))
    gs  = gridspec.GridSpec(n_layers + 1, 1, hspace=0.45)

    # ── Score plot ──────────────────────────────────────────────────────────
    ax_score = fig.add_subplot(gs[0])
    ax_score.plot(secs, trn_scores, color='#CC2200', linewidth=1.8,
                  label='TRN score', zorder=3)
    ax_score.plot(secs, mil_scores, color='#2266CC', linewidth=1.4,
                  linestyle='--', alpha=0.7, label='MIL score', zorder=2)
    ax_score.fill_between(secs, trn_scores, alpha=0.12, color='#CC2200')
    ax_score.axhline(0.5, color='gray', linewidth=0.8, linestyle=':', alpha=0.6)
    ax_score.set_xlim(secs[0], secs[-1])
    ax_score.set_ylim(-0.05, 1.05)
    ax_score.set_ylabel('Score', fontsize=9)
    ax_score.set_title(f'{category}  —  {filename}', fontsize=11, fontweight='bold')
    ax_score.legend(fontsize=8, loc='upper right')
    ax_score.tick_params(labelsize=8)
    ax_score.spines['top'].set_visible(False)
    ax_score.spines['right'].set_visible(False)
    ax_score.grid(True, alpha=0.2)

    # ── Attention heatmaps ──────────────────────────────────────────────────
    for li, attn in enumerate(layer_attns):
        ax = fig.add_subplot(gs[li + 1])
        im = ax.imshow(attn, aspect='auto', cmap=ATTN_CMAP,
                       interpolation='nearest', vmin=0.0, vmax=attn.max())

        # Overlay TRN score as white line on heatmap
        score_y = (1.0 - trn_scores) * (T - 1)   # invert: high score = top
        ax.plot(np.arange(T), score_y, color='white', linewidth=1.2,
                alpha=0.85, label='TRN score')

        ax.set_title(f'Layer {li+1} Self-Attention  (T×T = {T}×{T})',
                     fontsize=9, fontweight='bold')
        ax.set_xlabel('Key segment index', fontsize=8)
        ax.set_ylabel('Query segment', fontsize=8)
        ax.tick_params(labelsize=7)

        # Add second x-axis showing time in seconds
        ax2 = ax.twiny()
        tick_positions = np.linspace(0, T - 1, min(8, T))
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels([f'{secs[int(p)]:.0f}s' for p in tick_positions],
                            fontsize=7)
        ax2.tick_params(labelsize=7)

        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02).ax.tick_params(labelsize=7)

    fig.suptitle(
        f'TRN Self-Attention Heatmaps — {category}',
        fontsize=13, fontweight='bold', y=1.01
    )
    fig.tight_layout()

    safe_name = filename.replace('_x264_i3d.npy', '').replace(' ', '_')
    out_path  = os.path.join(output_dir, f'attn_{safe_name}.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Summary heatmap: average attention across all videos ──────────────────────
def plot_summary_heatmap(all_attns_per_layer, n_layers, T, output_dir):
    """
    Plot average T×T attention heatmap across all processed videos per layer.
    Shows the model's general temporal attention pattern.
    """
    fig, axes = plt.subplots(1, n_layers, figsize=(7 * n_layers, 6))
    if n_layers == 1:
        axes = [axes]

    for li in range(n_layers):
        avg_attn = np.mean(all_attns_per_layer[li], axis=0)   # (T, T)
        ax = axes[li]
        im = ax.imshow(avg_attn, aspect='auto', cmap=ATTN_CMAP,
                       interpolation='nearest')
        ax.set_title(f'Layer {li+1} — Average Attention\n(across all videos)',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Key segment', fontsize=9)
        ax.set_ylabel('Query segment', fontsize=9)
        ax.tick_params(labelsize=8)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02).ax.tick_params(labelsize=8)

        # Highlight diagonal (self-attention)
        diag = np.diag(np.diag(avg_attn))
        ax.contour(diag > 0, levels=[0.5], colors=['cyan'], linewidths=0.5, alpha=0.4)

    fig.suptitle('Average TRN Self-Attention — All Processed Videos',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    out_path = os.path.join(output_dir, 'attn_summary_avg.png')
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = load_model(args.checkpoint, device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine number of TRN layers
    n_layers = len(list(model.trn.transformer.layers))
    T        = args.num_segments
    print(f"TRN layers: {n_layers}  |  Segments: {T}\n")

    # Accumulate average attention per layer
    all_attns_per_layer = [[] for _ in range(n_layers)]

    # Process each video
    for category, fname in DEFAULT_VIDEOS:
        fpath = os.path.join(args.feature_dir, fname)
        if not os.path.exists(fpath):
            print(f"  [SKIP] Not found: {fpath}")
            continue

        print(f"Processing: {category} — {fname}")
        feat        = load_feature(fpath, args.num_segments)
        layer_attns, trn_scores, mil_scores = extract_attention_and_scores(
            model, feat, device
        )

        if not layer_attns:
            print(f"  [WARN] No attention weights returned — skipping plot")
            continue

        plot_video_attention(
            category, fname, layer_attns, trn_scores, mil_scores,
            args.fps, args.frames_per_segment, args.output_dir
        )

        for li, attn in enumerate(layer_attns):
            all_attns_per_layer[li].append(attn)

    # Summary plot
    if len(all_attns_per_layer[0]) > 0:
        print("\nGenerating summary heatmap...")
        plot_summary_heatmap(all_attns_per_layer, n_layers, T, args.output_dir)

    print("\nDone. All attention heatmaps saved to:", args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',        required=True)
    parser.add_argument('--feature_dir',       required=True)
    parser.add_argument('--output_dir',        default='./visualizations/attention')
    parser.add_argument('--num_segments',      type=int,   default=32)
    parser.add_argument('--fps',               type=float, default=25.0)
    parser.add_argument('--frames_per_segment',type=int,   default=16)
    args = parser.parse_args()
    main(args)
