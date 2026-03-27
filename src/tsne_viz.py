"""
t-SNE visualization of penultimate-layer features for Violence Event Detection.

Extracts the MIL scorer's penultimate-layer representations for all videos
in the validation set, projects to 2D with t-SNE, and plots colored by
ground-truth category.

Usage:
    python src/tsne_viz.py \
        --checkpoint /path/to/best.pt \
        --feature_dir /path/to/UCF_Train_ten_crop_i3d \
        --list_file   /path/to/ucf-i3d-train.list \
        --output_dir  /path/to/output \
        --num_segments 32
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ViolenceDetector
from dataset import UCFCrimeDataset, make_train_val_split, InMemoryDataset

# ── Category config ────────────────────────────────────────────────────────────
CAT_ID_TO_NAME = {
    0: 'Normal',
    1: 'Abuse',
    2: 'Fighting',
    3: 'Shooting',
    4: 'Explosion',
    5: 'Robbery',
    6: 'Riot',
}

# Colorblind-friendly palette
CATEGORY_COLORS = {
    'Normal':    '#4E9AF1',   # blue
    'Abuse':     '#E63946',   # red
    'Fighting':  '#F4A261',   # orange
    'Shooting':  '#2DC653',   # green
    'Explosion': '#9B5DE5',   # purple
    'Robbery':   '#F7B731',   # yellow
    'Riot':      '#FF6B6B',   # pink
}

MARKERS = {
    'Normal':    'o',
    'Abuse':     's',
    'Fighting':  '^',
    'Shooting':  'D',
    'Explosion': 'P',
    'Robbery':   'X',
    'Riot':      '*',
}


# ── Feature hook — extract penultimate MIL layer ───────────────────────────────
class PenultimateExtractor:
    """
    Hooks into the MIL scorer to extract the penultimate-layer representation.
    We use the output just before the final sigmoid — the 128-dim hidden rep.
    """
    def __init__(self, model: ViolenceDetector):
        self.features = []
        self._hook = None
        self._register(model)

    def _register(self, model):
        # MIL scorer is model.mil_scorer — find the last Linear before sigmoid
        mil = model.mil_scorer
        # Walk layers: find the last Linear layer
        layers = list(mil.children())
        # Find the second-to-last module (before the final activation)
        # mil_scorer structure: Linear→ReLU→Dropout→Linear→ReLU→Dropout→Linear→Sigmoid
        # We hook just before the last Linear
        target = None
        for layer in layers:
            if isinstance(layer, nn.Linear):
                target = layer
        # Hook the last-but-one ReLU (before final linear)
        # Instead: hook the output of the penultimate Linear (128-dim)
        # Find all Linears in order
        linears = [m for m in mil.modules() if isinstance(m, nn.Linear)]
        if len(linears) >= 2:
            penultimate_linear = linears[-2]   # second-to-last Linear
        else:
            penultimate_linear = linears[-1]

        def hook_fn(module, input, output):
            # output: (B, T, D) or (B, D) — take mean over T
            self.features.append(output.detach().cpu())

        self._hook = penultimate_linear.register_forward_hook(hook_fn)

    def clear(self):
        self.features = []

    def remove(self):
        if self._hook:
            self._hook.remove()


# ── Extract representations ────────────────────────────────────────────────────
@torch.no_grad()
def extract_representations(model, dataset, device, extractor):
    """
    Run inference on all videos and collect:
    - penultimate-layer feature (mean-pooled over segments) per video
    - category label per video
    """
    model.eval()
    reps   = []   # (N_videos, D)
    labels = []   # category name per video
    scores = []   # TRN anomaly score per video (for coloring intensity)

    for i in range(len(dataset)):
        item    = dataset[i]
        feat    = item['features'].unsqueeze(0).to(device)   # (1, T, 2048)
        cat_id  = item['cat_id'].item()
        label   = item['label'].item()

        extractor.clear()
        out = model(feat)

        if extractor.features:
            # penultimate output: (1, T, D) → mean over T → (D,)
            h = extractor.features[0]          # (1, T, D)
            if h.ndim == 3:
                h = h.mean(dim=1)              # (1, D)
            h = h.squeeze(0).numpy()           # (D,)
        else:
            # fallback: use TRN scores as representation
            h = out['trn_scores'].squeeze(0).cpu().numpy()

        cat_name = CAT_ID_TO_NAME.get(cat_id, f'cat_{cat_id}')
        trn_max  = float(out['trn_scores'].squeeze(0).max().cpu())

        reps.append(h)
        labels.append(cat_name)
        scores.append(trn_max)

        if (i + 1) % 50 == 0:
            print(f"  Extracted {i+1}/{len(dataset)}")

    return np.array(reps), labels, np.array(scores)


# ── t-SNE projection ───────────────────────────────────────────────────────────
def run_tsne(reps: np.ndarray, perplexity: int = 30,
             max_iter: int = 1000, seed: int = 42) -> np.ndarray:
    """Standardize features then run t-SNE. Returns (N, 2) embedding."""
    print(f"  Running t-SNE on {reps.shape} features "
          f"(perplexity={perplexity}, max_iter={max_iter})...")
    scaler   = StandardScaler()
    reps_std = scaler.fit_transform(reps)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=max_iter,
        random_state=seed,
        init='pca',
        learning_rate='auto',
    )
    embedding = tsne.fit_transform(reps_std)
    return embedding


# ── Plotting ───────────────────────────────────────────────────────────────────
def plot_tsne(embedding: np.ndarray, labels: list, scores: np.ndarray,
              output_path: str):
    """
    Produce two plots:
    1. Category-colored t-SNE (main paper figure)
    2. Score-intensity t-SNE (anomaly score heat map)
    """
    cats_present = sorted(set(labels))

    # ── Plot 1: Category colors ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))

    for cat in cats_present:
        idx   = [i for i, l in enumerate(labels) if l == cat]
        color = CATEGORY_COLORS.get(cat, '#888888')
        marker = MARKERS.get(cat, 'o')
        ax.scatter(
            embedding[idx, 0], embedding[idx, 1],
            c=color, marker=marker,
            s=55, alpha=0.80, linewidths=0.3, edgecolors='white',
            label=f'{cat} (n={len(idx)})',
            zorder=3,
        )

    ax.set_title('t-SNE of Penultimate MIL Features — by Category',
                 fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('t-SNE dim 1', fontsize=10)
    ax.set_ylabel('t-SNE dim 2', fontsize=10)
    ax.tick_params(labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.25, linewidth=0.5)

    legend = ax.legend(
        loc='upper right', fontsize=8, framealpha=0.9,
        markerscale=1.2, title='Category', title_fontsize=9,
    )

    fig.tight_layout()
    out1 = output_path.replace('.png', '_categories.png')
    fig.savefig(out1, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out1}")

    # ── Plot 2: Anomaly score heat map ───────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(9, 7))

    # Normal videos: fixed blue
    normal_idx = [i for i, l in enumerate(labels) if l == 'Normal']
    anom_idx   = [i for i, l in enumerate(labels) if l != 'Normal']

    ax2.scatter(
        embedding[normal_idx, 0], embedding[normal_idx, 1],
        c='#A8DADC', s=40, alpha=0.6, linewidths=0.2, edgecolors='white',
        label='Normal', zorder=2,
    )

    sc = ax2.scatter(
        embedding[anom_idx, 0], embedding[anom_idx, 1],
        c=scores[anom_idx], cmap='YlOrRd',
        vmin=0.3, vmax=1.0,
        s=60, alpha=0.85, linewidths=0.3, edgecolors='white',
        label='Anomalous', zorder=3,
    )

    cbar = fig2.colorbar(sc, ax=ax2, fraction=0.03, pad=0.02)
    cbar.set_label('TRN anomaly score', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax2.set_title('t-SNE — Anomaly Score Intensity',
                  fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel('t-SNE dim 1', fontsize=10)
    ax2.set_ylabel('t-SNE dim 2', fontsize=10)
    ax2.tick_params(labelsize=8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, alpha=0.25, linewidth=0.5)
    ax2.legend(loc='upper right', fontsize=8, framealpha=0.9)

    fig2.tight_layout()
    out2 = output_path.replace('.png', '_scores.png')
    fig2.savefig(out2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"  Saved: {out2}")

    # ── Plot 3: Combined 2-panel figure (for paper) ──────────────────────────
    fig3, (axL, axR) = plt.subplots(1, 2, figsize=(16, 6))

    for cat in cats_present:
        idx    = [i for i, l in enumerate(labels) if l == cat]
        color  = CATEGORY_COLORS.get(cat, '#888888')
        marker = MARKERS.get(cat, 'o')
        axL.scatter(embedding[idx, 0], embedding[idx, 1],
                    c=color, marker=marker, s=45, alpha=0.80,
                    linewidths=0.3, edgecolors='white',
                    label=f'{cat} (n={len(idx)})', zorder=3)

    axL.set_title('(a) Category Labels', fontsize=11, fontweight='bold')
    axL.set_xlabel('t-SNE dim 1', fontsize=9)
    axL.set_ylabel('t-SNE dim 2', fontsize=9)
    axL.tick_params(labelsize=7)
    axL.spines['top'].set_visible(False)
    axL.spines['right'].set_visible(False)
    axL.grid(True, alpha=0.2, linewidth=0.5)
    axL.legend(fontsize=7, framealpha=0.9, markerscale=1.1,
               loc='upper right', title='Category', title_fontsize=8)

    axR.scatter(embedding[normal_idx, 0], embedding[normal_idx, 1],
                c='#A8DADC', s=35, alpha=0.55, linewidths=0.2,
                edgecolors='white', label='Normal', zorder=2)
    sc2 = axR.scatter(embedding[anom_idx, 0], embedding[anom_idx, 1],
                      c=scores[anom_idx], cmap='YlOrRd',
                      vmin=0.3, vmax=1.0,
                      s=50, alpha=0.85, linewidths=0.3,
                      edgecolors='white', label='Anomalous', zorder=3)
    cb2 = fig3.colorbar(sc2, ax=axR, fraction=0.04, pad=0.02)
    cb2.set_label('TRN score', fontsize=8)
    cb2.ax.tick_params(labelsize=7)

    axR.set_title('(b) Anomaly Score Intensity', fontsize=11, fontweight='bold')
    axR.set_xlabel('t-SNE dim 1', fontsize=9)
    axR.set_ylabel('t-SNE dim 2', fontsize=9)
    axR.tick_params(labelsize=7)
    axR.spines['top'].set_visible(False)
    axR.spines['right'].set_visible(False)
    axR.grid(True, alpha=0.2, linewidth=0.5)
    axR.legend(fontsize=7, framealpha=0.9, loc='upper right')

    fig3.suptitle('t-SNE Visualization of Learned Feature Space — Violence Event Detection',
                  fontsize=12, fontweight='bold', y=1.01)
    fig3.tight_layout()
    out3 = output_path.replace('.png', '_combined.png')
    fig3.savefig(out3, dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f"  Saved: {out3}")


# ── Main ───────────────────────────────────────────────────────────────────────
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

    # Register penultimate layer hook
    extractor = PenultimateExtractor(model)

    # Build dataset — use val split for clean evaluation
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

    # Extract representations
    print("\nExtracting penultimate-layer features...")
    reps, labels, scores = extract_representations(model, dataset, device, extractor)
    extractor.remove()
    print(f"Feature matrix: {reps.shape}")

    # Print category distribution
    from collections import Counter
    dist = Counter(labels)
    print("\nCategory distribution:")
    for cat, count in sorted(dist.items()):
        print(f"  {cat:12s}: {count}")

    # t-SNE
    print("\nRunning t-SNE...")
    embedding = run_tsne(reps, perplexity=args.perplexity,
                         max_iter=args.n_iter, seed=42)

    # Save embedding for reuse
    os.makedirs(os.path.expanduser(args.output_dir), exist_ok=True)
    np.save(os.path.join(args.output_dir, 'tsne_embedding.npy'), embedding)
    np.save(os.path.join(args.output_dir, 'tsne_labels.npy'),    np.array(labels))
    np.save(os.path.join(args.output_dir, 'tsne_scores.npy'),    scores)
    print("  Embedding saved.")

    # Plot
    print("\nGenerating plots...")
    out_path = os.path.join(os.path.expanduser(args.output_dir), 'tsne.png')
    plot_tsne(embedding, labels, scores, out_path)

    print("\nDone.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',   required=True)
    parser.add_argument('--feature_dir',  required=True)
    parser.add_argument('--list_file',    required=True)
    parser.add_argument('--output_dir',   default='./visualizations/tsne')
    parser.add_argument('--num_segments', type=int, default=32)
    parser.add_argument('--val_ratio',    type=float, default=0.2)
    parser.add_argument('--perplexity',   type=int, default=30)
    parser.add_argument('--n_iter',       type=int, default=1000)
    args = parser.parse_args()
    main(args)
