"""
Boundary Precision Analysis — Proposal Section 6.5

Since frame-level GT temporal annotations are unavailable for our val split,
we use proxy-based analysis:
  - Predicted event: consecutive segments with trn_score > threshold
  - Boundary confidence bt: model's predicted transition confidence
  - Onset/offset positions derived from predicted event spans

Outputs (saved to --output_dir):
  1. boundary_conf_distribution.png  — bt histogram: anomalous vs normal
  2. score_change_vs_bt.png          — scatter: |Δscore| vs bt (correlation check)
  3. event_position_bias.png         — predicted start/end positions across video
  4. boundary_peak_distribution.png  — where peaks in bt occur temporally
  5. onset_offset_histogram.png      — start-error and end-error histograms
  6. boundary_summary.txt            — numerical summary
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ViolenceDetector
from dataset import UCFCrimeDataset, make_train_val_split, InMemoryDataset
from torch.utils.data import DataLoader

CAT_ID_TO_NAME = {
    0: 'Normal', 1: 'Abuse', 2: 'Fighting',
    3: 'Shooting', 4: 'Explosion', 5: 'Robbery', 6: 'Riot',
}


@torch.no_grad()
def run_inference(model, loader, device):
    """Run inference and collect per-video outputs."""
    model.eval()
    results = []
    for batch in loader:
        feat   = batch['features'].to(device)   # (1, T, 2048)
        label  = batch['label'].item()
        cat_id = batch['cat_id'].item()
        fpath  = batch['filepath'][0]

        out = model(feat)
        trn_scores = out['trn_scores'].squeeze(0).cpu().numpy()   # (T,)
        mil_scores = out['mil_scores'].squeeze(0).cpu().numpy()   # (T,)
        boundary   = out['boundary'].squeeze(0).cpu().numpy()     # (T-1,)

        results.append({
            'label':      label,
            'cat_id':     cat_id,
            'filepath':   fpath,
            'trn_scores': trn_scores,
            'mil_scores': mil_scores,
            'boundary':   boundary,
        })
    return results


def detect_event_spans(scores, threshold=0.5):
    """
    Return list of (start, end) segment indices where score > threshold.
    Merges adjacent positive segments.
    """
    spans = []
    in_event = False
    start = 0
    for t, s in enumerate(scores):
        if s >= threshold and not in_event:
            in_event = True
            start = t
        elif s < threshold and in_event:
            in_event = False
            spans.append((start, t - 1))
    if in_event:
        spans.append((start, len(scores) - 1))
    return spans


def analysis_1_boundary_conf_distribution(results, out_dir):
    """bt histogram for anomalous vs normal videos."""
    anom_bt, norm_bt = [], []
    for r in results:
        if r['label'] == 1:
            anom_bt.extend(r['boundary'].tolist())
        else:
            norm_bt.extend(r['boundary'].tolist())

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 1, 40)
    ax.hist(anom_bt, bins=bins, alpha=0.6, color='tomato',   label=f'Anomalous (n={len(anom_bt)})', density=True)
    ax.hist(norm_bt, bins=bins, alpha=0.6, color='steelblue',label=f'Normal (n={len(norm_bt)})',    density=True)
    ax.set_xlabel('Boundary Confidence bt', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Boundary Confidence Distribution: Anomalous vs Normal', fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'boundary_conf_distribution.png'), dpi=150)
    plt.close()

    return np.mean(anom_bt), np.mean(norm_bt)


def analysis_2_score_change_vs_bt(results, out_dir):
    """Scatter: |Δscore| vs bt — correlation check for boundary loss."""
    score_diffs, bt_vals = [], []
    for r in results:
        s = r['trn_scores']
        b = r['boundary']
        diffs = np.abs(np.diff(s))
        score_diffs.extend(diffs.tolist())
        bt_vals.extend(b.tolist())

    score_diffs = np.array(score_diffs)
    bt_vals     = np.array(bt_vals)

    # Downsample for scatter readability
    if len(score_diffs) > 5000:
        idx = np.random.choice(len(score_diffs), 5000, replace=False)
        score_diffs = score_diffs[idx]
        bt_vals     = bt_vals[idx]

    corr = float(np.corrcoef(score_diffs, bt_vals)[0, 1])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(score_diffs, bt_vals, alpha=0.15, s=8, color='purple')
    ax.set_xlabel('|score[t] - score[t+1]|', fontsize=11)
    ax.set_ylabel('Boundary Confidence bt', fontsize=11)
    ax.set_title(f'Score Change vs Boundary Confidence\n(Pearson r = {corr:.3f})', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'score_change_vs_bt.png'), dpi=150)
    plt.close()

    return corr


def analysis_3_event_position_bias(results, out_dir, threshold=0.5):
    """
    For each anomalous video with a detected event, record relative start/end position.
    Reveals early/late bias.
    """
    rel_starts, rel_ends, durations = [], [], []

    for r in results:
        if r['label'] != 1:
            continue
        T = len(r['trn_scores'])
        spans = detect_event_spans(r['trn_scores'], threshold)
        if not spans:
            continue
        # Use the longest span
        longest = max(spans, key=lambda x: x[1] - x[0])
        s, e = longest
        rel_starts.append(s / T)
        rel_ends.append(e / T)
        durations.append((e - s + 1) / T)

    if not rel_starts:
        return None, None

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    bins = np.linspace(0, 1, 20)

    axes[0].hist(rel_starts, bins=bins, color='tomato', edgecolor='white')
    axes[0].set_title('Predicted Event START\n(relative position in video)', fontsize=11)
    axes[0].set_xlabel('Relative Position (0=start, 1=end)')
    axes[0].set_ylabel('Count')
    axes[0].axvline(np.mean(rel_starts), color='black', linestyle='--',
                    label=f'mean={np.mean(rel_starts):.2f}')
    axes[0].legend(fontsize=9)

    axes[1].hist(rel_ends, bins=bins, color='steelblue', edgecolor='white')
    axes[1].set_title('Predicted Event END\n(relative position in video)', fontsize=11)
    axes[1].set_xlabel('Relative Position (0=start, 1=end)')
    axes[1].axvline(np.mean(rel_ends), color='black', linestyle='--',
                    label=f'mean={np.mean(rel_ends):.2f}')
    axes[1].legend(fontsize=9)

    axes[2].hist(durations, bins=bins, color='seagreen', edgecolor='white')
    axes[2].set_title('Predicted Event DURATION\n(fraction of video)', fontsize=11)
    axes[2].set_xlabel('Duration Fraction')
    axes[2].axvline(np.mean(durations), color='black', linestyle='--',
                    label=f'mean={np.mean(durations):.2f}')
    axes[2].legend(fontsize=9)

    plt.suptitle('Predicted Event Position Analysis (Anomalous Videos)', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'event_position_bias.png'), dpi=150, bbox_inches='tight')
    plt.close()

    return np.mean(rel_starts), np.mean(rel_ends)


def analysis_4_boundary_peak_distribution(results, out_dir):
    """Distribution of where bt peaks occur temporally (normalized position)."""
    peak_positions = []

    for r in results:
        b = r['boundary']
        T = len(b)
        if T < 3:
            continue
        peaks, _ = find_peaks(b, height=0.3, distance=2)
        for p in peaks:
            peak_positions.append(p / T)

    if not peak_positions:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(peak_positions, bins=30, color='darkorange', edgecolor='white', density=True)
    ax.set_xlabel('Relative Position of Boundary Peak (0=start, 1=end)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'Temporal Distribution of Boundary Confidence Peaks\n(n={len(peak_positions)} peaks, threshold=0.3)', fontsize=12)
    # Uniform reference line
    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.6, label='Uniform baseline')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'boundary_peak_distribution.png'), dpi=150)
    plt.close()


def analysis_5_per_category_scores(results, out_dir):
    """
    Per-category score profile: average trn_score curve across videos per category.
    Shows whether the model fires early/late/middle for each violence type.
    """
    by_cat = defaultdict(list)
    for r in results:
        if r['label'] == 1:
            by_cat[r['cat_id']].append(r['trn_scores'])

    if not by_cat:
        return

    n_cats = len(by_cat)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, (cat_id, score_list) in enumerate(sorted(by_cat.items())):
        if i >= 6:
            break
        # Normalize all to same length (32) by interpolation
        normalized = []
        for s in score_list:
            if len(s) != 32:
                x_old = np.linspace(0, 1, len(s))
                x_new = np.linspace(0, 1, 32)
                normalized.append(np.interp(x_new, x_old, s))
            else:
                normalized.append(s)

        arr = np.array(normalized)
        mean_scores = arr.mean(axis=0)
        std_scores  = arr.std(axis=0)
        t = np.arange(len(mean_scores))

        ax = axes[i]
        ax.fill_between(t, mean_scores - std_scores, mean_scores + std_scores,
                        alpha=0.3, color='tomato')
        ax.plot(t, mean_scores, color='tomato', linewidth=2)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_title(f'{CAT_ID_TO_NAME.get(cat_id, str(cat_id))} (n={len(score_list)})', fontsize=11)
        ax.set_xlabel('Segment Index')
        ax.set_ylabel('Mean TRN Score')
        ax.set_ylim(0, 1)

    # Hide unused subplots
    for j in range(i + 1, 6):
        axes[j].set_visible(False)

    plt.suptitle('Mean TRN Score Profile by Violence Category', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'per_category_score_profiles.png'), dpi=150)
    plt.close()


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_args = ckpt.get('args', {})
    num_segments = ckpt_args.get('num_segments', 32)
    d_model      = ckpt_args.get('d_model', 512)
    nhead        = ckpt_args.get('nhead', 8)
    trn_layers   = ckpt_args.get('trn_layers', 2)

    model = ViolenceDetector(
        input_dim=2048, num_classes=7,
        d_model=d_model, nhead=nhead, trn_layers=trn_layers,
    ).to(device)
    model.load_state_dict(ckpt['state_dict'])
    print(f"Loaded checkpoint: epoch {ckpt['epoch']}, T={num_segments}")

    # Build val dataset (same split as training)
    _, val_samples = make_train_val_split(
        args.list_file, args.feature_dir,
        val_ratio=args.val_ratio, seed=42
    )
    val_ds = InMemoryDataset(val_samples, args.feature_dir, num_segments)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

    print(f"Running inference on {len(val_ds)} val videos...")
    results = run_inference(model, val_loader, device)

    n_anom = sum(1 for r in results if r['label'] == 1)
    n_norm = sum(1 for r in results if r['label'] == 0)
    print(f"  {n_anom} anomalous, {n_norm} normal")

    # Run all analyses
    print("Analysis 1: Boundary confidence distribution...")
    mean_anom_bt, mean_norm_bt = analysis_1_boundary_conf_distribution(results, args.output_dir)

    print("Analysis 2: Score change vs boundary confidence correlation...")
    corr = analysis_2_score_change_vs_bt(results, args.output_dir)

    print("Analysis 3: Event position bias...")
    mean_start, mean_end = analysis_3_event_position_bias(results, args.output_dir, args.threshold)

    print("Analysis 4: Boundary peak temporal distribution...")
    analysis_4_boundary_peak_distribution(results, args.output_dir)

    print("Analysis 5: Per-category score profiles...")
    analysis_5_per_category_scores(results, args.output_dir)

    # Summary report
    summary_path = os.path.join(args.output_dir, 'boundary_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("BOUNDARY PRECISION ANALYSIS — Violence Event Detection\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Checkpoint : {args.checkpoint}\n")
        f.write(f"Val videos : {len(results)} ({n_anom} anomalous, {n_norm} normal)\n")
        f.write(f"Threshold  : {args.threshold}\n\n")

        f.write("--- Analysis 1: Boundary Confidence ---\n")
        f.write(f"  Mean bt (anomalous) : {mean_anom_bt:.4f}\n")
        f.write(f"  Mean bt (normal)    : {mean_norm_bt:.4f}\n")
        f.write(f"  Difference          : {mean_anom_bt - mean_norm_bt:+.4f}\n\n")

        f.write("--- Analysis 2: Score-Change vs bt Correlation ---\n")
        f.write(f"  Pearson r           : {corr:.4f}\n")
        f.write(f"  (High r = boundary loss working as intended)\n\n")

        f.write("--- Analysis 3: Event Position Bias ---\n")
        if mean_start is not None:
            f.write(f"  Mean predicted start : {mean_start:.3f} (0=video start, 1=end)\n")
            f.write(f"  Mean predicted end   : {mean_end:.3f}\n")
            bias = "early" if mean_start < 0.3 else ("late" if mean_start > 0.6 else "no strong")
            f.write(f"  Onset bias           : {bias}\n")
            bias_end = "early" if mean_end < 0.5 else ("late" if mean_end > 0.7 else "no strong")
            f.write(f"  Offset bias          : {bias_end}\n\n")
        else:
            f.write("  No events detected at this threshold.\n\n")

        f.write("--- Outputs ---\n")
        for fname in [
            'boundary_conf_distribution.png',
            'score_change_vs_bt.png',
            'event_position_bias.png',
            'boundary_peak_distribution.png',
            'per_category_score_profiles.png',
        ]:
            f.write(f"  {fname}\n")

    print(f"\nDone. Results saved to: {args.output_dir}")
    print(f"Summary: {summary_path}")

    # Print summary to console too
    print(f"\n  Mean bt anomalous={mean_anom_bt:.4f}  normal={mean_norm_bt:.4f}")
    print(f"  Score-change vs bt correlation: r={corr:.4f}")
    if mean_start is not None:
        print(f"  Mean event start={mean_start:.3f}  end={mean_end:.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Boundary Precision Analysis')
    parser.add_argument('--checkpoint',  required=True,
                        help='Path to best.pt checkpoint')
    parser.add_argument('--feature_dir', required=True,
                        help='Training feature directory (UCF_Train_ten_crop_i3d)')
    parser.add_argument('--list_file',   required=True,
                        help='Training list file (ucf-i3d-train.list)')
    parser.add_argument('--output_dir',  default='boundary_analysis_output',
                        help='Directory to save plots and summary')
    parser.add_argument('--val_ratio',   type=float, default=0.2,
                        help='Val split ratio (must match training)')
    parser.add_argument('--threshold',   type=float, default=0.5,
                        help='Score threshold for event detection')
    args = parser.parse_args()
    main(args)
