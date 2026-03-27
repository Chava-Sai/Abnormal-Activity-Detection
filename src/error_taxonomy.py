"""
Error Taxonomy Analysis for Violence Event Detection.

Runs inference on the entire validation set, identifies False Positives (FP)
and False Negatives (FN) at video level, then categorizes failure modes.

Output:
  - error_taxonomy.csv     — full FP/FN list with scores + categories
  - error_summary.txt      — grouped failure mode counts
  - fp_cases.png           — TRN score plots for top-50 worst FP
  - fn_cases.png           — TRN score plots for top-50 worst FN

Usage:
    python src/error_taxonomy.py \
        --checkpoint /path/to/best.pt \
        --feature_dir /path/to/UCF_Train_ten_crop_i3d \
        --list_file   /path/to/ucf-i3d-train.list \
        --output_dir  /path/to/output \
        --threshold   0.5
"""

import os
import sys
import csv
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ViolenceDetector
from dataset import UCFCrimeDataset, make_train_val_split, InMemoryDataset


# ── Category map ───────────────────────────────────────────────────────────────
CAT_ID_TO_NAME = {
    0: 'Normal',
    1: 'Abuse',
    2: 'Fighting',
    3: 'Shooting',
    4: 'Explosion',
    5: 'Robbery',
    6: 'Riot',
}

# Anomalous categories (non-Normal)
ANOMALOUS_CATS = {'Abuse', 'Fighting', 'Shooting', 'Explosion', 'Robbery', 'Riot'}


# ── Failure mode taxonomy ──────────────────────────────────────────────────────
def classify_fp_mode(trn_scores, mil_scores, cat_name, video_name):
    """
    Classify a False Positive (Normal predicted as Anomalous) into a failure mode.

    Returns (mode_code, description)
    """
    max_trn  = float(np.max(trn_scores))
    mean_trn = float(np.mean(trn_scores))
    peak_idx = int(np.argmax(trn_scores))
    T        = len(trn_scores)

    # Persistent high score throughout video
    if mean_trn > 0.55:
        return 'FP_PERSISTENT', 'Uniformly high score throughout — likely scene mismatch'

    # Single brief spike
    spike_len = int(np.sum(trn_scores > 0.5))
    if spike_len <= 2:
        return 'FP_SPIKE', 'Single-segment spike — camera artifact / abrupt cut'

    # Score concentrated at boundaries
    boundary_zone = int(T * 0.15)
    in_boundary = (peak_idx < boundary_zone) or (peak_idx > T - boundary_zone)
    if in_boundary:
        return 'FP_BOUNDARY', 'Peak near video boundary — lighting/scene transition'

    # Gradual rise — motion ambiguity
    rising = np.all(np.diff(trn_scores[:peak_idx + 1]) >= -0.05) and peak_idx > T // 2
    if rising:
        return 'FP_MOTION', 'Gradually rising score — energetic but non-violent motion'

    # MIL and TRN disagree
    max_mil = float(np.max(mil_scores))
    if max_trn > 0.5 and max_mil < 0.35:
        return 'FP_TRN_OVERFIT', 'TRN fires but MIL does not — TRN temporal over-generalization'

    return 'FP_OTHER', 'Unclassified false positive'


def classify_fn_mode(trn_scores, mil_scores, cat_name, video_name):
    """
    Classify a False Negative (Anomalous predicted as Normal) into a failure mode.

    Returns (mode_code, description)
    """
    max_trn  = float(np.max(trn_scores))
    mean_trn = float(np.mean(trn_scores))
    T        = len(trn_scores)

    # Very low scores across board — model blind to this event type
    if max_trn < 0.3:
        return 'FN_UNSEEN', f'Very low scores — {cat_name} features not learned (Riot-like)'

    # Score just below threshold — marginal miss
    if 0.3 <= max_trn < 0.5:
        return 'FN_MARGINAL', 'Score just below threshold — marginal decision boundary miss'

    # High MIL, low TRN — TRN suppressed valid signal
    max_mil = float(np.max(mil_scores))
    if max_mil > 0.5 and max_trn < 0.5:
        return 'FN_TRN_SUPPRESSED', 'MIL fires but TRN suppresses — temporal smoothing too aggressive'

    # Short event in long video — diluted by normal segments
    high_seg = int(np.sum(trn_scores > 0.4))
    if high_seg <= int(T * 0.1):
        return 'FN_SHORT_EVENT', 'Violence occupies <10% of video — diluted by normal context'

    # Score concentrated in a region but still below threshold
    if max_trn >= 0.4:
        return 'FN_WEAK_SIGNAL', 'Moderate scores present but peak below threshold — subtle violence'

    return 'FN_OTHER', 'Unclassified false negative'


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
    print(f"Loaded checkpoint: epoch={epoch}  vAUC={v_auc:.4f}")
    return model


# ── Inference on val set ───────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, dataset, device, threshold):
    """
    Run inference on every video in dataset.
    Returns a list of dicts with per-video results.
    """
    results = []
    for i in range(len(dataset)):
        item    = dataset[i]
        feat    = item['features'].unsqueeze(0).to(device)  # (1, T, 2048)
        label   = int(item['label'].item())                  # 1=anomalous, 0=normal
        cat_id  = int(item['cat_id'].item())
        vid_id  = item.get('video_id', f'video_{i}')
        if isinstance(vid_id, torch.Tensor):
            vid_id = vid_id.item()

        out        = model(feat)
        trn_scores = out['trn_scores'].squeeze(0).cpu().numpy()   # (T,)
        mil_scores = out['mil_scores'].squeeze(0).cpu().numpy()   # (T,)
        cls_logits = out['cls_logits'].squeeze(0).cpu().numpy()   # (T, 7)

        vid_score  = float(np.max(trn_scores))
        predicted  = 1 if vid_score >= threshold else 0
        cat_name   = CAT_ID_TO_NAME.get(cat_id, f'cat_{cat_id}')
        pred_cls   = int(np.argmax(cls_logits.mean(axis=0)))  # majority class

        results.append({
            'idx':         i,
            'cat_id':      cat_id,
            'cat_name':    cat_name,
            'label':       label,           # ground-truth (1=anom, 0=normal)
            'predicted':   predicted,
            'vid_score':   vid_score,
            'trn_scores':  trn_scores,
            'mil_scores':  mil_scores,
            'pred_cls':    CAT_ID_TO_NAME.get(pred_cls, f'cls_{pred_cls}'),
        })

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(dataset)}")

    return results


# ── Score plot for error cases ─────────────────────────────────────────────────
def plot_error_cases(cases, error_type, output_path, max_cases=50):
    """
    Plot TRN score timelines for up to max_cases error videos.
    Sorts by worst-case score (highest for FP, lowest for FN).
    """
    if not cases:
        print(f"  No {error_type} cases found.")
        return

    # Sort
    if error_type == 'FP':
        cases_sorted = sorted(cases, key=lambda r: -r['vid_score'])
    else:
        cases_sorted = sorted(cases, key=lambda r: r['vid_score'])

    cases_sorted = cases_sorted[:max_cases]
    n = len(cases_sorted)

    ncols = 5
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 2.4))
    axes = np.array(axes).reshape(-1) if nrows > 1 else np.array([axes]).reshape(-1)

    for i, r in enumerate(cases_sorted):
        ax  = axes[i]
        T   = len(r['trn_scores'])
        t   = np.arange(T)
        ax.plot(t, r['trn_scores'], color='#CC2200', linewidth=1.5, label='TRN')
        ax.plot(t, r['mil_scores'], color='#2266CC', linewidth=1.0,
                linestyle='--', alpha=0.6, label='MIL')
        ax.axhline(0.5, color='gray', linewidth=0.7, linestyle=':', alpha=0.7)
        ax.fill_between(t, r['trn_scores'], alpha=0.1, color='#CC2200')
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, T - 1)
        ax.set_title(
            f"{r['cat_name']}  [{r['mode_code']}]\nscore={r['vid_score']:.3f}",
            fontsize=7, fontweight='bold'
        )
        ax.tick_params(labelsize=6)
        if i == 0:
            ax.legend(fontsize=5, loc='upper right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Hide empty subplots
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    title = (f'Top-{n} False Positives (Normal → Predicted Anomalous)'
             if error_type == 'FP'
             else f'Top-{n} False Negatives (Anomalous → Predicted Normal)')
    fig.suptitle(title, fontsize=11, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Summary text report ────────────────────────────────────────────────────────
def write_summary(fp_cases, fn_cases, output_path, threshold, total_videos):
    """Write human-readable error taxonomy summary."""
    n_fp = len(fp_cases)
    n_fn = len(fn_cases)

    fp_by_mode = defaultdict(list)
    fn_by_mode = defaultdict(list)
    for r in fp_cases:
        fp_by_mode[r['mode_code']].append(r)
    for r in fn_cases:
        fn_by_mode[r['mode_code']].append(r)

    fp_by_cat = defaultdict(int)
    fn_by_cat = defaultdict(int)
    for r in fp_cases:
        fp_by_cat[r['cat_name']] += 1
    for r in fn_cases:
        fn_by_cat[r['cat_name']] += 1

    lines = []
    lines.append("=" * 70)
    lines.append("ERROR TAXONOMY REPORT — Violence Event Detection")
    lines.append("=" * 70)
    lines.append(f"Threshold        : {threshold}")
    lines.append(f"Total val videos : {total_videos}")
    lines.append(f"False Positives  : {n_fp}  (Normal predicted as Anomalous)")
    lines.append(f"False Negatives  : {n_fn}  (Anomalous predicted as Normal)")
    lines.append("")

    lines.append("─" * 70)
    lines.append("FALSE POSITIVE BREAKDOWN BY FAILURE MODE")
    lines.append("─" * 70)
    for mode, cases in sorted(fp_by_mode.items(), key=lambda x: -len(x[1])):
        desc = cases[0]['mode_desc']
        lines.append(f"  {mode:<30s}  n={len(cases):3d}  — {desc}")
    lines.append("")
    lines.append("  FP by original category (all should be Normal):")
    for cat, cnt in sorted(fp_by_cat.items(), key=lambda x: -x[1]):
        lines.append(f"    {cat:<15s}: {cnt}")
    lines.append("")

    lines.append("─" * 70)
    lines.append("FALSE NEGATIVE BREAKDOWN BY FAILURE MODE")
    lines.append("─" * 70)
    for mode, cases in sorted(fn_by_mode.items(), key=lambda x: -len(x[1])):
        desc = cases[0]['mode_desc']
        lines.append(f"  {mode:<30s}  n={len(cases):3d}  — {desc}")
    lines.append("")
    lines.append("  FN by ground-truth category:")
    for cat, cnt in sorted(fn_by_cat.items(), key=lambda x: -x[1]):
        lines.append(f"    {cat:<15s}: {cnt}")
    lines.append("")

    lines.append("─" * 70)
    lines.append("INTERPRETATION NOTES")
    lines.append("─" * 70)
    lines.append("  FP_SPIKE       : Camera artifacts — future work: temporal smoothing")
    lines.append("  FP_BOUNDARY    : Scene/lighting transitions — add scene-change detection")
    lines.append("  FP_MOTION      : Energetic non-violent motion (sports, crowds)")
    lines.append("  FP_PERSISTENT  : Scene-level mismatch (indoor crowds, construction)")
    lines.append("  FP_TRN_OVERFIT : TRN propagating noise — regularization may help")
    lines.append("")
    lines.append("  FN_SHORT_EVENT : Violence <2 segments — model needs segment-level GT")
    lines.append("  FN_MARGINAL    : Threshold tuning may recover these")
    lines.append("  FN_WEAK_SIGNAL : Subtle/non-stereotypical violence forms")
    lines.append("  FN_TRN_SUPPRESSED: TRN over-smoothing — reduce temporal context")
    lines.append("  FN_UNSEEN      : Category not represented in training features")
    lines.append("=" * 70)

    text = "\n".join(lines)
    with open(output_path, 'w') as f:
        f.write(text)
    print(f"  Saved: {output_path}")
    print()
    print(text)


# ── Save CSV ───────────────────────────────────────────────────────────────────
def save_csv(fp_cases, fn_cases, output_path):
    fieldnames = ['error_type', 'cat_name', 'label', 'predicted',
                  'vid_score', 'mode_code', 'mode_desc', 'pred_cls', 'idx']
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in fp_cases:
            writer.writerow({k: r[k] for k in fieldnames})
        for r in fn_cases:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"  Saved: {output_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = load_model(args.checkpoint, device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Build val dataset
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

    # Run inference
    print(f"\nRunning inference (threshold={args.threshold})...")
    results = run_inference(model, dataset, device, args.threshold)

    # Separate TP, TN, FP, FN
    tp_cases, tn_cases, fp_cases, fn_cases = [], [], [], []
    for r in results:
        gt   = r['label']       # 1=anomalous, 0=normal
        pred = r['predicted']   # 1=predicted anom, 0=predicted normal
        if gt == 0 and pred == 1:
            # FP: Normal predicted as Anomalous
            mode_code, mode_desc = classify_fp_mode(
                r['trn_scores'], r['mil_scores'], r['cat_name'], str(r['idx'])
            )
            r['error_type'] = 'FP'
            r['mode_code']  = mode_code
            r['mode_desc']  = mode_desc
            fp_cases.append(r)
        elif gt == 1 and pred == 0:
            # FN: Anomalous predicted as Normal
            mode_code, mode_desc = classify_fn_mode(
                r['trn_scores'], r['mil_scores'], r['cat_name'], str(r['idx'])
            )
            r['error_type'] = 'FN'
            r['mode_code']  = mode_code
            r['mode_desc']  = mode_desc
            fn_cases.append(r)
        elif gt == 1 and pred == 1:
            r['error_type'] = 'TP'
            r['mode_code']  = 'TP'
            r['mode_desc']  = 'Correct detection'
            tp_cases.append(r)
        else:
            r['error_type'] = 'TN'
            r['mode_code']  = 'TN'
            r['mode_desc']  = 'Correct rejection'
            tn_cases.append(r)

    total = len(results)
    print(f"\nConfusion matrix (threshold={args.threshold}):")
    print(f"  TP={len(tp_cases)}  TN={len(tn_cases)}  FP={len(fp_cases)}  FN={len(fn_cases)}")
    if len(tp_cases) + len(fp_cases) > 0:
        prec = len(tp_cases) / (len(tp_cases) + len(fp_cases))
        print(f"  Precision = {prec:.4f}")
    if len(tp_cases) + len(fn_cases) > 0:
        rec = len(tp_cases) / (len(tp_cases) + len(fn_cases))
        print(f"  Recall    = {rec:.4f}")

    # Save CSV
    print("\nSaving CSV...")
    save_csv(fp_cases, fn_cases,
             os.path.join(args.output_dir, 'error_taxonomy.csv'))

    # Save summary
    print("\nWriting summary...")
    write_summary(fp_cases, fn_cases,
                  os.path.join(args.output_dir, 'error_summary.txt'),
                  args.threshold, total)

    # Plot FP score timelines
    print("\nPlotting FP cases...")
    plot_error_cases(fp_cases, 'FP',
                     os.path.join(args.output_dir, 'fp_cases.png'),
                     max_cases=50)

    # Plot FN score timelines
    print("\nPlotting FN cases...")
    plot_error_cases(fn_cases, 'FN',
                     os.path.join(args.output_dir, 'fn_cases.png'),
                     max_cases=50)

    print("\nDone. Results saved to:", args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',    required=True)
    parser.add_argument('--feature_dir',   required=True)
    parser.add_argument('--list_file',     required=True)
    parser.add_argument('--output_dir',    default='./visualizations/error_taxonomy')
    parser.add_argument('--num_segments',  type=int,   default=32)
    parser.add_argument('--val_ratio',     type=float, default=0.2)
    parser.add_argument('--threshold',     type=float, default=0.5)
    args = parser.parse_args()
    main(args)
