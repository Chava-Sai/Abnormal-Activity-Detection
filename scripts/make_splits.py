"""
Create stratified train/val split from UCF-Crime training features.

Fixes the rstrip() bug: uses regex to correctly extract category names from
filenames like 'Abuse001_x264_i3d.npy', 'Normal_Videos_001_x264_i3d.npy'.

Usage:
    python scripts/make_splits.py \
        --feature_dir /projectnb/cs585/students/saichava/datasets/ucf_train/UCF_Train_ten_crop_i3d \
        --list_file   ~/violence_detection/list/ucf-i3d-train.list \
        --output_dir  ~/violence_detection/list \
        --val_ratio   0.2
"""

import os
import re
import sys
import random
import argparse
from collections import defaultdict

VIOLENCE_CATEGORIES = {'Abuse', 'Fighting', 'Shooting', 'Explosion', 'Robbery', 'Riot'}
NON_VIOLENCE_ANOMALIES = {
    'Arrest', 'Arson', 'Assault', 'Burglary', 'RoadAccidents',
    'Shoplifting', 'Stealing', 'Vandalism',
}


def get_category(filename: str) -> str:
    """
    Robustly extract category from filename using regex.

    Examples:
      'Abuse001_x264_i3d.npy'           -> 'Abuse'
      'Normal_Videos_001_x264_i3d.npy'  -> 'Normal'
      'Shooting038_x264_i3d.npy'        -> 'Shooting'
      'RoadAccidents001_x264_i3d.npy'   -> 'RoadAccidents'
    """
    basename = os.path.basename(filename)
    # Remove known suffixes first
    basename = re.sub(r'_x264_i3d\.npy$', '', basename)
    basename = re.sub(r'_i3d\.npy$', '', basename)
    basename = re.sub(r'\.npy$', '', basename)
    # Strip trailing digits (optionally preceded by underscore)
    category = re.sub(r'_?\d+$', '', basename)
    if category.startswith('Normal'):
        return 'Normal'
    return category


def main(args):
    random.seed(args.seed)

    # Read list file
    with open(os.path.expanduser(args.list_file)) as f:
        lines = [l.strip() for l in f if l.strip()]

    feature_dir = os.path.expanduser(args.feature_dir)

    # Categorize each file
    by_cat = defaultdict(list)
    missing = 0
    for line in lines:
        filename = line.replace('\\', '/').split('/')[-1]
        filepath = os.path.join(feature_dir, filename)

        if not os.path.exists(filepath):
            missing += 1
            continue

        cat = get_category(filename)
        if cat in NON_VIOLENCE_ANOMALIES and not args.include_nonviolence:
            continue  # skip non-violence anomalies

        by_cat[cat].append(filename)

    if missing:
        print(f"Warning: {missing} files from list not found in feature_dir (skipped)")

    # Print category distribution
    print("\nCategory distribution:")
    total = 0
    for cat in sorted(by_cat.keys()):
        n = len(by_cat[cat])
        total += n
        marker = '[VIOLENCE]' if cat in VIOLENCE_CATEGORIES else \
                 '[NORMAL]'   if cat == 'Normal' else '[NON-VIO]'
        print(f"  {marker:12s} {cat:20s}: {n:4d}")
    print(f"  {'TOTAL':33s}: {total:4d}")

    # Stratified split
    train_files, val_files = [], []
    for cat in sorted(by_cat.keys()):
        files = by_cat[cat]
        random.shuffle(files)
        n_val = max(1, int(len(files) * args.val_ratio))
        val_files.extend(files[:n_val])
        train_files.extend(files[n_val:])

    # Count labels in each split
    def count_labels(files):
        n_anom = sum(1 for f in files
                     if get_category(f) != 'Normal')
        n_norm = len(files) - n_anom
        return n_anom, n_norm

    tr_a, tr_n = count_labels(train_files)
    va_a, va_n = count_labels(val_files)
    print(f"\nTrain split: {len(train_files)} total ({tr_a} anomalous, {tr_n} normal)")
    print(f"Val   split: {len(val_files)} total ({va_a} anomalous, {va_n} normal)")

    # Write output list files
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    train_list = os.path.join(output_dir, 'ucf-i3d-train-split.list')
    val_list   = os.path.join(output_dir, 'ucf-i3d-val-split.list')

    with open(train_list, 'w') as f:
        f.write('\n'.join(sorted(train_files)) + '\n')
    with open(val_list, 'w') as f:
        f.write('\n'.join(sorted(val_files)) + '\n')

    print(f"\nSaved:")
    print(f"  Train: {train_list}")
    print(f"  Val:   {val_list}")

    # Also write val category breakdown for reference
    val_by_cat = defaultdict(list)
    for fn in val_files:
        val_by_cat[get_category(fn)].append(fn)
    print("\nVal category breakdown:")
    for cat in sorted(val_by_cat.keys()):
        print(f"  {cat}: {len(val_by_cat[cat])}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create stratified train/val split')
    parser.add_argument('--feature_dir',       required=True,
                        help='Directory containing .npy feature files')
    parser.add_argument('--list_file',         required=True,
                        help='Full training list file (all 1610 videos)')
    parser.add_argument('--output_dir',        required=True,
                        help='Output directory for train/val list files')
    parser.add_argument('--val_ratio',         type=float, default=0.2,
                        help='Fraction of each category to use for validation')
    parser.add_argument('--seed',              type=int, default=42)
    parser.add_argument('--include_nonviolence', action='store_true',
                        help='Include non-violence anomalies (Arrest, Arson, etc.)')
    args = parser.parse_args()
    main(args)
