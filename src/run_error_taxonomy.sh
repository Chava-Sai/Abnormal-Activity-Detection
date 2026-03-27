#!/usr/bin/env bash
# ── Run error taxonomy analysis on SCC ───────────────────────────────────────
# Copy this to SCC and run with: bash run_error_taxonomy.sh

PROJ=/projectnb/cs585/students/saichava
CKPT=$PROJ/experiments/run_v4/checkpoints/best.pt
FEAT=$PROJ/datasets/ucf_train/UCF_Train_ten_crop_i3d
LIST=~/violence_detection/list/ucf-i3d-train.list
OUT=$PROJ/experiments/run_v4/error_taxonomy
SRC=~/violence_detection/src

mkdir -p "$OUT"

python "$SRC/error_taxonomy.py" \
    --checkpoint   "$CKPT" \
    --feature_dir  "$FEAT" \
    --list_file    "$LIST" \
    --output_dir   "$OUT" \
    --num_segments 32 \
    --val_ratio    0.2 \
    --threshold    0.5

echo "Done: $OUT"
