#!/bin/bash -l
#$ -S /bin/bash
#$ -P cs585
#$ -l h_rt=4:00:00
#$ -l mem_per_core=16G
#$ -l gpus=1
#$ -N shanghai_eval
#$ -j y
#$ -o /projectnb/cs585/students/shrishty/shanghaitech/voilence_eval/logs/shanghai_eval.log

# ── paths ─────────────────────────────────────────────────────────────────────
BASE=/projectnb/cs585/students/shrishty/shanghaitech
FRAMES_ROOT=$BASE/shanghaitech/testing/frames
MASK_DIR=$BASE/shanghaitech/testing/test_frame_mask
FEAT_DIR=$BASE/voilence_eval/shanghai_test_features
SCRIPT_DIR=$BASE/voilence_eval
CHECKPOINT=$BASE/shanghaitech_features/best.pt

# ── environment ───────────────────────────────────────────────────────────────
module load miniconda
conda activate violence_eval

echo "========================================"
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "========================================"

mkdir -p $FEAT_DIR
mkdir -p $SCRIPT_DIR/logs

# ── Step 1: Extract test features ─────────────────────────────────────────────
echo ""
echo "STEP 1: Extracting I3D features from ShanghaiTech test frames..."
python $SCRIPT_DIR/extract_shanghai_test_features.py \
    --frames_root $FRAMES_ROOT \
    --out_dir     $FEAT_DIR \
    --device      cuda

echo ""
echo "Feature extraction complete. Files in $FEAT_DIR:"
ls $FEAT_DIR | wc -l

# ── Step 2: Run evaluation ────────────────────────────────────────────────────
echo ""
echo "STEP 2: Running ShanghaiTech robustness evaluation..."
python $SCRIPT_DIR/evaluate_shanghaitech.py \
    --checkpoint        $CHECKPOINT \
    --feat_dir          $FEAT_DIR \
    --mask_dir          $MASK_DIR \
    --num_segments      32 \
    --frames_per_segment 16 \
    --gpus              0 \
    --output_json       $SCRIPT_DIR/shanghai_eval_results.json

echo ""
echo "========================================"
echo "Job finished: $(date)"
echo "Results: $SCRIPT_DIR/shanghai_eval_results.json"
echo "========================================"
