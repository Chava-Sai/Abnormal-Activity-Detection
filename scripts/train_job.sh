#!/bin/bash
#$ -P cs585
#$ -N violence_progressive
#$ -o /projectnb/cs585/students/saichava/logs/progressive_$JOB_ID.out
#$ -e /projectnb/cs585/students/saichava/logs/progressive_$JOB_ID.err
#$ -l h_rt=24:00:00
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l gpu_memory=40G
#$ -pe omp 4
#$ -l mem_per_core=8G
#$ -m bea
#$ -M saichava@bu.edu

# Load environment
module load miniconda
conda activate violence_det

echo "Job $JOB_ID starting on $(hostname) at $(date)"
nvidia-smi | head -15

PROJ=/projectnb/cs585/students/saichava
TRAIN_DIR=$PROJ/datasets/ucf_train/UCF_Train_ten_crop_i3d
TEST_DIR=$PROJ/datasets/ucf_test/UCF_test_feature
LIST_DIR=$HOME/violence_detection/list
OUT_DIR=$PROJ/experiments/progressive_run_$(date +%Y%m%d_%H%M%S)
SCHEDULE=progressive

mkdir -p $OUT_DIR/checkpoints $OUT_DIR/logs/tensorboard
mkdir -p $PROJ/logs  # for SGE stdout/stderr

echo "Output dir: $OUT_DIR"
echo "Training schedule: $SCHEDULE"

python $HOME/violence_detection/src/train.py \
  --train_dir   $TRAIN_DIR \
  --test_dir    $TEST_DIR \
  --train_list  $LIST_DIR/ucf-i3d-train.list \
  --test_list   $LIST_DIR/ucf-i3d-test.list \
  --val_ratio   0.2 \
  --epochs      100 \
  --batch_size  32 \
  --lr          1e-4 \
  --weight_decay 1e-4 \
  --num_segments 32 \
  --frames_per_segment 16 \
  --d_model     512 \
  --nhead       8 \
  --trn_layers  2 \
  --lambda_cls  0.5 \
  --lambda_bnd  0.3 \
  --topk_ratio  0.1 \
  --topk_cls    5 \
  --eval_freq   5 \
  --save_freq   10 \
  --seed        42 \
  --checkpoint_dir $OUT_DIR/checkpoints \
  --log_dir        $OUT_DIR/logs/tensorboard

echo "Training finished at $(date)"
echo "Best checkpoint: $OUT_DIR/checkpoints/best.pt"
