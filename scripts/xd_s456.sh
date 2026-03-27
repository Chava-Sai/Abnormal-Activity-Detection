#!/bin/bash
#$ -P cs585
#$ -N xd_s456
#$ -o /projectnb/cs585/students/saichava/logs/xd_s456_$JOB_ID.out
#$ -e /projectnb/cs585/students/saichava/logs/xd_s456_$JOB_ID.err
#$ -l h_rt=12:00:00
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l gpu_memory=20G
#$ -pe omp 4

module load miniconda
conda activate violence_det

PROJ=/projectnb/cs585/students/saichava

python3 ~/violence_detection/src/xd_train.py \
  --rgb_dir  $PROJ/datasets/i3d-features/RGB \
  --flow_dir $PROJ/datasets/i3d-features/Flow \
  --checkpoint_dir $PROJ/experiments/xd_s456/checkpoints \
  --log_dir        $PROJ/experiments/xd_s456/logs \
  --seed 456 --epochs 100
