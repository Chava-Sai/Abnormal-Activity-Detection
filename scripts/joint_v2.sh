#!/bin/bash
#$ -P cs585
#$ -N joint_v2
#$ -o /projectnb/cs585/students/saichava/logs/joint_v2_$JOB_ID.out
#$ -e /projectnb/cs585/students/saichava/logs/joint_v2_$JOB_ID.err
#$ -l h_rt=12:00:00
#$ -l gpus=1
#$ -l gpu_c=7.0
#$ -l gpu_memory=20G
#$ -pe omp 4

module load miniconda
conda activate violence_det

PROJ=/projectnb/cs585/students/saichava

python3 ~/violence_detection/src/train.py \
  --train_dir $PROJ/datasets/ucf_train/UCF_Train_ten_crop_i3d \
  --test_dir  $PROJ/datasets/ucf_test/UCF_test_feature \
  --train_list ~/violence_detection/list/ucf-i3d-train.list \
  --test_list  ~/violence_detection/list/ucf-i3d-test.list \
  --num_segments 32 --seed 42 --epochs 100 --joint \
  --checkpoint_dir $PROJ/experiments/joint_v2/checkpoints \
  --log_dir        $PROJ/experiments/joint_v2/logs
