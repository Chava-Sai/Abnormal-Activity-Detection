#!/bin/bash
#$ -P cs585
#$ -N seq16_s123
#$ -o /projectnb/cs585/students/saichava/logs/seq16_s123_$JOB_ID.out
#$ -e /projectnb/cs585/students/saichava/logs/seq16_s123_$JOB_ID.err
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
  --num_segments 16 --seed 123 --epochs 100 \
  --checkpoint_dir $PROJ/experiments/seq16_seed123/checkpoints \
  --log_dir        $PROJ/experiments/seq16_seed123/logs
