#!/bin/bash

source /homeappl/home/zhangh/Slurm/setenv2.sh
cd ..

data_dir='/homeappl/home/zhangh/Work/DONOTREMOVE/Databases/300W'
filename_list='/homeappl/home/zhangh/Work/DONOTREMOVE/Databases/300W/300W_LP.txt'
output_prefix='hopenet_bs16'
alpha=1

python3 \
  code/train_hopenet.py \
  --num_epochs 25 \
  --batch_size 16 \
  --lr 0.00001 \
  --alpha $alpha \
  --data_dir $data_dir \
  --filename_list $filename_list \
  --output_string $output_prefix \
  --dataset Pose_300W_LP

