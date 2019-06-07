#!/bin/bash

# check input argument
if [ $1 == '' ]; then
  echo No learning rate is given, using default value 0.001
  lr=0.001
else
  lr=$1
fi

echo -----------------------------------
echo lr: $lr

source /homeappl/home/zhangh/Slurm/setenv3.sh
cd ..

data_dir='/homeappl/home/zhangh/Work/DONOTREMOVE/Databases/300W'
filename_list='/homeappl/home/zhangh/Work/DONOTREMOVE/Databases/300W/300W_LP.txt'
output_prefix="resnet50_lr${lr}"

python3 \
  code/train_resnet50_regression.py \
  --num_epochs 25 \
  --batch_size 16 \
  --lr $lr \
  --data_dir $data_dir \
  --filename_list $filename_list \
  --output_string $output_prefix \
  --dataset Pose_300W_LP

