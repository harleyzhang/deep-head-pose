#!/bin/bash

# input argument
# $1: batch size
# $2: learning rate
# $3: alpha

source /homeappl/home/zhangh/Slurm/setenv3.sh
cd ..

data_dir='/homeappl/home/zhangh/Work/DONOTREMOVE/Databases/300W'
filename_list='/homeappl/home/zhangh/Work/DONOTREMOVE/Databases/300W/300W_LP.txt'
output_prefix='hopenet_alpha2_lrdrop'
alpha=2

python3 \
  code/train_hopenet.py \
  --num_epochs 25 \
  --batch_size 64 \
  --lr 0.00001 \
  --alpha $alpha \
  --data_dir $data_dir \
  --filename_list $filename_list \
  --output_string $output_prefix \
  --dataset Pose_300W_LP

exit 0

filename_list='/homeappl/home/zhangh/Work/DONOTREMOVE/Databases/300W/AFLW2000_valid.txt'
for epoch_id in {1..25}; do

  snapshot=/homeappl/home/zhangh/Work/DONOTREMOVE/headpose/deep-head-pose/output/snapshots/${output_prefix}_epoch_${epoch_id}.pkl

  echo -------------------------------------------------
  echo       Epoch ID: ${epoch_id}
  echo -------------------------------------------------
  python3 \
    code/test_hopenet.py \
    --snapshot $snapshot \
    --batch_size 64 \
    --data_dir $data_dir \
    --filename_list $filename_list \
    --save_viz true\
    --dataset AFLW2000

done

