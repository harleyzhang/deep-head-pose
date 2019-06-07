#!/bin/bash

set -e

source /homeappl/home/zhangh/Slurm/setenv2.sh
cd ..

data_dir='/homeappl/home/zhangh/Work/DONOTREMOVE/Databases/300W'
filename_list='/homeappl/home/zhangh/Work/DONOTREMOVE/Databases/300W/AFLW2000_valid.txt'
for epoch_id in {1..25}; do
#for epoch_id in {13..13}; do
  #snapshot=/homeappl/home/zhangh/Work/DONOTREMOVE/headpose/deep-head-pose/output/snapshots/hopenet_300W_LP_epoch_${epoch_id}.pkl

  snapshot=/homeappl/home/zhangh/Work/DONOTREMOVE/headpose/deep-head-pose/output/snapshots/hopenet_bs16_epoch_${epoch_id}.pkl

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

