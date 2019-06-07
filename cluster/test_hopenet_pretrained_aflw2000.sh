#!/bin/bash

set -e

source /homeappl/home/zhangh/Slurm/setenv2.sh
cd ..

data_dir='/homeappl/home/zhangh/Work/DONOTREMOVE/Databases/300W'
filename_list='/homeappl/home/zhangh/Work/DONOTREMOVE/Databases/300W/AFLW2000_valid.txt'

snapshot=/homeappl/home/zhangh/Work/DONOTREMOVE/headpose/dhp_pretrained/hopenet_alpha2.pkl


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


