#!/bin/bash

source /homeappl/home/zhangh/Slurm/setenv3.sh

cd ..

base_model=resnet101
pretrained_model='/homeappl/home/zhangh/Work/DONOTREMOVE/headpose/deep-head-pose/output/prune/resnet101_alpha2_lrdrop_lr0.000001_epoch_20.pkl'

fname=$(basename ${pretrained_model})
pruned_model=${pretrained_model}"_pruned.pkl"

python3 \
  code/prune_anynet.py \
  --base_model $base_model \
  --percent 0.5 \
  --model $pretrained_model \
  --save $pruned_model 


data_dir='/homeappl/home/zhangh/Work/DONOTREMOVE/Databases/300W'
filename_list='/homeappl/home/zhangh/Work/DONOTREMOVE/Databases/300W/AFLW2000_valid.txt'

#testing 
echo -------------------------------------------------
echo   Before pruning
echo -------------------------------------------------
python3 -u \
  code/test_anynet.py \
  --base_model $base_model \
  --snapshot $pretrained_model \
  --batch_size 32 \
  --data_dir $data_dir \
  --filename_list $filename_list \
  --dataset AFLW2000


#testing 
echo -------------------------------------------------
echo   Pruned model
echo -------------------------------------------------
python3 -u \
  code/test_anynet.py \
  --base_model $base_model \
  --snapshot $pruned_model\
  --batch_size 32 \
  --data_dir $data_dir \
  --filename_list $filename_list \
  --dataset AFLW2000


