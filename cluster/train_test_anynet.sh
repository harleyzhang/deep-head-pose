#!/bin/bash

# input argument
# $1: batch size
# $2: learning rate
# $3: alpha

source /homeappl/home/zhangh/Slurm/setenv3.sh
cd ..

# check input argument
if [ $# -lt 1 ]; then
  echo No learning rate is given, using default value 5E-7
  lr=$(printf '%.10f' '5E-7')
else
  lr=$1
fi

#check based model
if [ $# -lt 2 ]; then
  echo No base model is given, using default resnet18
  base_model='resnet18'
else
  base_model=$2
fi


echo -----------------------------------
echo lr: $lr
echo base model: $base_model

data_dir='/homeappl/home/zhangh/Work/DONOTREMOVE/Databases/300W'
filename_list='/homeappl/home/zhangh/Work/DONOTREMOVE/Databases/300W/300W_LP.txt'
alpha=2
output_prefix="${base_model}_alpha2_lrdrop_lr${lr}"

python3 \
  code/train_anynet.py \
  --base_model $base_model \
  --num_epochs 25 \
  --batch_size 64 \
  --lr $lr \
  --alpha $alpha \
  --data_dir $data_dir \
  --filename_list $filename_list \
  --output_string $output_prefix \
  --dataset Pose_300W_LP


filename_list='/homeappl/home/zhangh/Work/DONOTREMOVE/Databases/300W/AFLW2000_valid.txt'
for epoch_id in {1..25}; do

  snapshot=/homeappl/home/zhangh/Work/DONOTREMOVE/headpose/deep-head-pose/output/snapshots/${output_prefix}_epoch_${epoch_id}.pkl

  echo -------------------------------------------------
  echo       Epoch ID: ${epoch_id}
  echo -------------------------------------------------
  python3 \
    code/test_anynet.py \
    --base_model $base_model \
    --snapshot $snapshot \
    --batch_size 32 \
    --data_dir $data_dir \
    --filename_list $filename_list \
    --dataset AFLW2000

done

