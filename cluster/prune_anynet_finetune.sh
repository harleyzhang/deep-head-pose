#!/bin/bash

source /homeappl/home/zhangh/Slurm/setenv3.sh

cd ..

if [ $# -lt 1 ]; then
  echo No base model is giving, using resnet50
  base_model=resnet50
else
  base_model=$1
fi


#percentage of channels to be removed
percent=0.3
lr=0.000001
pretrained_model="/homeappl/home/zhangh/Work/DONOTREMOVE/headpose/deep-head-pose/output/snapshots/sparse_${base_model}_alpha2_lrdrop_lr${lr}_epoch_20.pkl"

data_dir='/homeappl/home/zhangh/Work/DONOTREMOVE/Databases/300W'

# testing 
echo -------------------------------------------------
echo   Testing pretrained model
echo -------------------------------------------------

filename_list='/homeappl/home/zhangh/Work/DONOTREMOVE/Databases/300W/AFLW2000_valid.txt'


snapshot=/homeappl/home/zhangh/Work/DONOTREMOVE/headpose/deep-head-pose/output/snapshots/${output_prefix}_epoch_${epoch_id}.pkl

python3 -u \
    code/test_anynet.py \
    --base_model $base_model \
    --snapshot $pretrained_model \
    --batch_size 32 \
    --data_dir $data_dir \
    --filename_list $filename_list \
    --dataset AFLW2000

# prune
echo -------------------------------------------------
echo   pruning
echo -------------------------------------------------

pruned_model=${pretrained_model}"_pruned.pkl"

python3 \
  code/prune_anynet.py \
  --base_model $base_model \
  --percent $percent \
  --model $pretrained_model \
  --save $pruned_model 


# finetuning
echo -------------------------------------------------
echo   Finetune
echo -------------------------------------------------

data_dir='/homeappl/home/zhangh/Work/DONOTREMOVE/Databases/300W'
filename_list='/homeappl/home/zhangh/Work/DONOTREMOVE/Databases/300W/300W_LP.txt'
lr=0.000001
alpha=2
output_prefix="finetuned_${base_model}_alpha2_lrdrop_lr${lr}"

python3 \
  code/train_anynet_sparse.py \
  --base_model $base_model \
  --num_epochs 25 \
  --batch_size 64 \
  --lr $lr \
  --sparsity 0 \
  --alpha $alpha \
  --data_dir $data_dir \
  --filename_list $filename_list \
  --snapshot $pruned_model \
  --output_string $output_prefix \
  --dataset Pose_300W_LP


# testing 
echo -------------------------------------------------
echo   Testing
echo -------------------------------------------------

filename_list='/homeappl/home/zhangh/Work/DONOTREMOVE/Databases/300W/AFLW2000_valid.txt'

for epoch_id in {1..25}; do

  echo -------------------------------------------------
  echo       Epoch ID: ${epoch_id}
  echo -------------------------------------------------
  snapshot=/homeappl/home/zhangh/Work/DONOTREMOVE/headpose/deep-head-pose/output/snapshots/${output_prefix}_epoch_${epoch_id}.pkl

  python3 -u \
    code/test_anynet.py \
    --base_model $base_model \
    --snapshot $snapshot \
    --batch_size 32 \
    --data_dir $data_dir \
    --filename_list $filename_list \
    --dataset AFLW2000

done

