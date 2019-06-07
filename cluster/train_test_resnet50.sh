#!/bin/bash

cd ..

echo ################################
echo "test lr drop "


# check input argument
if [ $# == 0 ]; then
  echo No learning rate is given, using default value 5E-7
  lr=$(printf '%.10f' '5E-7')
else
  lr=$1
fi

echo -----------------------------------
echo lr: $lr

source /homeappl/home/zhangh/Slurm/setenv2.sh
cd ..

data_dir='/homeappl/home/zhangh/Work/DONOTREMOVE/Databases/300W'
filename_list='/homeappl/home/zhangh/Work/DONOTREMOVE/Databases/300W/300W_LP.txt'
output_prefix="resnet50_lr_drop${lr}"

python3 \
  code/train_resnet50_regression.py \
  --num_epochs 25 \
  --batch_size 64 \
  --lr $lr \
  --data_dir $data_dir \
  --filename_list $filename_list \
  --output_string $output_prefix \
  --dataset Pose_300W_LP


filename_list='/homeappl/home/zhangh/Work/DONOTREMOVE/Databases/300W/AFLW2000_valid.txt'
for epoch_id in {1..25}; do
  snapshot=/homeappl/home/zhangh/Work/DONOTREMOVE/headpose/deep-head-pose/output/snapshots/${output_prefix}_epoch_${epoch_id}.pkl

  echo -------------------------------------------------
  echo       Epoch ID: ${epoch_id}
  echo $snapshot
  echo -------------------------------------------------
  python3 \
    code/test_resnet50_regression.py \
    --snapshot $snapshot \
    --batch_size 64 \
    --data_dir $data_dir \
    --filename_list $filename_list \
    --dataset AFLW2000
done

