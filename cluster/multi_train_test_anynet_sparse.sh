#!/bin/bash

# train anynet with different base model and learning rate

#base_model='resnet18'
#base_model='resnet50'
#base_model='resnet101'
#base_model='se_resnext50_32x4d'
#base_model='nasnetamobile'

#for base_model in resnet50 resnet101 se_resnext50_32x4d se_resnext101_32x4d squeezenet1_1; do
for base_model in resnet18; do

  for lr in 0.0001 0.00001 0.000001 0.0000001;  do
    echo learning rate: $lr
    echo base model: $base_model
    #./sbg_p100.sh train_test_anynet_sparse.sh $lr $base_model
    ./sbg_k80.sh train_test_anynet_sparse.sh $lr $base_model
  done

done


