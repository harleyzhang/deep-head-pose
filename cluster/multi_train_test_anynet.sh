#!/bin/bash

# train anynet with different base model and learning rate

#base_model='resnet18'
#base_model='resnet50'
#base_model='resnet101'
#base_model='se_resnext50_32x4d'
#base_model='nasnetamobile'

#for base_model in resnet50 resnet101 se_resnext50_32x4d nasnetamobile squeezenet1_1; do
#for base_model in resnet101 se_resnext50_32x4d nasnetamobile; do
for base_model in nasnetamobile; do

  for lr in 0.001 0.0001 0.00001 0.000001;  do
    echo learning rate: $lr
    echo base model: $base_model
    #./sbg_p100.sh train_test_anynet.sh $lr $base_model
    ./sbg_k80.sh train_test_anynet.sh $lr $base_model
  done

done


