#!/bin/bash

# train anynet with different base model and learning rate

#base_model='resnet18'
#base_model='resnet50'
#base_model='resnet101'
#base_model='se_resnext50_32x4d'
#base_model='nasnetamobile'

#for base_model in squeezenet1_1 resnet18 resnet50 resnet101 se_resnext50_32x4d se_resnext101_32x4d; do
#squeezenet has no BN, thus can't be used
for base_model in resnet18 resnet50 resnet101 se_resnext50_32x4d se_resnext101_32x4d; do
    echo base model: $base_model
    ./sbg_p100.sh prune_anynet_finetune.sh $base_model
done


