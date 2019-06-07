import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import numpy as np
import torch.nn.functional as F

import pretrainedmodels

class Anynet(nn.Module):
    # Anynet with 3 output layers for yaw, pitch and roll
    # Predicts Euler angles by binning and regression with the expected value
    def __init__(self, model_name, num_bins, init_weight=True):
        super(Anynet, self).__init__()
     
        imgnet_settings = pretrainedmodels.pretrained_settings[model_name]['imagenet']

        self.base_num_classes = imgnet_settings['num_classes']
        self.input_size = imgnet_settings['input_size']
        self.base_mean = imgnet_settings['mean']
        self.base_std = imgnet_settings['std']

        pretrained = None
        if init_weight: pretrained = 'imagenet'

        self.base_model = pretrainedmodels.__dict__[model_name]( \
          num_classes=self.base_num_classes, \
           pretrained=pretrained)

        feature_shape = self._cal_feature_shape()
        #print('feature shape:{}'.format(feature_shape))
        self.avgpool = nn.AvgPool2d(feature_shape[2:])
        self.fc_yaw = nn.Linear(feature_shape[1], num_bins)
        self.fc_pitch = nn.Linear(feature_shape[1], num_bins)
        self.fc_roll = nn.Linear(feature_shape[1], num_bins)

    def _cal_feature_shape(self):
        self.base_model.cpu()
        input_tensor = torch.autograd.Variable(torch.rand(1, 
          *self.input_size))
        output_tensor = self.base_model.features(input_tensor)
        return output_tensor.shape

    def forward(self, x):
        x = self.base_model.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        pre_yaw = self.fc_yaw(x)
        pre_pitch = self.fc_pitch(x)
        pre_roll = self.fc_roll(x)

        return pre_yaw, pre_pitch, pre_roll

