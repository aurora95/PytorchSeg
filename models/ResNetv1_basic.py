from __future__ import absolute_import

import math
import torch
import torch.nn as nn
from models.base_models.resnet import *

class ResNet50_16s(nn.Module):
    def __init__(self, classes, **kwargs):
        super(ResNet50_16s, self).__init__(**kwargs)
        self.base = resnet50(pretrained=True, stride=16)
        self.classifier = nn.Conv2d(in_channels=2048, out_channels=classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.upsample = nn.Upsample(scale_factor=16, mode='bilinear')

        nn.init.kaiming_normal(self.classifier.weight.data)
        self.classifier.bias.data.zero_()

    def forward(self, x):
        x = self.base(x)
        x = self.classifier(x)
        x = self.upsample(x)

        return x
