# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 21:20:45 2020

@author: zn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CNN_2d(nn.Module):
    def __init__(self, input_bands, num_classification):
        super(CNN_2d, self).__init__()
        '''
        一个简单的2D-CNN
        '''
        self.conv0 = nn.Conv2d(input_bands, 16, 5, 1, 2)  # (indim,outdim,filter,stride,pad)
        self.bn0 = nn.BatchNorm2d(16)  # (indim)此处应该是输出维度，也就是输出的特征图个数
        self.pool0 = nn.MaxPool2d(2, 2)  # 31*31,25*25,35*35   13*13
        self.conv1 = nn.Conv2d(16, 32, 5, 1, 2)  # (indim,outdim,filter,stride,pad)
        self.bn1 = nn.BatchNorm2d(32)  # (indim)
        self.pool1 = nn.MaxPool2d(2, 2)  # 15*15,12*12,17*17 6*6
        self.conv2 = nn.Conv2d(32, 64, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # 7*7 6*6 8*8  3*3
        self.conv3 = nn.Conv2d(64, 128, 5, 1, 2)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # 3*3 3*3 4*4  1*1

        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, num_classification)

    def forward(self, x):
        out = self.pool0(F.relu(self.bn0(self.conv0(x))))
        out = self.pool1(F.relu(self.bn1(self.conv1(out))))
        out = self.pool2(F.relu(self.bn2(self.conv2(out))))
        out = self.pool3(F.relu(self.bn3(self.conv3(out))))
        out = out.reshape(-1, 128)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
