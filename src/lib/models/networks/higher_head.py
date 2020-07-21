from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HigherHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HigherHead, self).__init__()

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels, 
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

        res_layers = []
        for _ in range(4):
            res_layers.append(nn.Sequential(
                BasicBlock(out_channels, out_channels)
            ))

        self.res_block = nn.Sequential(*res_layers)

    def forward(self, x):
        x = self.deconv(x)
        x = self.res_block(x)
        return x
