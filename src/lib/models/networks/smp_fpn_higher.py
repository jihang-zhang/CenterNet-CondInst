from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import segmentation_models_pytorch as smp
from .higher_head import HigherHead

logger = logging.getLogger(__name__)

class PoseFPNNet(nn.Module):
    def __init__(self, base_name, heads, head_conv=256):
        super(PoseFPNNet, self).__init__()

        base = smp.FPN(base_name, encoder_weights='imagenet', decoder_dropout=0, decoder_segmentation_channels=64, upsampling=1)
        self.encoder = base.encoder
        self.decoder = base.decoder

        self.heads = heads
        self.higher = HigherHead(64+self.heads['hm2'], 64)

        for head in self.heads:
            classes = self.heads[head]
            fc = nn.Sequential(
                nn.Conv2d(64, head_conv,
                          kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, classes,
                          kernel_size=1, stride=1,
                          padding=0, bias=True))
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)

        self.heads.pop('hm2', None)

        del base

    def forward(self, x):
        features = self.encoder(x)
        y = self.decoder(*features)

        z = {}
        z['hm2'] = self.__getattr__('hm2')(y)

        x = torch.cat((y, z['hm2']), 1)
        x = self.higher(x)

        for head in self.heads:
            z[head] = self.__getattr__(head)(x)
        return [z]

    def freeze_backbone(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False

    def freeze_head(self, heads):
        for head in heads:
            for p in self.__getattr__(head).parameters():
                p.requires_grad = False

    def set_mode(self, mode, is_freeze_bn=False):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
            if is_freeze_bn==True: ##freeze
                for m in self.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
                        # m.weight.requires_grad = False
                        # m.bias.requires_grad   = False

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            # torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            # torch.nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def get_pose_net(base_name, heads, head_conv):
    model = PoseFPNNet(base_name, heads, head_conv)
    return model