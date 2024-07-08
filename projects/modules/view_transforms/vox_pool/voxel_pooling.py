
'''
    file:   voxel_pooling.py
    author: zhangxiong (1025679612@qq.com)
    date:   2023/08/23
'''

from ..register import VOXEL_POOLING

import torch
import torch.nn as nn
from einops import rearrange
import torchvision

@VOXEL_POOLING.register_module()
class VoxelPooling(nn.Module):
    def __init__(self, c_in, c_hidden, dcn=True, *args, **kwargs):
        super(VoxelPooling, self).__init__()
        self.c_in, self.c_hidden, self.m_conv = c_in, c_hidden, nn.Conv2d
        self._build_layers_()
    
    def _build_layers_(self):
        layers = []
        c_in = self.c_in

        for c_out in self.c_hidden:
            layers = layers + [self.m_conv(c_in, c_out, 5, 1, 2), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
            c_in = c_out
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        n, c, d, h, w = x.shape #receive a 5d tensor
        x = rearrange(x, 'n c d h w -> n (c d) h w')
        x = self.layers(x)
        return rearrange(x, 'n (c d) h w -> n c d h w', c=1)
