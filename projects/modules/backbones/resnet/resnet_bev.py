

'''
    file:   resnet_bev.py
    author: zhangxiong (1025679612@qq.com)
    date:   2023/08/09
'''

from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..register import (
    BACKBONES,
)

from .resnet import conv1x1, conv3x3, BasicBlock, Bottleneck

@BACKBONES.register_module()
class ResNetBEV(nn.Module):
    def __init__(self, numC_input, num_layer=[2,2,2], num_channels=None, stride=[2,2,2],
                 backbone_output_ids=None, norm_cfg=dict(type='BN'),
                 with_cp=False, block_type='Basic',*args, **kwargs):
        super(ResNetBEV, self).__init__()
        assert len(num_layer)==len(stride)
        num_channels = [numC_input*2**(i+1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids
        layers = []
        if block_type == 'BottleNeck':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer=[Bottleneck(curr_numC, num_channels[i]//4, stride=stride[i],
                                  downsample=nn.Conv2d(curr_numC,num_channels[i],3,stride[i],1),
                                  norm_cfg=norm_cfg)]
                curr_numC= num_channels[i]
                layer.extend([Bottleneck(curr_numC, curr_numC//4,
                                         norm_cfg=norm_cfg) for _ in range(num_layer[i]-1)])
                layers.append(nn.Sequential(*layer))
        elif block_type == 'Basic':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer=[BasicBlock(curr_numC, num_channels[i], stride=stride[i],
                                  downsample=nn.Conv2d(curr_numC,num_channels[i],3,stride[i],1),
                                  norm_cfg=norm_cfg)]
                curr_numC= num_channels[i]
                layer.extend([BasicBlock(curr_numC, curr_numC, norm_cfg=norm_cfg) for _ in range(num_layer[i]-1)])
                layers.append(nn.Sequential(*layer))
        else:
            assert False
        self.layers = nn.Sequential(*layers)

        self.with_cp = with_cp

    def forward(self, x):
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats

def build_norm_layer(norm_cfg, out_channels, *args, **kargs):
    if norm_cfg in ['BN', 'bn']:
        return nn.BatchNorm2d(out_channels, **kargs)
    else:
        assert 'invalid norm type', f'only BatchNorm2D supported, but get {norm_cfg}'