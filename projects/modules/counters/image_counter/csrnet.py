
'''
    file:   crsnet.py
    author: zhangxiong (1025679612@qq.com)
    date:   2023/07/20
'''

from .register import COUNTERS

import torch.nn as nn
import torch.nn.functional as F
import torch

@COUNTERS.register_module()
class CSRNet(nn.Module):
    def __init__(self, net_cfg, in_channels, stack=False, rescale=1.0, *args, **kwargs):
        super(CSRNet, self).__init__()
        self.backend = make_layers(net_cfg, in_channels=in_channels, dilation=True, batch_norm=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1, bias=True)
        self.stack, self.rescale = stack, rescale
        self._initialize_weights()

    def stack_feat(self, x):
        _, _, h, w = x[0].shape
        return torch.cat([F.interpolate(feat, (h, w), mode='bilinear') for feat in x], 1)
        
    def forward(self, x):
        if self.stack:
            x = self.stack_feat(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x if self.rescale is None else x/self.rescale
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)