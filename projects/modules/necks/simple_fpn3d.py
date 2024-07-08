

'''
    file:   simple_fpn3d.py
    author: zhangxiong (1025679612@qq.com)
    date:   2023/08/11
'''

import torch.nn as nn
import torch.nn.functional as F

from ..bricks import build_conv_layer, build_upsample_layer, build_norm_layer
from .register import NECKS

@NECKS.register_module()
class SimpleFPN3d(nn.Module):
    def __init__(self, dim_vox_fpn=None, stride_vox_fpn=None, out_feat_ids=None, *args, **kwargs):
        super(SimpleFPN3d, self).__init__()
        self.dim_vox_fpn, self.stride_vox_fpn, self.out_feat_ids = dim_vox_fpn, stride_vox_fpn, out_feat_ids

        self.deblocks = nn.ModuleList()
       
        norm_cfg=dict(type='GN', num_groups=16, requires_grad=True)
        upsample_cfg=dict(type='trilinear', scale_factor=2, mode='trilinear')
        conv_cfg=dict(type='Conv3d', bias=False)

        #create fpn modules
        for (i, (c, s)) in enumerate(zip(self.dim_vox_fpn, self.stride_vox_fpn)):
            if s > 1:
                # upsample_layer = build_upsample_layer(
                #     upsample_cfg,
                #     in_channels=c,
                #     out_channels=self.dim_vox_fpn[i+1],
                #     kernel_size=s,
                #     stride=s
                # )
                upsample_layer = nn.Sequential(
                    build_upsample_layer(
                        upsample_cfg
                    ),
                    build_conv_layer(
                        conv_cfg,
                        in_channels=c,
                        out_channels=self.dim_vox_fpn[i+1],
                        kernel_size=3,
                        stride=1,
                        padding=1
                    ),
                )
            else:
                upsample_layer = build_conv_layer(
                    conv_cfg,
                    in_channels=c,
                    out_channels=self.dim_vox_fpn[i+1],
                    kernel_size=3,
                    stride=1,
                    padding=1
                )

            self.deblocks.append(
                nn.Sequential(
                    upsample_layer,
                    build_norm_layer(norm_cfg, self.dim_vox_fpn[i+1])[1],
                    nn.ReLU(inplace=True)
                )
            )

    def forward(self, vox_feats):
        fpn_feats = []
        feat = vox_feats.pop()
        for (i, (c, s, op)) in enumerate(zip(self.dim_vox_fpn, self.stride_vox_fpn, self.deblocks)):
            if s==2 and len(vox_feats)>0:
                feat=op(feat)+vox_feats.pop()
            else:
                feat = op(feat)
            fpn_feats.append(feat)

        return [fpn_feats[_id] for _id in self.out_feat_ids]