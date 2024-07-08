
'''
    file:   VOXCounter.py
    author: zhangxiong (1025679612@qq.com)
    date:   2023/07/19
'''

from ..register            import COUNTERS
from ...backbones          import BACKBONES
from ...view_transforms    import VOXEL_POOLING
from ...necks              import NECKS
from ...embedding          import EMBEDDINGS
from ...bricks             import build_conv_layer, build_norm_layer, build_activation_layer, build_loss_layer

import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from easydict import EasyDict as edict

@COUNTERS.register_module()
class VOXCounter(nn.Module):
    def __init__(self, cfg, *args, **kwargs,):
        super(VOXCounter, self).__init__()
        self.cfg =  copy.deepcopy(cfg)
        self.build_sub_modules(self.cfg)
        self.build_counter_head(self.cfg.vox_counter_head)
        self.build_loss_fn()

    def build_sub_modules(self, cfg):
        self.image_feature_backbone = BACKBONES.build(cfg.image_feature_backbone)
        self.image_feature_fusion   = NECKS.build(cfg.image_feature_fusion)
        self.image_feature_embed    = EMBEDDINGS.build(cfg.image_feature_cam_embedding) if 'image_feature_cam_embedding' in cfg else None
        self.image_counter_head     = COUNTERS.build(cfg.image_counter_head)
        self.feature_pooling        = VOXEL_POOLING.build(cfg.feature_pooling)
        self.vox_feature_fusion     = NECKS.build(cfg.vox_feature_fusion) if 'vox_feature_fusion' in cfg else None

    def build_loss_fn(self):
        default_loss_cfg=edict(
            img_density_weight=0.1,
            vox_density_weight=1.0,
            img_mae_weight=1e-7,
            vox_mae_weight=1e-4,
            img_density_loss_fn=edict(
                type='MSELoss',
            ),
            vox_density_loss_fn=edict(
                type='MSELoss',
            ),
        ),
        
        cfg = self.cfg.get('loss_cfg', default_loss_cfg)
        self.img_density_loss_fn = build_loss_layer(cfg.img_density_loss_fn)
        self.vox_density_loss_fn = build_loss_layer(cfg.vox_density_loss_fn)
        self.loss_cfg=cfg

    def __build_naive_counter_head__(self, cfg):
        self.counter = nn.ModuleList()
        conv_cfg=dict(type='Conv3d', bias=False)
        norm_cfg=dict(type='GN', num_groups=16, requires_grad=True)
        for dim_in in cfg.dims_in:
            counter = []
            for dim in cfg.dims:
                if dim==1: #the last conv layer
                    counter.append(build_conv_layer(edict(type='Conv3d'), in_channels=dim_in,out_channels=dim,kernel_size=3,stride=1,padding=1, bias=True))
                    if 'act_layer' in cfg and cfg.act_layer is not None:
                        counter.append(build_activation_layer(cfg.act_layer))
                else:
                    counter.append(build_conv_layer(conv_cfg,in_channels=dim_in,out_channels=dim,kernel_size=3,stride=1,padding=1,))
                    counter.append(build_norm_layer(norm_cfg,dim)[1])
                    counter.append(nn.ReLU(inplace=True))
                dim_in=dim
            self.counter.append(nn.Sequential(*counter))

    def __build_bev_counter_head__(self, cfg):
        self.counter = nn.ModuleList()
        for (dim_in, dim_out) in zip(cfg.dims_in, cfg.dims_out):
            cfg_ = copy.deepcopy(cfg)
            cfg_.c_in = dim_in
            cfg_.c_hidden=cfg.dims_hidden + [dim_out]
            self.counter.append(VOXEL_POOLING.build(cfg_))

    def build_counter_head(self, cfg):
        if 'type' not in cfg:
            return self.__build_naive_counter_head__(cfg=cfg)
        if cfg.type in ['conv', 'conv3d']:
            return self.__build_naive_counter_head__(cfg=cfg)
        if cfg.type=='VoxelPooling':
            return self.__build_bev_counter_head__(cfg=cfg)

    def loss_fn(self, p, input_dict):
        img_density_map_scale, scene_density_map_scale = self.cfg.get('img_density_map_scale', 1.0), self.cfg.get('scene_density_map_scale', 1.0)
        bs, n_cam = p.image_density_map.shape[:2]
        p_img_density_map = p.image_density_map[:,:,0]
        g_img_density_map = input_dict.labels.density_image
        density_image     = self.img_density_loss_fn(p_img_density_map, g_img_density_map)
        g_vox_density_map = input_dict.labels.density_vox[:, None]

        mae_iamge = [(g.sum()-p.sum()).abs() for (g, p) in zip(g_img_density_map, p_img_density_map)]
        mae_iamge = sum(mae_iamge) / len(mae_iamge) / n_cam

        density_vox = []
        n_mlvl = len(p.vox_density_map)
        loss_weight = [math.pow(0.3, n_mlvl-i-1) for i in range(n_mlvl)]

        for (i, mlvl_p_vox_density_map) in enumerate(p.vox_density_map):
            mlvl_g_vox_density_map = F.interpolate(g_vox_density_map, mlvl_p_vox_density_map.shape[2:], mode='trilinear')
            scale_factor = g_vox_density_map.sum(1, keepdim=True).sum(2, keepdim=True).sum(3, keepdim=True).sum(4, keepdim=True) / \
                (mlvl_g_vox_density_map.sum(1, keepdim=True).sum(2, keepdim=True).sum(3, keepdim=True).sum(4, keepdim=True) +1e-6)
            mlvl_g_vox_density_map = mlvl_g_vox_density_map * scale_factor
            density_vox.append(self.vox_density_loss_fn(mlvl_p_vox_density_map, mlvl_g_vox_density_map) * loss_weight[i])
        density_vox = sum(density_vox) / sum(loss_weight)

        mae_vox = [(g.sum() - p.sum()).abs() for (g, p) in zip(g_vox_density_map, p.vox_density_map[-1])]
        mae_vox = sum(mae_vox) / len(mae_vox)

        return edict(
            density_image = density_image,
            density_bev   = density_vox,
            total_loss    = density_image*self.loss_cfg.img_density_weight + \
                density_vox * self.loss_cfg.vox_density_weight + \
                mae_vox     * self.loss_cfg.vox_mae_weight + \
                mae_iamge   * self.loss_cfg.img_mae_weight,
            mae_image     = mae_iamge / img_density_map_scale,
            mae_bev       = mae_vox   / scene_density_map_scale,
        )
    
    def benchmark(self, input_dict, *args, **kwargs):
        # p = self.forward(input_dict)
        # print('what the fuck')
        raise "TODO: I will write the TTA asap."

    def forward(self, input_dict, *args, **kwargs):
        with torch.cuda.amp.autocast(enabled=self.cfg.get('amp', False)):
            x = input_dict.input_data.image_set
            b, n, _, _, _ = x.shape
            x = rearrange(x, 'b n ... -> (b n) ...')
            x = self.image_feature_backbone(x)
            x = self.image_feature_fusion(x)
            
            #apply camera-embedding to image-features
            if self.image_feature_embed is not None:
                x = self.image_feature_embed(mlvl_feats=x, R=input_dict)

            image_density_map = self.image_counter_head(x)
            image_density_map = rearrange(image_density_map, '(b n)... -> b n ...', b=b, n=n)
            mlvl_feats = [rearrange(feat, '(b n)... -> b n ...', b=b, n=n) for feat in x]

            #we may apply mask to maskout some spatial feature in image space
            if self.cfg.get('apply_mask', False):
                org_mask   = input_dict.input_data.image_masks
                mlvl_feats = [feat * F.interpolate(org_mask, feat.shape[3:], mode='bilinear')[:,:,None]/255.0 for feat in mlvl_feats]

            mlvl_vox_feats = self.feature_pooling(mlvl_feats=mlvl_feats, input_dict=input_dict)
            mlvl_vox_feats = self.vox_feature_fusion(mlvl_vox_feats)
            mlvl_vox_density_map = [counter(vox_feats) for (counter, vox_feats) in zip(self.counter, mlvl_vox_feats)]

            return edict(
                image_density_map=image_density_map,
                vox_density_map=mlvl_vox_density_map,
            )
        