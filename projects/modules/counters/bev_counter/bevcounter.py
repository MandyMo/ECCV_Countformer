
'''
    file:   BEVCounter.py
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
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from easydict import EasyDict as edict

@COUNTERS.register_module()
class BEVCounter(nn.Module):
    def __init__(self, cfg, *args, **kwargs,):
        super(BEVCounter, self).__init__()
        self.cfg =  copy.deepcopy(cfg) 
        self.build_sub_modules(self.cfg)
        self.build_loss_fn()

    def build_sub_modules(self, cfg):
        self.image_feature_backbone = BACKBONES.build(cfg.image_feature_backbone)
        self.image_feature_fusion   = NECKS.build(cfg.image_feature_fusion)
        self.image_feature_embed    = EMBEDDINGS.build(cfg.image_feature_cam_embedding) if 'image_feature_cam_embedding' in cfg else None
        self.image_counter_head     = COUNTERS.build(cfg.image_counter_head)
        self.feature_pooling        = VOXEL_POOLING.build(cfg.feature_pooling)
        self.bev_counter_head       = COUNTERS.build(cfg.bev_counter_head)
        self.bev_feature_backbone   = BACKBONES.build(cfg.bev_feature_backbone) if 'bev_feature_backbone' in cfg else None
        self.bev_feature_fusion     = NECKS.build(cfg.bev_feature_fusion) if 'bev_feature_fusion' in cfg else None

    def build_loss_fn(self):
        default_loss_cfg=edict(
            img_density_weight=0.1,
            bev_density_weight=1.0,
            img_mae_weight=1e-7,
            bev_mae_weight=1e-4,
            img_density_loss_fn=edict(
                type='MSELoss',
            ),
            bev_density_loss_fn=edict(
                type='MSELoss',
            ),
        )
        
        cfg = self.cfg.get('loss_cfg', default_loss_cfg)
        self.img_density_loss_fn = build_loss_layer(cfg.img_density_loss_fn)
        self.bev_density_loss_fn = build_loss_layer(cfg.bev_density_loss_fn)
        self.loss_cfg=cfg

    def loss_fn(self, p, input_dict):
        img_density_map_scale, scene_density_map_scale = self.cfg.get('img_density_map_scale', 1.0), self.cfg.get('scene_density_map_scale', 1.0)
        bs, n_cam = p.image_density_map.shape[:2]
        p_img_density_map = p.image_density_map[:,:,0]
        g_img_density_map = input_dict.labels.density_image
        density_image     = self.img_density_loss_fn(p_img_density_map, g_img_density_map)

        p_bev_density_map = p.bev_density_map[:,0]
        g_bev_density_map = input_dict.labels.density_bev
        density_bev       = self.bev_density_loss_fn(p_bev_density_map, g_bev_density_map)

        mae_iamge = [(g.sum()-p.sum()).abs() for (g, p) in zip(g_img_density_map, p_img_density_map)]
        mae_bev   = [(g.sum()-p.sum()).abs() for (g, p) in zip(g_bev_density_map, p_bev_density_map)]
        mae_iamge = sum(mae_iamge) / len(mae_iamge)/n_cam
        mae_bev   = sum(mae_bev) / len(mae_bev)
        
        return edict(
            density_image = density_image,
            density_bev   = density_bev,
            total_loss    = density_image*self.loss_cfg.img_density_weight + \
                density_bev * self.loss_cfg.bev_density_weight + \
                mae_bev     * self.loss_cfg.bev_mae_weight + \
                mae_iamge   * self.loss_cfg.img_mae_weight,
            mae_image     = mae_iamge / img_density_map_scale,
            mae_bev       = mae_bev / scene_density_map_scale,
        )
    
    def benchmark(self, input_dict, *args, **kwargs):
        # p = self.forward(input_dict)
        # print('what the fuck')
        raise "TODO: I will write the TTA asap."

    def forward(self, input_dict, *args, **kwargs):
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

        bev_feat = self.feature_pooling(mlvl_feats=mlvl_feats, input_dict=input_dict)
        if self.bev_feature_backbone is not None and self.bev_feature_fusion is not None:
            bev_feat = self.bev_feature_fusion(self.bev_feature_backbone(bev_feat))[0]
        bev_density_map = self.bev_counter_head(bev_feat)
        return edict(
            image_density_map=image_density_map,
            bev_density_map=bev_density_map,
        )