
'''
    file:   image_feature_embedding.py
    author: zhangxiong (1025679612@qq.com)
    date:   2023/08/28
'''

from .register   import EMBEDDINGS
from ..backbones import BACKBONES

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

@EMBEDDINGS.register_module()
class ImageFeatureEmbedding(nn.Module):
    def __init__(self, image_neck_channels, mode, dim_hidden, encode_params=['K_e', 'K_i', 'I_a', 'world_range'], *args, **kwargs):
        super(ImageFeatureEmbedding, self).__init__()
        self.image_neck_channels, self.mode, self.dim_hidden, self.encode_params = image_neck_channels, mode, dim_hidden, encode_params

        #mlp function for embedding
        dim_cam = 0
        dim_cam = dim_cam + (12  if 'K_e'         in encode_params else 0)
        dim_cam = dim_cam + (9   if 'K_i'         in encode_params else 0)
        dim_cam = dim_cam + (9   if 'I_a'         in encode_params else 0)
        dim_cam = dim_cam + (6   if 'world_range' in encode_params else 0)

        self.bn = nn.BatchNorm1d(dim_cam)
        self.mlp = BACKBONES.get('MLP')(in_channels=dim_cam, hidden_channels=dim_hidden + [sum(self.image_neck_channels)], norm_layer=nn.BatchNorm1d, activation_layer=nn.ReLU, inplace=True, bias=True, dropout=0.0)
        self.fusion_ops = dict(
            mul=lambda feat, embed : feat * embed,
            add=lambda feat, embed : feat + embed,
        )

    def forward(self, mlvl_feats, R, *args, **kwargs):
        def _align_extrinsic_(R):
            if 'world_range' in R.input_data and R.input_data.world_range is not None:
                world_range, e = R.input_data.world_range, R.input_data.k_ext
                b, n, _, _ = e.shape
                R, T = e[..., :3], e[..., 3]
                T = T + torch.einsum('bnij,bj->bni', R, world_range[...,:3])
                return torch.cat([R, T[...,None]], -1)
            else:
                return R.input_data.k_ext
            
        b, n, _, _ = R.input_data.k_ext.shape

        K_e, K_i, I_a = _align_extrinsic_(R), R.input_data.k_int, R.aug_cfg.img_aug_aff
        K_e, K_i, I_a = rearrange(K_e, 'b n i j -> (b n)(i j)'), rearrange(K_i, 'b n i j -> (b n)(i j)'), rearrange(I_a, 'b n i j -> (b n)(i j)')

        cam_embed = []
        if 'K_e' in self.encode_params:
            cam_embed.append(K_e)
        if 'K_i' in self.encode_params:
            cam_embed.append(K_i)
        if 'I_a' in self.encode_params:
            cam_embed.append(I_a)

        if 'world_range' in self.encode_params:
            world_range = torch.Tensor(R.input_data.world_range).reshape(b, 1, 6).expand(-1, n, -1).to(K_e.device)
            world_range = rearrange(world_range, 'b n k -> (b n) k')
            cam_embed.append(world_range)

        mlvl_embeds = self.mlp(self.bn(torch.cat(cam_embed, -1))).sigmoid().split([feat.shape[1] for feat in mlvl_feats], -1)
        return [self.fusion_ops[self.mode](feat, embed[..., None, None]) for (feat, embed) in zip(mlvl_feats, mlvl_embeds)]
 