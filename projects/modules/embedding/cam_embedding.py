
'''
    file:   cam_embedding.py
    author: zhangxiong (1025679612@qq.com)
    date:   2023/08/21
'''

from .register   import EMBEDDINGS
from ..backbones import BACKBONES

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange

@EMBEDDINGS.register_module()
class CamEmbedding(nn.Module):
    def __init__(self, size, dim_hidden, dim_final, dim_shared=0, encode_params=['K_e', 'K_i', 'I_a', 'world_range'], *args, **kwargs):
        super(CamEmbedding, self).__init__()
        self.size, self.dim_hidden, self.dim_final, self.dim_shared, self.encode_params = size, dim_hidden, dim_final, dim_shared, encode_params
        self.shared_embedding = EMBEDDINGS.get('NaiveEmbedding')([np.prod(size), dim_shared]) if dim_shared else None

        #mlp function for embedding
        dim_cam, dim_pos = 0, len(self.size) if 'pos' in encode_params else 0
        dim_cam = dim_cam + (12  if 'K_e' in encode_params else 0)
        dim_cam = dim_cam + (9   if 'K_i' in encode_params else 0)
        dim_cam = dim_cam + (9   if 'I_a' in encode_params else 0)

        self.bn = nn.BatchNorm1d(dim_cam)
        self.mlp = BACKBONES.get('MLP')(in_channels=dim_cam+dim_pos, hidden_channels=dim_hidden + [dim_final - dim_shared], norm_layer=nn.BatchNorm1d, activation_layer=nn.ReLU, inplace=True, bias=True, dropout=0.0)

    def forward(self, R, *args, **kwargs):
        def _align_extrinsic_(R):
            if 'world_range' in R.input_data and R.input_data.world_range is not None:
                world_range, e = R.input_data.world_range, R.input_data.k_ext
                b, n, _, _ = e.shape
                R, T = e[..., :3], e[..., 3]
                T = T + torch.einsum('bnij,bj->bni', R, world_range[...,:3])
                return torch.cat([R, T[...,None]], -1)
            else:
                return R.input_data.k_ext
            
        def _gen_pos_embedding_(size, dtype, device):
            if len(size)==2:
                H, W = size[0], size[1]
                xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device).view(1, W).expand(H, W) / W
                ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device).view(H, 1).expand(H, W) / H
                ref_3d = torch.stack((xs, ys), -1)
                return ref_3d.permute(2, 0, 1).flatten(1).permute(1, 0).contiguous() # shape (h*w, 2)
            else: #len(size)==3
                Z, H, W = size[0], size[1], size[2]
                zs = torch.linspace(0.5, Z - 0.5, Z, dtype=dtype, device=device).view(Z, 1, 1).expand(Z, H, W) / Z
                xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device).view(1, 1, W).expand(Z, H, W) / W
                ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device).view(1, H, 1).expand(Z, H, W) / H
                ref_3d = torch.stack((xs, ys, zs), -1)
                return ref_3d.permute(3, 0, 1, 2).flatten(1).permute(1, 0).contiguous() # shape (z*h*w, 3)

        b, n, _, _ = R.input_data.k_ext.shape

        K_e, K_i, I_a = _align_extrinsic_(R), R.input_data.k_int, R.aug_cfg.img_aug_aff
        K_e, K_i, I_a = rearrange(K_e, 'b n i j -> (b n)(i j)'), rearrange(K_i, 'b n i j -> (b n)(i j)'), rearrange(I_a, 'b n i j -> (b n)(i j)')

        num_query = np.prod(self.size)

        cam_embed = []
        if 'K_e' in self.encode_params:
            cam_embed.append(K_e)
        if 'K_i' in self.encode_params:
            cam_embed.append(K_i)
        if 'I_a' in self.encode_params:
            cam_embed.append(I_a)
        cam_embed = self.bn(torch.cat(cam_embed, -1))[:, None].expand(-1, num_query, -1)
        if 'pos' in  self.encode_params:
            pos_embed = _gen_pos_embedding_(self.size, dtype=K_e.dtype, device=K_e.device)[None].expand(b*n, -1, -1)
            embed = torch.cat([cam_embed, pos_embed], -1)
        else:
            embed = cam_embed
        
        embed = self.mlp(embed.reshape(b*n*num_query, -1)).reshape(b*n, num_query, -1)

        if self.shared_embedding is not None:
            shared_embed = self.shared_embedding()[None].expand(b*n, -1, -1)
            embed = torch.cat([embed, shared_embed], -1)

        return rearrange(embed, '(b n) q c -> b n q c', b=b, n=n).sigmoid()