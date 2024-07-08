

'''
    file:   transformer.py
    author: zhangxiong (1025679612@qq.com)
    date:   2023/06/18
'''

from ..register import VOXEL_POOLING
from ...attentions import build_transformer_layer
from ...embedding import EMBEDDINGS

from torch import nn as nn
import torch
import numpy as np
import warnings
import copy
from einops import rearrange
from easydict import EasyDict as edict

@VOXEL_POOLING.register_module()
class BEVAttnPooling(nn.Module):
    def __init__(self, 
        world_range, 
        bev_w, bev_h, 
        embed_dims=256, 
        use_cams_embeds=True,
        dynamic_cam_embed_cfg=dict(
            active=False,
            type='CamEmbedding',
            mode='mul',
            dim_hidden=[], 
            dim_shared=0, 
            dim_final=256,
            encode_params=['K_e', 'K_i', 'I_a']
        ),
        num_feature_levels=4,
        num_cams=3,
        num_points_in_pillar=4, 
        return_intermediate=False,
        transformerlayers=None, 
        num_layers=None, 
        init_cfg=None,
        **kwargs
    ):
        super(BEVAttnPooling, self).__init__()
        self.world_range            = world_range
        self.bev_w                  = bev_w
        self.bev_h                  = bev_h
        self.embed_dims             = embed_dims
        self.use_cams_embeds        = use_cams_embeds
        self.num_feature_levels     = num_feature_levels
        self.num_cams               = num_cams
        self.num_points_in_pillar   = num_points_in_pillar
        self.dynamic_cam_embed_cfg  = edict(dynamic_cam_embed_cfg)
        self.create_dynamic_cam_embedding(self.dynamic_cam_embed_cfg)

        if isinstance(transformerlayers, dict):
            transformerlayers = [
                copy.deepcopy(transformerlayers) for _ in range(num_layers)
            ]
        else:
            assert isinstance(transformerlayers, list) and \
                   len(transformerlayers) == num_layers
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(build_transformer_layer(transformerlayers[i]))
        self.embed_dims = self.layers[0].embed_dims
        self.pre_norm = self.layers[0].pre_norm

        self.register_parameter('bev_embedding', nn.Parameter(torch.randn(self.bev_h*self.bev_w,   self.embed_dims)))
        self.register_parameter('level_embeds',  nn.Parameter(torch.randn(self.num_feature_levels, self.embed_dims)))
        if self.use_cams_embeds:
            self.register_parameter('cams_embeds',   nn.Parameter(torch.randn(self.num_cams,           self.embed_dims)))

    def create_dynamic_cam_embedding(self, cfg):
        cfg = edict(cfg)
        if not cfg.active:
            self.dynamic_cam_embed_fn = None
            return
        assert cfg.type=='CamEmbedding'
        if cfg.dim_hidden is None or len(cfg.dim_hidden)==0:
            cfg.dim_hidden=[self.embed_dims, self.embed_dims*2, self.embed_dims]
        if cfg.mode in ['mul', 'sum']:
            cfg.dim_final = self.embed_dims
        if 'size' not in cfg or cfg.size is None:
            cfg.size=[self.bev_h, self.bev_w]
        self.dynamic_cam_embed_fn = EMBEDDINGS.build(cfg)

    def collect_img_metas(self, input_dict):
        from easydict import EasyDict as edict
        return edict(
            k_ext=input_dict.input_data.k_ext,
            k_int=input_dict.input_data.k_int,
            img_aug_aff=input_dict.aug_cfg.img_aug_aff,
            bev_aug_aff=input_dict.aug_cfg.bev_aug_aff,
            img_shape=[input_dict.input_data.image_set[b].shape[2:] for b in range(input_dict.input_data.image_set.shape[0])]
        )

    def forward(self, mlvl_feats, input_dict, **kwargs):
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        bev_queries = self.bev_embedding.unsqueeze(1).repeat(1, bs, 1)
        feat_flatten = []
        spatial_shapes = []
        cam_rigs=input_dict.input_data.cam_rigs.reshape(-1) # (b n)
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2).contiguous()
            if self.use_cams_embeds:
                cams_embeds = self.cams_embeds[cam_rigs].reshape(bs, num_cam, 1, -1).permute(1, 0, 2, 3).contiguous()
                feat = feat + cams_embeds.to(feat.dtype)
            feat = feat + self.level_embeds[None,None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=bev_queries.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        feat_flatten = feat_flatten.permute(0, 2, 1, 3).contiguous()  # (num_cam, H*W, bs, embed_dims)

        ref_3d = self.get_reference_points(H=self.bev_h, W=self.bev_w, Z=self.num_points_in_pillar, bs=bev_queries.size(1),  device=bev_queries.device, dtype=bev_queries.dtype)
        reference_points, bev_mask = self.point_sampling(ref_3d, input_dict.input_data.world_range if self.world_range is None else self.world_range, self.collect_img_metas(input_dict=input_dict))

        bev_queries = bev_queries.permute(1, 0, 2).contiguous()
        dynamic_cam_embed=None if self.dynamic_cam_embed_fn is None else self.dynamic_cam_embed_fn(input_dict)
        for lid, layer in enumerate(self.layers):
            bev_queries = layer(
                bev_queries,
                feat_flatten,
                feat_flatten,
                bev_pos=None,
                ref_3d=ref_3d,
                bev_h=self.bev_h,
                bev_w=self.bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points=reference_points,
                bev_mask=bev_mask,
                dynamic_cam_embed=dynamic_cam_embed,
                dynamic_cam_embed_cfg=self.dynamic_cam_embed_cfg,
                **kwargs
            )
        bev_queries = rearrange(bev_queries, 'b (h w) c -> b c h w', h=self.bev_h, w=self.bev_w).contiguous()
        return bev_queries
    
    @staticmethod
    def get_reference_points(H, W, Z=4, bs=1, device='cuda', dtype=torch.float):
        zs = torch.linspace(0.5, Z - 0.5, Z, dtype=dtype,device=device).view(-1, 1, 1).expand(Z, H, W) / Z
        xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device).view(1, 1, W).expand(Z, H, W) / W
        ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device).view(1, H, 1).expand(Z, H, W) / H
        ref_3d = torch.stack((xs, ys, zs), -1)
        ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1).contiguous()
        ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)  #shape: (bs,z,h*w,3)
        return ref_3d

    def point_sampling(self, reference_points, world_range, img_metas):
        w2c = img_metas.k_ext # (B, N, 3, 4)
        c2p = img_metas.k_int # (B, N, 3, 3)
        img_aff = img_metas.img_aug_aff # (B, N, 3, 3)
        bev_aff = img_metas.bev_aug_aff # (B, 4, 4)
        
        reference_points = reference_points.clone()
        if isinstance(world_range, (torch.Tensor, )): #dynamic world_range
            scale, corner = (world_range[..., 3:] - world_range[..., :3]).reshape(reference_points.shape[0], 1, 1, 3), world_range[..., :3].reshape(reference_points.shape[0], 1, 1, 3)
            reference_points = reference_points * scale + corner            
        else:
            reference_points[..., 0:1] = reference_points[..., 0:1] * (world_range[3] - world_range[0]) + world_range[0]
            reference_points[..., 1:2] = reference_points[..., 1:2] * (world_range[4] - world_range[1]) + world_range[1]
            reference_points[..., 2:3] = reference_points[..., 2:3] * (world_range[5] - world_range[2]) + world_range[2]

        reference_points = torch.cat(
            (reference_points, torch.ones_like(reference_points[..., :1])), -1)

        reference_points = reference_points.permute(1, 0, 2, 3).contiguous() #shape: (num_points_in_pillar,bs,h*w,4)

        D, B, num_query  = reference_points.size()[:3] # D=num_points_in_pillar , num_query=h*w
        bev_aff = torch.linalg.inv(bev_aff).view(1, B, 1, 4, 4)
        reference_points = torch.einsum('...ij,...j->...i',bev_aff, reference_points)

        num_cam = w2c.size(1)
        w2c = w2c.view(1, B, num_cam, 1, 3, 4)
        reference_points = reference_points.view(D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1)  #shape: (num_points_in_pillar,bs,num_cam,h*w,4)
        reference_points = torch.einsum('...ij,...j->...i', w2c, reference_points) # world space -> camera space

        img_aff = (img_aff @ c2p).view(1, B, num_cam, 1, 3, 3)
        reference_points = torch.einsum('...ij,...j->...i', img_aff, reference_points)

        eps = 1e-5
        bev_mask = (reference_points[..., 2:3] > eps)
        reference_points = reference_points[..., 0:2] / torch.maximum(reference_points[..., 2:3], torch.ones_like(reference_points[..., 2:3]) * eps)

        reference_points[..., 0] /= img_metas['img_shape'][0][1]
        reference_points[..., 1] /= img_metas['img_shape'][0][0]

        bev_mask = (bev_mask & (reference_points[..., 1:2] > 0.0)
            & (reference_points[..., 1:2] < 1.0)
            & (reference_points[..., 0:1] < 1.0)
            & (reference_points[..., 0:1] > 0.0)
        )
        
        bev_mask = torch.nan_to_num(bev_mask)

        reference_points = reference_points.permute(2, 1, 3, 0, 4).contiguous() #shape: (num_cam,bs,h*w,num_points_in_pillar,2)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1).contiguous()

        return reference_points, bev_mask

from ...attentions import build_attention, build_feedforward_network
from ...bricks import build_norm_layer
from ...helpers import ConfigDict

@VOXEL_POOLING.register_module()
class BEVCrossViewLayer(nn.Module):
    def __init__(self, 
            attn_cfgs,
            ffn_cfgs=dict(
                type='FFN',
                embed_dims=256,
                feedforward_channels=1024,
                num_fcs=2,
                ffn_drop=0.,
                act_cfg=dict(type='ReLU', inplace=True),
            ),
            operation_order=None,
            act_cfg=dict(type='ReLU', inplace=True),
            norm_cfg=dict(type='LN'),
            ffn_num_fcs=2,
            batch_first=True,
            **kwargs
        ):
        super(BEVCrossViewLayer, self).__init__()
        self.num_attn = operation_order.count('cross_attn')
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.batch_first=batch_first
        self.pre_norm = operation_order[0] == 'norm'
        self.attentions = nn.ModuleList()

        deprecated_args = dict(
            feedforward_channels='feedforward_channels',
            ffn_dropout='ffn_drop',
            ffn_num_fcs='num_fcs'
        )
        for ori_name, new_name in deprecated_args.items():
            if ori_name in kwargs:
                ffn_cfgs[new_name] = kwargs[ori_name]

        index = 0
        for operation_name in operation_order:
            if operation_name in ['self_attn', 'cross_attn']:
                if 'batch_first' in attn_cfgs[index]:
                    assert self.batch_first == attn_cfgs[index]['batch_first']
                else:
                    attn_cfgs[index]['batch_first'] = self.batch_first
                attention = build_attention(attn_cfgs[index])
                attention.operation_name = operation_name
                self.attentions.append(attention)
                index += 1

        self.embed_dims = self.attentions[0].embed_dims

        self.ffns = nn.ModuleList()
        num_ffns = operation_order.count('ffn')
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = ConfigDict(ffn_cfgs)
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
        assert len(ffn_cfgs) == num_ffns
        for ffn_index in range(num_ffns):
            if 'embed_dims' not in ffn_cfgs[ffn_index]:
                ffn_cfgs['embed_dims'] = self.embed_dims
            else:
                assert ffn_cfgs[ffn_index]['embed_dims'] == self.embed_dims
            self.ffns.append(build_feedforward_network(ffn_cfgs[ffn_index]))

        self.norms = nn.ModuleList()
        num_norms = operation_order.count('norm')
        for _ in range(num_norms):
            self.norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])

    def forward(
            self,
            query,
            key=None,
            value=None,
            bev_pos=None,
            query_pos=None,
            key_pos=None,
            attn_masks=None,
            query_key_padding_mask=None,
            key_padding_mask=None,
            ref_3d=None,
            reference_points=None,
            mask=None,
            spatial_shapes=None,
            level_start_index=None,
            prev_bev=None,
            dynamic_cam_embed=None,
            dynamic_cam_embed_cfg=None,
            **kwargs
        ):
        
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            if layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1
            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    dynamic_cam_embed=dynamic_cam_embed,
                    dynamic_cam_embed_cfg=dynamic_cam_embed_cfg,
                    **kwargs)
                attn_index += 1
                identity = query
            elif layer == 'ffn':
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1

        return query