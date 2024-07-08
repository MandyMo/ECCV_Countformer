

'''
    file:   vox_cross_view.py
    author: zhangxiong (1025679612@qq.com)
    date:   2023/06/18
'''

from ..register import VOXEL_POOLING
from ...attentions import build_transformer_layer
from ...embedding  import EMBEDDINGS

from torch import nn as nn
import torch
import numpy as np
import warnings
import copy
from einops import rearrange
from easydict import EasyDict as edict

@VOXEL_POOLING.register_module()
class MSVoxAttnPooling(nn.Module):
    def __init__(self,
        world_range,
        vox_z, 
        vox_h, 
        vox_w,
        embed_dims,
        image_neck_channels,
        num_points,
        ffn_dims,
        use_cams_embeds=False,
        use_lvls_embeds=False,
        num_cams=3,
        return_intermediate=False,
        transformerlayers=None, 
        num_layers=None, 
        init_cfg=None,
        dynamic_cam_embed_cfg=dict(
            active=False,
            type='CamEmbedding',
            mode='mul',
            dim_hidden=[], 
            dim_shared=0, 
            dim_final=256,
            encode_params=['K_e', 'K_i', 'I_a']
        ),
        **kwargs
    ):
        super(MSVoxAttnPooling, self).__init__()
        self.fpn_level = len(embed_dims)

        #create multi-scale vox pooling layer
        self.transfer_conv, self.vox_layers = nn.ModuleList(), nn.ModuleList()
        for _fpn_level in range(self.fpn_level):
            _transformerlayers = copy.deepcopy(transformerlayers)
            _vox_z, _vox_h, _vox_w, _layer = vox_z[_fpn_level], vox_h[_fpn_level], vox_w[_fpn_level], num_layers[_fpn_level]
            ch_input, embed_dim, num_point, ffn_dim = image_neck_channels[_fpn_level], embed_dims[_fpn_level], num_points[_fpn_level], ffn_dims[_fpn_level]

            #channel alignment layer
            conv_cfg=dict(type='Conv2d', bias=True)
            transfer_layer = build_conv_layer(
                conv_cfg,
                in_channels=ch_input,
                out_channels=embed_dim,
                kernel_size=1,
                stride=1
            )
            transfer_block = nn.Sequential(transfer_layer)
            self.transfer_conv.append(transfer_block)

            _transformerlayers = copy.deepcopy(transformerlayers)
            _transformerlayers.attn_cfgs[0].embed_dims = embed_dim
            _transformerlayers.attn_cfgs[0].deformable_attention.embed_dims = embed_dim
            _transformerlayers.attn_cfgs[0].deformable_attention.num_points = num_point
            _transformerlayers.attn_cfgs[0].deformable_attention.num_levels = 1
            _transformerlayers.attn_cfgs[0].num_points = num_point
            _transformerlayers.attn_cfgs[0].num_levels = 1
            _transformerlayers.feedforward_channels = ffn_dim
            _transformerlayers.embed_dims = embed_dim

            vox_layer = VOXAttnPooling(
                world_range=world_range,
                vox_h=_vox_h,
                vox_w=_vox_w,
                vox_z=_vox_z,
                embed_dims=embed_dim,
                use_cams_embeds=use_cams_embeds,
                use_lvls_embeds=use_lvls_embeds,
                num_feature_levels=1,
                num_cams=num_cams,
                num_layers=_layer,
                transformerlayers=_transformerlayers,
                dynamic_cam_embed_cfg=dynamic_cam_embed_cfg,
            )
            self.vox_layers.append(vox_layer)
            
    def forward(self, mlvl_feats, input_dict, **kwargs):
        bs, ncam, _, _, _ = mlvl_feats[0].shape
        mlvl_feats = [trans_conv(rearrange(feat, 'b n c h w -> (b n) c h w')) for (trans_conv, feat) in zip(self.transfer_conv, mlvl_feats)]
        mlvl_feats = [rearrange(feat, '(b n) c h w -> b n c h w', b=bs, n=ncam) for feat in mlvl_feats]
        
        feats_3d = []
        for (feat, vox_atten) in zip(mlvl_feats, self.vox_layers):
            feats_3d.append(vox_atten([feat], input_dict))

        return feats_3d


@VOXEL_POOLING.register_module()
class VOXAttnPooling(nn.Module):
    def __init__(self, 
        world_range, 
        vox_z,
        vox_h,
        vox_w,  
        embed_dims=256, 
        use_cams_embeds=False,
        dynamic_cam_embed_cfg=dict(
            active=False,
            type='CamEmbedding',
            mode='mul',
            dim_hidden=[], 
            dim_shared=0, 
            dim_final=256,
            encode_params=['K_e', 'K_i', 'I_a']
        ),
        use_lvls_embeds=False,
        num_feature_levels=4,
        num_cams=3,
        return_intermediate=False,
        transformerlayers=None, 
        num_layers=None, 
        init_cfg=None,
        **kwargs
    ):
        super(VOXAttnPooling, self).__init__()
        self.world_range             = world_range
        self.vox_w                   = vox_w
        self.vox_h                   = vox_h
        self.vox_z                   = vox_z
        self.embed_dims              = embed_dims
        self.use_cams_embeds         = use_cams_embeds
        self.use_lvls_embeds         = use_lvls_embeds
        self.num_feature_levels      = num_feature_levels
        self.num_cams                = num_cams

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

        self.register_parameter('vox_embedding', nn.Parameter(torch.randn(self.vox_z*self.vox_h*self.vox_w,   self.embed_dims)))

        if self.use_lvls_embeds:
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
            cfg.size=[self.vox_z, self.vox_h, self.vox_w]
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
        vox_queries = self.vox_embedding.unsqueeze(1).repeat(1, bs, 1)
        feat_flatten = []
        spatial_shapes = []
        cam_rigs=input_dict.input_data.cam_rigs.reshape(-1) # (b n)
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2).contiguous()

            #use cam & level embeddings
            if self.use_cams_embeds:
                cams_embeds = self.cams_embeds[cam_rigs].reshape(bs, num_cam, 1, -1).permute(1, 0, 2, 3).contiguous()
                feat = feat + cams_embeds.to(feat.dtype)

            if self.use_lvls_embeds:
                feat = feat + self.level_embeds[None,None, lvl:lvl + 1, :].to(feat.dtype)

            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=vox_queries.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        feat_flatten = feat_flatten.permute(0, 2, 1, 3).contiguous()  # (num_cam, H*W, bs, embed_dims)

        ref_3d = self.get_reference_points(H=self.vox_h, W=self.vox_w, Z=self.vox_z, bs=vox_queries.size(1),  device=vox_queries.device, dtype=vox_queries.dtype)
        reference_points, bev_mask = self.point_sampling(ref_3d, input_dict.input_data.world_range if self.world_range is None else self.world_range, self.collect_img_metas(input_dict=input_dict))

        vox_queries = vox_queries.permute(1, 0, 2).contiguous()
        dynamic_cam_embed=None if self.dynamic_cam_embed_fn is None else self.dynamic_cam_embed_fn(input_dict)
        for lid, layer in enumerate(self.layers):
            vox_queries = layer(
                vox_queries,
                feat_flatten,
                feat_flatten,
                bev_pos=None,
                ref_3d=ref_3d,
                vox_h=self.vox_h,
                vox_w=self.vox_w,
                vox_z=self.vox_z,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points=reference_points,
                bev_mask=bev_mask,
                dynamic_cam_embed=dynamic_cam_embed,
                dynamic_cam_embed_cfg=self.dynamic_cam_embed_cfg,
                **kwargs
            )
        vox_queries = rearrange(vox_queries, 'b (z h w) c -> b c z h w', z=self.vox_z, h=self.vox_h, w=self.vox_w).contiguous()
        return vox_queries
    
    @staticmethod
    def get_reference_points(H, W, Z, bs, device='cuda', dtype=torch.float):
        zs = torch.linspace(0.5, Z - 0.5, Z, dtype=dtype, device=device).view(Z, 1, 1).expand(Z, H, W) / Z
        xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device).view(1, 1, W).expand(Z, H, W) / W
        ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device).view(1, H, 1).expand(Z, H, W) / H
        ref_3d = torch.stack((xs, ys, zs), -1)
        ref_3d = ref_3d.permute(3, 0, 1, 2).flatten(1).permute(1, 0).contiguous() # shape (z*h*w, 3)
        ref_3d = ref_3d[None, None].repeat(bs, 1, 1, 1) #shape (bs, 1, z*h*w, 3)
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

        reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1) # shape: (bs, D=1, z*h*w, 4) 

        reference_points = reference_points.permute(1, 0, 2, 3).contiguous() #shape: (D,bs,z*h*w,4)

        D, B, num_query  = reference_points.size()[:3] # D=1, B=bs, num_query=z*h*w
        bev_aff = torch.linalg.inv(bev_aff).view(1, B, 1, 4, 4)
        reference_points = torch.einsum('...ij,...j->...i',bev_aff, reference_points) # (D, bs, z*h*w, 4)

        num_cam = w2c.size(1)
        w2c = w2c.view(1, B, num_cam, 1, 3, 4)
        reference_points = reference_points.view(D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1)  #shape: (D, bs, num_cam, z*h*w,4)
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

        reference_points = reference_points.permute(2, 1, 3, 0, 4).contiguous() #shape: (num_cam, bs, num_query, D, 2)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1).contiguous()     #shape: (num_cam, bs, num_query, D)

        return reference_points, bev_mask

from ...attentions import build_attention, build_feedforward_network
from ...bricks     import build_norm_layer, build_conv_layer
from ...helpers    import ConfigDict

@VOXEL_POOLING.register_module()
class VOXCrossViewLayer(nn.Module):
    def __init__(self, 
            attn_cfgs,
            ffn_cfgs=dict(
                type='FFN',
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
        super(VOXCrossViewLayer, self).__init__()
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
                ffn_cfgs[ffn_index]['embed_dims'] = self.embed_dims
            else:
                assert ffn_cfgs[ffn_index]['embed_dims'] == self.embed_dims
            self.ffns.append(build_feedforward_network(ffn_cfgs[ffn_index]))
        
        #build norm
        self.norms = nn.ModuleList()
        num_norms = operation_order.count('norm')
        for _ in range(num_norms):
            self.norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])

        #build conv
        num_convs = operation_order.count('ffn')
        self.deblock = nn.ModuleList()
        conv_cfg=dict(type='Conv3d', bias=False)
        norm_cfg=dict(type='GN', num_groups=16, requires_grad=True)
        for i in range(num_convs):
            conv_layer = build_conv_layer(
                conv_cfg,
                in_channels=self.embed_dims,
                out_channels=self.embed_dims,
                kernel_size=3,
                stride=1,
                padding=1
            )
            deblock = nn.Sequential(conv_layer,
                build_norm_layer(norm_cfg, self.embed_dims)[1],
                nn.ReLU(inplace=True)
            )
            self.deblock.append(deblock)

    def forward(
            self,
            query,
            key=None,
            value=None,
            bev_pos=None,
            vox_z=None,
            vox_h=None,
            vox_w=None,
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
            if layer=='conv':
                bs = query.shape[0]
                identity = query
                query = query.reshape(bs, vox_z, vox_h, vox_w, -1).permute(0, 4, 3, 2, 1)
                for i in range(len(self.deblock)):
                    query = self.deblock[i](query)
                query = query.permute(0, 4, 3, 2, 1).reshape(bs, vox_z*vox_h*vox_w, -1)
                query = query + identity
            elif layer == 'norm':
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