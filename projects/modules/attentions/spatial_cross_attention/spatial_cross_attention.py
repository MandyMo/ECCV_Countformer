

'''
    file:   spatial_cross_attention.py
    author: zhangxiong (1025679612@qq.com)
    date:   2023/06/19
'''


from ..register import ATTENTION, build_attention
from ...bricks import xavier_init

import itertools
import torch
import torch.nn as nn

from einops import rearrange


@ATTENTION.register_module()
class SpatialCrossAttention(nn.Module):
    def __init__(self,
            embed_dims=256,
            num_cams=6,
            pc_range=None,
            dropout=0.1,
            init_cfg=None,
            batch_first=False,
            deformable_attention=dict(
                type='MSDeformableAttention3D',
                embed_dims=256,
                num_levels=4
            ),
            attention_cross_all_level=False,
            **kwargs 
        ):
        super(SpatialCrossAttention, self).__init__()
        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.fp16_enabled = False
        self.deformable_attention = build_attention(deformable_attention)
        self.embed_dims = embed_dims
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.batch_first = batch_first
        self.attention_cross_all_level=attention_cross_all_level
        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    def fuse_dynamic_cam_embed(self, queries, indexes, cam_embed=None, cam_embed_cfg=None):
        if cam_embed is None or cam_embed_cfg is None:
            return queries
        bs, ncam, _, _ = cam_embed.shape
        fused_queries = torch.zeros_like(queries)
        for j in range(bs):
            for i in range(ncam): 
                index_query_per_img = indexes[i][j]
                if cam_embed_cfg.mode=='mul':
                    fused_queries[j, i, :len(index_query_per_img)] = queries[j, i, :len(index_query_per_img)] * cam_embed[j, i, index_query_per_img]
                else: #sum
                    fused_queries[j, i, :len(index_query_per_img)] = queries[j, i, :len(index_query_per_img)] + cam_embed[j, i, index_query_per_img]
        return fused_queries

    def get_attention_weights(self, queries, indexes, max_len):
        bs, num_cams, num_heads, num_levels, num_points = len(queries), len(queries[0]), self.deformable_attention.num_heads, self.deformable_attention.num_levels, self.deformable_attention.num_points
        atten_weights = self.deformable_attention.attention_weights(queries)
        new_atten_weights = torch.zeros_like(atten_weights)-10.0 # b n q (h l p)

        #gather attention weights
        for j in range(bs):
            for i in range(num_cams):
                index_query_per_img = indexes[i][j]
                new_atten_weights[j,i,index_query_per_img] = atten_weights[j,i,index_query_per_img]
        #conduct the normalization operation, softmax (n l p)
        atten_weights = rearrange(new_atten_weights, 'b n q (h l p) -> b h q (n l p)', n=num_cams, l=num_levels, p=num_points).softmax(-1)
        atten_weights = rearrange(atten_weights, 'b h q (n l p) -> b n q (h l p)', n=num_cams, l=num_levels, p=num_points).contiguous()
        atten_weights_rebatch = atten_weights.new_zeros([bs, num_cams, max_len, atten_weights.shape[-1]])

        for j in range(bs):
            for i in range(num_cams):   
                index_query_per_img = indexes[i][j]
                atten_weights_rebatch[j, i, :len(index_query_per_img)] = atten_weights[j,i,index_query_per_img]
        return atten_weights_rebatch

    def forward(self,
            query,
            key,
            value,
            residual=None,
            query_pos=None,
            key_padding_mask=None,
            reference_points=None,
            spatial_shapes=None,
            reference_points_cam=None,
            bev_mask=None,
            level_start_index=None,
            dynamic_cam_embed=None,
            dynamic_cam_embed_cfg=None,
            **kwargs
        ):
        
        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
            slots = torch.zeros_like(query)
        if query_pos is not None:
            query = query + query_pos
        num_cams, l, bs, embed_dims = key.shape
        bs, num_query, _ = query.size()
        # bevformer reference_points_cam shape: (num_cam,bs,h*w,num_points_in_pillar,2)
        D = reference_points_cam.size(3)
        indexes = []
        for i, batch_mask_per_img in enumerate(bev_mask):
            index_query_per_img = [mask_per_img.sum(-1).nonzero().squeeze(-1) for mask_per_img in batch_mask_per_img]
            indexes.append(index_query_per_img)
        max_len = max([len(each) for batch_indexes in indexes for each in batch_indexes])

        # each camera only interacts with its corresponding BEV queries. This step can  greatly save GPU memory.
        queries_rebatch = query.new_zeros(
            [bs, num_cams, max_len, self.embed_dims])
        reference_points_rebatch = reference_points_cam.new_zeros(
            [bs, num_cams, max_len, D, 2])
        
        if dynamic_cam_embed is not None and dynamic_cam_embed_cfg is not None:
            fusion_op = (lambda a, b: a*b) if dynamic_cam_embed_cfg.mode=='mul' else (lambda a, b: a + b)
        else:
            fusion_op = None
        
        fused_query = query[:, None].expand(-1, num_cams, -1, -1)
        fused_query = fusion_op(fused_query, dynamic_cam_embed) if fusion_op is not None else fused_query

        for j in range(bs):
            for i, reference_points_per_img in enumerate(reference_points_cam[:,j, ...]):   
                index_query_per_img = indexes[i][j]
                queries_rebatch[j, i, :len(index_query_per_img)] = fused_query[j, i, index_query_per_img]
                reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[index_query_per_img]

        attention_weights = self.get_attention_weights(fused_query, indexes, max_len) if self.attention_cross_all_level else None

        num_cams, l, bs, embed_dims = key.shape

        key = key.permute(2, 0, 1, 3).reshape(
            bs * num_cams, l, self.embed_dims).contiguous()
        value = value.permute(2, 0, 1, 3).reshape(
            bs * num_cams, l, self.embed_dims).contiguous()

        queries = self.deformable_attention(
            query=queries_rebatch.view(bs*num_cams, max_len, self.embed_dims), 
            key=key,
            value=value,
            reference_points=reference_points_rebatch.view(bs*num_cams, max_len, D, 2), 
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index, 
            num_cams=num_cams if self.attention_cross_all_level else None,
            atten_weights=attention_weights,
        ).view(bs, num_cams, max_len, self.embed_dims)
        
        for (i, j) in itertools.product(range(bs), range(num_cams)):
            index_query_per_img = indexes[j][i]
            slots[i, index_query_per_img] += queries[i, j, :len(index_query_per_img)]

        if not self.attention_cross_all_level:
            count = bev_mask.sum(-1) > 0
            count = count.permute(1, 2, 0).sum(-1).contiguous()
            count = torch.clamp(count, min=1.0)
            slots = slots / count[..., None]
            
        slots = self.output_proj(slots)
        return self.dropout(slots) + inp_residual