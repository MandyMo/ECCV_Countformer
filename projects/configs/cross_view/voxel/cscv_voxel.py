
import math
from os import path as osp

exp_name = 'cross_view/vox/swin'

#config of the image backbone
image_bkb_indices      = [1,3,5,7]
image_bkb_channels     = [96, 192, 384, 768]
image_neck_indices     = [0,1,2,3]
image_neck_channels    = 256
_num_levels_           = len(image_neck_indices)

#the basic meta-info about the dataset
rescale = 0.5 #to reduce the system load
ori_w = int(1920 * rescale)
ori_h = int(1080 * rescale)
world_range = None #use dynamic world_range
root = 'data/cross_view'
total_view=100

#config of the head-info
down_scale=4           #the down-scale of the image space density map
_dim_ = [64, 128, 192, 256]
_ffn_dim_ = [d*2 for d in _dim_]
vox_h=[256,128,64,32]
vox_w=[256,128,64,32]
vox_z=[16,8,4,2]
_num_points_=[4,8,12,16]
_num_layers_=[3,5,7,9]

#for vox-feature fusion
dim_vox_fpn    = [_dim_[3], 256, 256, _dim_[2], 128, 128, _dim_[1], 64, 64, _dim_[0], 32, 16, 16]
stride_vox_fpn = [1,          1,   2,        1,    1,  2,        1,  1,  2,        1,  2,  1, ] 
out_feat_ids   = [2,  5, 8, 11]

#config of the dataloader
stride=32
scale=1.0
crop_w=math.ceil(ori_w*scale/stride) * stride
crop_h=math.ceil(ori_h*scale/stride) * stride
n_sel_view=5

#for debug only
vox_hm_z=32
vox_hm_h=512
vox_hm_w=512

#for heatmap-generator
img_density_map_scale=100
scene_density_map_scale=100

from easydict import EasyDict as edict

train_data_pipeline = edict(
    type='Compose',
    ops=[
        edict(
            type='FilterOutlier',
            employ_mask=False,
            world_range=None,
        ),
        edict(
            type='ImageAffineTransform',
            scale=[0.8*scale, 1.2*scale], 
            angle=[-5.4, 5.4], 
            center=[[0.45, 0.55], [0.45, 0.55]], 
            crop=['random', 'random', crop_w, crop_h],
            p_hf=0.5,
            p_vf=0.0,
            iso_aff=False,
            center_coord='ratio',
            crop_coord='pixel',
        ),
        edict(
            type='BEVAffineTransform',
            scale=[0.8, 1.2],
            angle=[[-5.4, 5.4], [-5.4, 5.4], [-180, 180]],
            trans=None, #use dynamic trans
            p_xf=0.5,
            p_yf=0.5,
            p_zf=0.5,
        ),
        edict(
            type='FilterOutlier',
            employ_mask=False,
            world_range=world_range,
            filter_z=True,
        ),
        edict(
            type='Oneof',
            ops=[
                edict(type='ProbOp', op=edict(type='GammaAdj', scale=[0.7, 1.3], iso_color=False, iso_cam_rigs=False),p=0.8),
                edict(type='ProbOp', op=edict(type='Gray', iso=False), p=0.3),
                edict(type='ProbOp', op=edict(type='ChannelShuffle', iso_cam_rigs=False), p=0.8),
            ],
            probs=[1, 1, 0.5],
        ),
        edict(
            type='GenDensity3D',
            down_scale=down_scale,
            gen_image_density_map=True,
            gen_vox_density_map=True,
            world_range=world_range,
            vox_hm_z=vox_hm_z,
            vox_hm_h=vox_hm_h,
            vox_hm_w=vox_hm_w,
            img_density_generator=edict(type='GaussianFilterDensityMapGenerator', sigma=3, scale=img_density_map_scale),
            vox_density_generator=edict(type='GaussianFilterDensityMapGenerator', sigma=3, scale=scene_density_map_scale),
        ),
    ]
)

val_data_pipeline = edict(
    type='Compose',
    ops=[
        edict(
            type='ImageAffineTransform',
            scale=scale, 
            angle=0, 
            center=[0.5, 0.5], 
            crop=['center', 'bottom', crop_w, crop_h],
            p_hf=0.0,
            p_vf=0.0,
            iso_aff=False,
            center_coord='ratio',
            crop_coord='pixel',
        ),
        edict(
            type='FilterOutlier',
            employ_mask=False,
            world_range=world_range,
            filter_z=True,
        ),
        edict(
            type='GenDensity3D',
            down_scale=down_scale,
            gen_image_density_map=True,
            gen_vox_density_map=True,
            world_range=world_range,
            vox_hm_z=vox_hm_z,
            vox_hm_h=vox_hm_h,
            vox_hm_w=vox_hm_w,
            img_density_generator=edict(type='GaussianFilterDensityMapGenerator', sigma=3, scale=img_density_map_scale),
            vox_density_generator=edict(type='GaussianFilterDensityMapGenerator', sigma=3, scale=scene_density_map_scale),
        ),
    ]
)

test_data_pipeline = edict( # test-time augment
    type='BEVTTA',
    image_scales=[1.0*scale, 0.8*scale, 0.9*scale, 1.1*scale],
    bev_scales=[1.0],
    bev_xf=[False, True],
    bev_yf=[False, True],
    bev_rots=[0,-90,-45,45,90],
    crop_info=['center', 'bottom', crop_w, crop_h],
    pt_filter=edict(
        type='FilterOutlier',
        employ_mask=True,
        world_range=world_range,
        filter_z=True,
    ),
    density_generator = edict(
        type='GenDensity3D',
        down_scale=down_scale,
        gen_image_density_map=True,
        gen_vox_density_map=True,
        world_range=world_range,
        vox_hm_z=vox_hm_z,
        vox_hm_h=vox_hm_h,
        vox_hm_w=vox_hm_w,
        img_density_generator=edict(type='GaussianFilterDensityMapGenerator', sigma=3, scale=img_density_map_scale),
        vox_density_generator=edict(type='GaussianFilterDensityMapGenerator', sigma=3, scale=scene_density_map_scale),
    ),
)

data = edict(
    train=edict(
        dataset=edict(
            type='CrossViewCrossScene', 
            root=root, 
            mode='train', 
            ncam=n_sel_view, 
            ori_h=ori_h, 
            ori_w=ori_w, 
            rescale=rescale,
            extend=1.0,
            prun_unvisible_pt_bev=True,
            bev_trans_scale=0.1,
            static_range=[102.4, 102.4, 6.4],
            data_pipeline=train_data_pipeline,
        ),
        num_workers=4,
        batch_size=1,
        persistent_workers=True,
        collate_fn='batch_collect_fn',
        pin_memory=True,
    ),
    val=edict(
        dataset=edict(
            type='CrossViewCrossScene', 
            root=root, 
            mode='test', 
            ncam=n_sel_view, 
            ori_h=ori_h, 
            ori_w=ori_w, 
            rescale=rescale,
            extend=1.0,
            prun_unvisible_pt_bev=True,
            bev_trans_scale=0.0,
            static_range=[102.4, 102.4, 6.4],
            data_pipeline=val_data_pipeline,
        ),
        num_workers=4,
        batch_size=1,
        persistent_workers=True,
        collate_fn='batch_collect_fn',
        pin_memory=True,
    ),
    test=edict(
        dataset=edict(
            type='CrossViewCrossScene', 
            root=root, 
            mode='test', 
            ncam=n_sel_view, 
            ori_h=ori_h, 
            ori_w=ori_w, 
            rescale=rescale,
            extend=1.0,
            prun_unvisible_pt_bev=True,
            bev_trans_scale=0.0,
            static_range=[102.4, 102.4, 6.4],
            data_pipeline=test_data_pipeline,
        ),
        num_workers=2,
        batch_size=1,
        persistent_workers=True,
        collate_fn='tta_batch_collect_fn',
        pin_memory=True,
    )
)

model = edict(
    type='VOXCounter',
    # image_feature_backbone=edict(
    #     type='resnet50', 
    #     out_feature_indices=image_bkb_indices, 
    #     n_feature_channels =image_bkb_channels,
    #     pretrained=True,
    # ),
    img_density_map_scale=img_density_map_scale,
    scene_density_map_scale=scene_density_map_scale,
    
    image_feature_backbone=edict(
        type='swin_t', 
        pretrained=True, 
        progress=True,
    ),
    image_feature_fusion=edict(
        type='SimpleFPN',
        in_channels=image_bkb_channels,
        out_channels=image_neck_channels,
        num_outs=len(image_neck_indices),
        start_level=0,
        out_ids=image_neck_indices,
    ),
    image_feature_cam_embedding=edict(
        type='ImageFeatureEmbedding',
        image_neck_channels=[image_neck_channels for _ in vox_z],
        encode_params=['K_e', 'K_i', 'I_a', 'world_range'],
        mode='mul',
        dim_hidden=[256,256,256], 
    ),
    image_counter_head=edict(
        type='CSRNet',
        net_cfg = [512, 256, 256, 128, 128, 64, 64],
        in_channels=image_neck_channels*len(image_neck_indices),
        stack=True,
    ),
    feature_pooling=dict(
        type='MSVoxAttnPooling',
        norm_cfg=dict(type='BN', ),
        embed_dims=_dim_,
        vox_z=vox_z,
        vox_h=vox_h,
        vox_w=vox_w,  
        num_points=_num_points_,
        ffn_dims=_ffn_dim_,
        image_neck_channels=[image_neck_channels for _ in vox_z],
        num_cams=total_view, #total views
        world_range=world_range,
        return_intermediate=False,
        num_layers=_num_layers_,
        use_cams_embeds=False,
        use_lvls_embeds=False,
        transformerlayers=dict(
            type='VOXCrossViewLayer',
            attn_cfgs=[
                dict(
                    type='SpatialCrossAttention',
                    embed_dims=_dim_,
                    num_cams=n_sel_view,
                    deformable_attention=dict(
                        type='MSDeformableAttention3D',
                        embed_dims=_dim_,
                        num_points=_num_points_,
                        num_levels=_num_levels_
                    ),
                    attention_cross_all_level=True,
                )
            ],
            feedforward_channels=_ffn_dim_,
            ffn_dropout=0.1,
            operation_order=('cross_attn', 'norm', 'ffn', 'norm', 'conv')
        ),
        dynamic_cam_embed_cfg=dict(
            active=True,
            type='CamEmbedding',
            mode='mul',
            dim_hidden=[], 
            dim_shared=0, 
            dim_final=256,
            encode_params=['K_e', 'K_i', 'I_a', 'pos']
        ),
    ),
    vox_feature_fusion=edict(
        type='SimpleFPN3d',
        dim_vox_fpn=dim_vox_fpn, 
        stride_vox_fpn=stride_vox_fpn, 
        out_feat_ids=out_feat_ids
    ),
    vox_counter_head=edict(
        dims_in=[dim_vox_fpn[_id+1] for _id in out_feat_ids],
        dims=[32, 16, 16, 1],
        # act_layer=edict(
        #     type='Softplus',
        # )
    ),
    loss_cfg=edict(
        img_density_weight=0.1,
        vox_density_weight=1.0,
        img_mae_weight=0.0,
        vox_mae_weight=0.0,
        img_density_loss_fn=edict(
            type='MSELoss',
        ),
        vox_density_loss_fn=edict(
            type='SmoothL1Loss',
            beta=0.1,
        ),
    ),
)

optimization = edict(
    optimizer=edict(
        activate='AdamW',
        RMSprop=edict(
            lr=2e-4,
            momentum=0.7,
            weight_decay=1e-2,
        ),
        Adam=edict(
            lr=8e-4,
            weight_decay=1e-2,
        ),
        AdamW=edict(
            lr=1e-5,
            weight_decay=0.01,
        ),
        clip_norm=True,
        max_norm=1.0,
        scale_lr=True,
        param_group_cfg={
            'counter':5.0,
            'vox_feature_fusion':5.0,
        },
    ),
    train_cfg=edict(
        epoches=400,
        resume=False,
        chkp_path=None,
        sync_bn=True,
    ),
    lr_schedule=edict(
        activate='LinearStepScheduler',
        PolyScheduler = edict(
            warmup_lr_ratio=1e-3,
            warmup_ratio=0.01,
            power=0.9,
        ),
        LinearStepScheduler = edict(
            gamma=0.1,
            warmup_lr_ratio=0.001,
            warmup_ratio=0,
            mile_stones=[0.6, 0.8, 0.9],
        ),
        CosineScheduler = edict(
            total_cycles=3, 
            min_lr_ratio=1e-4,
            warmup_ratio=200,
            warmup_lr_ratio=0.001,
        ),
    ),
    logger_cfg=edict(
        tf_log_loss_period=5,      #dump loss
        tf_log_record_period=30,   #dump training samples
        log_record_fn='visualize_vox_counting_samples', #dump fn
        log_dir=osp.join('workdirs', exp_name),
        chkp_sv_path=osp.join('workdirs', exp_name),
        chkp_sv_period=1,
    )
)