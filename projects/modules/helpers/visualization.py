
'''
    file:   visualization.py
    author: zhangxiong (1025679612@qq.com)
    date:   2023/06/25
'''

from .register import HELPERS
import torch.nn as nn
import numpy as np
import cv2

@HELPERS.register_module()
def draw_points_in_image(image, pts, radius=2, color=(0,0,255), thickness=2, lineType=-1, shift=None, **kwargs):
    for pt in pts:
        x, y = int(pt[0]+0.5), int(pt[1]+0.5)
        cv2.circle(image, (x,y), radius=radius, color=color, thickness=thickness, lineType=lineType, shift=shift, **kwargs)
    return image

@HELPERS.register_module()
def visualize_bev_counting_samples(logger, batch, p, step, prefix, n_max_sample=4):
    def normalize_density_map(density_map, a_min=None, colormap=cv2.COLORMAP_JET):
        a_max = max(density_map.max(), 0)
        a_min = 0.0 if a_min is None else a_min
        density_map = np.clip(density_map, a_min, a_max)
        density_map = density_map / (a_max+1e-6)
        density_map = (density_map*255.0).astype(np.uint8)
        density_map = cv2.applyColorMap(density_map, colormap)
        return density_map

    from ...dataset import unnormalize_image
    fn_draw = draw_points_in_image
    bs = batch.input_data.image_set.shape[0]
    g_batch_img_density_maps, g_batch_bev_density_maps = [], []
    p_batch_img_density_maps, p_batch_bev_density_maps = [], []
    batch_vis_per_pt, batch_vis_bev_pt = [], []
    
    ds=2

    for b in range(min(bs, n_max_sample)):
        vis_per_pt, vis_bev_pt, g_img_density_maps, p_img_density_maps = [], [], [], []
        pt_bev = batch.metas.pt_bev[b]
        bev_aug_aff = batch.aug_cfg.bev_aug_aff[b].cpu().numpy()
        g_batch_bev_density_maps.append(normalize_density_map(batch.labels.density_bev[b].cpu().numpy()))
        p_batch_bev_density_maps.append(normalize_density_map(p.bev_density_map[b, 0].cpu().numpy()))
        
        inv_bev_aug_aff = np.linalg.inv(bev_aug_aff)
        o_pt_bev = np.einsum('ij,nj->ni', inv_bev_aug_aff[:3,:3],pt_bev) + inv_bev_aug_aff[:3,3:4].T
        for (image, mask, pt_per, g_density_map, p_density_map, K_ext, K_int, img_aug_aff) in zip(
            batch.input_data.image_set[b], batch.input_data.image_masks[b], batch.metas.pt_image[b], batch.labels.density_image[b], p.image_density_map[b], 
            batch.input_data.k_ext[b], batch.input_data.k_int[b], batch.aug_cfg.img_aug_aff[b]
        ):
            K_ext, K_int, img_aug_aff = K_ext.cpu().numpy(), K_int.cpu().numpy(), img_aug_aff.cpu().numpy()
            pt_bev = np.einsum('ij,nj->ni', K_ext[:3,:3], o_pt_bev) + K_ext[:3, 3:4].T
            pt_bev = np.einsum('ij,nj->ni', img_aug_aff@K_int, pt_bev)
            pt_bev = pt_bev[:, :2] / pt_bev[:, 2:3]
            image, mask = unnormalize_image(image.cpu().numpy()).copy(), mask.cpu().numpy()[...,None]
            image = (image * 0.8 + mask * 0.2).astype(np.uint8)
            image = cv2.resize(image, (image.shape[1]//ds, image.shape[0]//ds))
            cv2.rectangle(image, (0, 0), (image.shape[1]-1, image.shape[0]-1), (0,255,0), 5)
            image_bev = image.copy()
            image_per = image.copy()
            fn_draw(image_per, pt_per/ds)
            fn_draw(image_bev, pt_bev/ds)
            vis_per_pt.append(image_per)
            vis_bev_pt.append(image_bev)
            g_img_density_maps.append(normalize_density_map(g_density_map.cpu().numpy()))
            p_img_density_maps.append(normalize_density_map(p_density_map.cpu().numpy()[0]))
        batch_vis_per_pt.append(np.concatenate(vis_per_pt, 1))
        batch_vis_bev_pt.append(np.concatenate(vis_bev_pt, 1))
        g_batch_img_density_maps.append(np.concatenate(g_img_density_maps, 1))
        p_batch_img_density_maps.append(np.concatenate(p_img_density_maps, 1))

    vis_img_pt             = np.concatenate([np.concatenate(batch_vis_per_pt, 1), np.concatenate(batch_vis_bev_pt, 1)], 0)
    batch_img_density_maps = np.concatenate([np.concatenate(g_batch_img_density_maps, 1), np.concatenate(p_batch_img_density_maps, 1)], 0)
    batch_bev_density_maps = np.concatenate([np.concatenate(g_batch_bev_density_maps, 1), np.concatenate(p_batch_bev_density_maps, 1)], 0)
    logger.add_images(prefix+'/vis_img_pt',      vis_img_pt,             global_step=step, dataformats='HWC')
    logger.add_images(prefix+'/img_density_map', batch_img_density_maps, global_step=step, dataformats='HWN')
    logger.add_images(prefix+'/bev_density_map', batch_bev_density_maps, global_step=step, dataformats='HWN')


@HELPERS.register_module()
def visualize_vox_counting_samples(logger, batch, p, step, prefix, n_max_sample=4):
    def normalize_density_map(density_map, a_min=None, colormap=cv2.COLORMAP_JET):
        a_max = max(density_map.max(), 0)
        a_min = 0.0 if a_min is None else a_min
        density_map = np.clip(density_map, a_min, a_max)
        density_map = density_map / (a_max+1e-6)
        density_map = (density_map*255.0).astype(np.uint8)
        density_map = cv2.applyColorMap(density_map, colormap)
        return density_map

    from ...dataset import unnormalize_image
    fn_draw = draw_points_in_image
    bs = batch.input_data.image_set.shape[0]
    g_batch_img_density_maps, g_batch_bev_density_maps = [], []
    p_batch_img_density_maps, p_batch_bev_density_maps = [], []
    batch_vis_per_pt, batch_vis_bev_pt = [], []
    
    ds=2

    for b in range(min(bs, n_max_sample)):
        vis_per_pt, vis_bev_pt, g_img_density_maps, p_img_density_maps = [], [], [], []
        pt_bev = batch.metas.pt_bev[b]
        bev_aug_aff = batch.aug_cfg.bev_aug_aff[b].cpu().numpy()
        g_batch_bev_density_maps.append(normalize_density_map(batch.labels.density_vox[b].sum(0).cpu().numpy()))
        p_batch_bev_density_maps.append(normalize_density_map(p.vox_density_map[-1][b, 0].sum(0).cpu().numpy()))
        
        inv_bev_aug_aff = np.linalg.inv(bev_aug_aff)
        o_pt_bev = np.einsum('ij,nj->ni', inv_bev_aug_aff[:3,:3],pt_bev) + inv_bev_aug_aff[:3,3:4].T
        for (image, mask, pt_per, g_density_map, p_density_map, K_ext, K_int, img_aug_aff) in zip(
            batch.input_data.image_set[b], batch.input_data.image_masks[b], batch.metas.pt_image[b], batch.labels.density_image[b], p.image_density_map[b], 
            batch.input_data.k_ext[b], batch.input_data.k_int[b], batch.aug_cfg.img_aug_aff[b]
        ):
            K_ext, K_int, img_aug_aff = K_ext.cpu().numpy(), K_int.cpu().numpy(), img_aug_aff.cpu().numpy()
            pt_bev = np.einsum('ij,nj->ni', K_ext[:3,:3], o_pt_bev) + K_ext[:3, 3:4].T
            pt_bev = np.einsum('ij,nj->ni', img_aug_aff@K_int, pt_bev)
            pt_bev = pt_bev[pt_bev[:, 2] > 1e-4]
            pt_bev = pt_bev[:, :2] / pt_bev[:, 2:3]
            image, mask = unnormalize_image(image.cpu().numpy()).copy(), mask.cpu().numpy()[...,None]
            image = (image * 0.8 + mask * 0.2).astype(np.uint8)
            image = cv2.resize(image, (image.shape[1]//ds, image.shape[0]//ds))
            cv2.rectangle(image, (0, 0), (image.shape[1]-1, image.shape[0]-1), (0,255,0), 5)
            image_bev = image.copy()
            image_per = image.copy()
            fn_draw(image_per, pt_per/ds)
            fn_draw(image_bev, pt_bev/ds)
            vis_per_pt.append(image_per)
            vis_bev_pt.append(image_bev)
            g_img_density_maps.append(normalize_density_map(g_density_map.cpu().numpy()))
            p_img_density_maps.append(normalize_density_map(p_density_map.cpu().numpy()[0]))
        batch_vis_per_pt.append(np.concatenate(vis_per_pt, 1))
        batch_vis_bev_pt.append(np.concatenate(vis_bev_pt, 1))
        g_batch_img_density_maps.append(np.concatenate(g_img_density_maps, 1))
        p_batch_img_density_maps.append(np.concatenate(p_img_density_maps, 1))

    vis_img_pt             = np.concatenate([np.concatenate(batch_vis_per_pt, 1), np.concatenate(batch_vis_bev_pt, 1)], 0)
    batch_img_density_maps = np.concatenate([np.concatenate(g_batch_img_density_maps, 1), np.concatenate(p_batch_img_density_maps, 1)], 0)
    batch_bev_density_maps = np.concatenate([np.concatenate(g_batch_bev_density_maps, 1), np.concatenate(p_batch_bev_density_maps, 1)], 0)
    logger.add_images(prefix+'/vis_img_pt',      vis_img_pt,             global_step=step, dataformats='HWC')
    logger.add_images(prefix+'/img_density_map', batch_img_density_maps, global_step=step, dataformats='HWN')
    logger.add_images(prefix+'/bev_density_map', batch_bev_density_maps, global_step=step, dataformats='HWN')