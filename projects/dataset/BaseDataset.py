

'''
    file:   BaseDataset.py
    author: zhangxiong (1025679612@qq.com)
    date:   2023/06/23
'''

from .register import DATASET

from os import path as osp
import torch
import copy
import cv2
import numpy as np
from easydict import EasyDict as edict
import copy

def collect_head_point(lbl, space='image'):
    pts = []
    for (_id, pt) in lbl['regions'].items():
        pt = pt['shape_attributes']
        cx, cy = pt['cx'], pt['cy']
        if cx is None or cy is None:
            continue
        if space=='image':
            pts.append([cx, cy])
        else:
            pts.append([cx, cy, pt['height'] if 'height' in pt else 1600])
    pts = np.array(pts)    
    return pts

@DATASET.register_module()
class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode, trans, ncam, ori_h, ori_w, unchanged):
        super(BaseDataset, self).__init__()
        self.root, self.mode, self.trans, self.ncam, self.ori_h, self.ori_w, self.unchanged = root, mode, trans, ncam, ori_h, ori_w, unchanged
        assert self.mode in ['train', 'test', 'all']
        self.load_dataset()
        self.load_remap()
        
    def load_images(self, image_path):
        return [cv2.imread(path) for path in image_path]

    def undistort(self, image, mask, cameraMatrix, distCoeffs, points, remap): 
        map_x, map_y = remap[...,0].astype(np.float32), remap[...,1].astype(np.float32)
        undistorted_image  = cv2.remap(image, map_x, map_y, cv2.INTER_LANCZOS4)
        undistorted_mask   = cv2.remap(mask, map_x, map_y, cv2.INTER_LANCZOS4)
        undistorted_points = cv2.undistortPoints(points.astype(np.float32), cameraMatrix, distCoeffs, None, cameraMatrix)
        return undistorted_image, undistorted_mask, undistorted_points.reshape(-1, 2)

    def load_remap(self):
        remap = []
        for (cam_id, dist_coef) in enumerate(self.k_distCoeffs):
            K = self.k_ints[cam_id]
            if self.unchanged:
                map_x = np.tile(np.arange(0, self.ori_w).reshape(1, self.ori_w), (self.ori_h, 1))
                map_y = np.tile(np.arange(0, self.ori_h).reshape(self.ori_h, 1), (1, self.ori_w))
            else:
                map_x, map_y = cv2.initUndistortRectifyMap(K, dist_coef, None, K, (self.ori_w, self.ori_h), 5)
            remap.append(np.stack([map_x, map_y], -1))
        remap = np.array(remap).reshape(-1, self.ori_h, self.ori_w, 2)
        self.remap = remap

    @staticmethod
    def img2cam(u, v, K):
        return (u-K[0,2])/K[0,0], (v-K[1,2])/K[1,1]

    @staticmethod
    def cam2img(x, y, K):
        return x*K[0,0]+K[0,2], y*K[1,1]+K[1,2]

    @staticmethod
    def distort_pixel(u, v, K, k1, k2):
        x, y = BaseDataset.img2cam(u, v, K)
        x, y = BaseDataset.distort_coord(x, y, k1, k2)
        return BaseDataset.cam2img(x, y, K)

    @staticmethod
    def distort_coord(x, y, k1, k2):
        r = x*x + y*y
        r = 1 + k1*r + k2*r*r
        return x*r, y*r

    def __len__(self):
        return len(self.image_lists)
    
    def __getitem__(self, index):
        sel_ele = lambda eles, indices: [eles[index] for index in indices]
        index = index % len(self)
        cam_rigs = sorted(np.random.choice(range(len(self.image_lists[0])), size=self.ncam, replace=False))

        '''
            fetch the sample & label according to give cam-rigs
        '''
        image_paths = sel_ele(self.image_lists[index], cam_rigs)
        pt_images   = sel_ele(self.pt_images[index],   cam_rigs)
        k_w2cs      = sel_ele(self.k_w2cs,             cam_rigs)
        k_ints      = sel_ele(self.k_ints,             cam_rigs)
        k_masks     = sel_ele(self.k_masks,            cam_rigs)
        distCoeffs  = sel_ele(self.k_distCoeffs,       cam_rigs)
        pt_bev      = self.pt_bev[index]


        k_mean_int  = copy.deepcopy(k_ints)

        image_lists = self.load_images(image_paths)
        undistorted_images, undistorted_points, undistorted_masks = [], [], []

        for (index, (image, cameraMatrix, distCoeff, points, mask)) in enumerate(zip(image_lists, k_ints, distCoeffs, pt_images, k_masks)):
            image, mask, points = self.undistort(image, mask, cameraMatrix, distCoeff, points, remap=self.remap[cam_rigs[index]])
            undistorted_images.append(image)
            undistorted_points.append(points)
            undistorted_masks.append(mask)
        
        input_dict = edict(
            input_data = edict(
                image_set   = undistorted_images,
                cam_rigs    = cam_rigs,
                image_masks = undistorted_masks,
                k_ext       = k_w2cs,
                k_int       = k_mean_int,
            ),
            aug_cfg=edict(
                img_aug_aff=[np.eye(3),np.eye(3),np.eye(3)],
                bev_aug_aff=np.eye(4),
            ),
            labels=edict(
                pt_bev = pt_bev,
                pt_image = undistorted_points,
            ),
            metas=edict(
                image_paths = image_paths,
            ),
        )

        return self.trans(copy.deepcopy(input_dict))