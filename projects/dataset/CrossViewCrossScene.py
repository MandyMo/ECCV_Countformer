
'''
    file:   CrossViewCrossScene.py
    author: zhangxiong (1025679612@qq.com)
    date:   2023/08/07
'''
from .register import DATASET

import torch
from os import path as osp
import numpy as np
import glob
import cv2
import random
from easydict import EasyDict as edict
import json
import copy
import numbers

@DATASET.register_module()
class CrossViewCrossScene(torch.utils.data.Dataset):
    def __init__(self, root, mode, trans, ncam=20, ori_h=1080, ori_w=1920, rescale=0.5, extend=1.4, prun_unvisible_pt_bev=True, bev_trans_scale=0.1, static_range=None):
        self.root, self.mode, self.trans, self.ncam, self.ori_h, self.ori_w, self.rescale, self.extend = root, mode, trans, ncam, ori_h, ori_w, rescale, extend
        self.prun_unvisible_pt_bev, self.bev_trans_scale, self.static_range = prun_unvisible_pt_bev, bev_trans_scale, static_range

        self.load_dataset()
    
    def __len__(self, *args, **kwargs):
        return len(self.samples)

    def __getitem__(self, index, *args, **kwargs):
        def get_cam_rigs(views, ncam):
            if not isinstance(ncam, numbers.Number):
                ncam = random.randint(int(ncam[0]), int(ncam[1]))
            delta = len(views) / ncam
            if self.mode=='train':
                return np.array([random.uniform(delta*k,delta*k+delta) for k in range(ncam)]).astype(np.int32).clip(0, len(views)-1)
            else: #test mode, then fixed the cam-rigs
                return np.array([delta*(k+0.5) for k in range(ncam)]).astype(np.int32).clip(0, len(views)-1)

        def _get_extrin_(rvec, tvec):
            rot_mat = cv2.Rodrigues(np.array(rvec))[0]
            return np.concatenate([rot_mat, np.array(tvec).reshape(3, 1)], 1)

        def load_label(lbl_path):
            label = edict(json.load(open(lbl_path, 'r')))
            label = edict(
                pts_2d={str(pt.idx):[pt.pixel[1], pt.pixel[0]] for pt in label.image_info if pt.pixel is not None},
                pts_3d={str(pt.idx):pt.world for pt in label.image_info if pt.world is not None},
                dist=np.array(label.distCoeffs, dtype=np.float32),
                K_e = _get_extrin_(rvec=label.rvec, tvec=label.tvec).astype(np.float32),
                K_i = np.array(label.cameraMatrix, dtype=np.float32),
            )
    
            sx, sy = 1920*self.rescale, 1080*self.rescale

            for key in label.pts_2d.keys():
                label.pts_2d[key][0] *= self.rescale
                label.pts_2d[key][1] *= self.rescale

            label.K_i[0] = label.K_i[0] * 1920 * self.rescale
            label.K_i[1] = label.K_i[1] * 1080 * self.rescale

            return label

        def load_image(img_path):
            img = cv2.imread(img_path)
            h, w, _ = img.shape
            return cv2.resize(img, (int(w*self.rescale), int(h*self.rescale)),interpolation=cv2.INTER_LINEAR)

        def undistort(image, label):
            image = cv2.undistort(image, label.K_i, label.dist)
            keys = list(label.pts_2d.keys())
            pts_2d = np.array([label.pts_2d[key] for key in keys]).reshape(-1, 2).copy()
            if pts_2d.size > 0: #certain images contain no head-points
                pts_2d = cv2.undistortPoints(pts_2d.astype(np.float32), label.K_i, label.dist, None, label.K_i).reshape(-1, 2)
            for (key, pt_2d) in zip(keys, pts_2d):
                label.pts_2d[key] = pt_2d

            return image, label

        def get_world_range(label):
            pts_3d = np.array([pt_3d for (_id, pt_3d) in label.pts_3d.items()]).reshape(-1, 3)
            min_pt_bev, max_pt_bev = pts_3d.min(0), pts_3d.max(0)
            ctr_pt_bev = (min_pt_bev + max_pt_bev) / 2.0
            if self.static_range is None:
                bev_range = (max_pt_bev - min_pt_bev) / 2.0
                bev_range[0], bev_range[1], bev_range[2] = max(bev_range[0], bev_range[1]), max(bev_range[0], bev_range[1]), bev_range[2] * 1.2
            else:
                bev_range = np.array(self.static_range)
            min_pt_bev, max_pt_bev =  ctr_pt_bev - bev_range * self.extend, ctr_pt_bev + bev_range * self.extend
            return min_pt_bev, max_pt_bev

        meta_views = self.samples[index % len(self.samples)]
        cam_rigs = get_cam_rigs(views=meta_views, ncam=self.ncam)
        labels = [load_label(meta_views[cam_rig][1]) for cam_rig in cam_rigs]

        #filter abnormal samples
        for index in range(len(labels)):
            if abs(labels[index].dist[4]) >= 2.0: 
                resample_index = random.randint(0, len(meta_views)-1)
                cam_rigs[index] = resample_index
                labels[index] = load_label(meta_views[resample_index][1])

        #load images
        images = [load_image(meta_views[cam_rig][0]) for cam_rig in cam_rigs]

        #undistort
        for index in range(len(images)):
            images[index], labels[index] = undistort(images[index], labels[index])

        #gather all pts_3d
        if self.prun_unvisible_pt_bev:
            all_ids = [key for label in labels for key in label.pts_2d.keys()]
            unq_ids = np.unique(all_ids)
            pts_3d = np.array([labels[0].pts_3d[_id] for _id in unq_ids if _id in labels[0].pts_3d]).reshape(-1, 3)
        else:
            pts_3d = np.array([pt_3d for (_id, pt_3d) in labels[0].pts_3d.items()]).reshape(-1, 3)

        #gather all pts_2d
        pts_2d = [np.array([pt_2d for (_id, pt_2d) in label.pts_2d.items()]).reshape(-1, 2) for label in labels]


        #gather masks
        h, w, _ = images[0].shape
        masks = [np.ones((h, w), dtype=np.uint8)*255 for _ in range(len(images))]

        min_pt_bev, max_pt_bev = get_world_range(label=labels[0])
        trans_bev_range = (max_pt_bev - min_pt_bev) / self.extend * self.bev_trans_scale

        input_dict = edict(
            input_data = edict(
                image_set   = images,
                cam_rigs    = cam_rigs,
                image_masks = masks,
                k_ext       = [label.K_e for label in labels],
                k_int       = [label.K_i for label in labels],
                world_range = [min_pt_bev[0], min_pt_bev[1], min_pt_bev[2], max_pt_bev[0], max_pt_bev[1], max_pt_bev[2]], #pay attention
            ),
            aug_cfg=edict(
                img_aug_aff=[np.eye(3) for _ in range(len(images))],
                bev_aug_aff=np.eye(4),
                trans=[[-trans_bev_range[0], trans_bev_range[0]], [-trans_bev_range[1], trans_bev_range[1]], [-trans_bev_range[2], trans_bev_range[2]]],
            ),
            labels=edict(
                pt_bev = pts_3d,
                pt_image = pts_2d,
            ),
            metas=edict(
                image_paths = [meta_views[cam_rig][0] for cam_rig in cam_rigs],
            ),
        )

        return self.trans(copy.deepcopy(input_dict))
       

    def load_dataset(self):
        if self.mode=='train':
            parts = ['train']
        else:
            parts = ['val']

        self.samples = []

        for part in parts:
            for abs_scene in glob.glob(osp.join(self.root, 'images', part, '*')):
                scene = osp.basename(abs_scene)
                for abs_frame in glob.glob(osp.join(abs_scene, '*')):
                    frame = osp.basename(abs_frame)
                    meta_views = []
                    for img_path in glob.glob(osp.join(abs_frame, 'jpgs', '*.jpg')):
                        img_name = osp.basename(img_path)
                        lbl_path = osp.join(self.root, 'labels', '100frames_labels_reproduce_640_480_CVCS', part, scene, frame, 'json_paras', img_name.replace('.jpg', '.json'))
                        if osp.exists(lbl_path):
                            meta_views.append((img_path, lbl_path, part, scene, frame))
                    meta_views.sort(key=lambda x: int(osp.splitext(osp.basename(x[0]))[0])) #sort by img_id
                    self.samples.append(meta_views)
        self.samples.sort(key=lambda x: int(x[0][3].split('_')[1])*10000 + int(x[0][4])) #sort by scene and frame