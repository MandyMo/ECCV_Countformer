
'''
    file:   CityStreet.py
    author: zhangxiong (1025679612@qq.com)
    date:   2023/06/23
'''

from .register import DATASET
from .BaseDataset import BaseDataset
from os import path as osp
import numpy as np
import h5py
import cv2

@DATASET.register_module()
class CityStreet(BaseDataset):
    def __init__(self, root, mode, trans, ncam=3, ori_h=1520, ori_w=2704, unchanged=False, use_refined_lable=True):
        self.use_refined_lable=use_refined_lable
        super(CityStreet, self).__init__(root=root, mode=mode, trans=trans, ncam=ncam, ori_h=ori_h, ori_w=ori_w, unchanged=unchanged)
    
    def load_dataset(self):
        from .BaseDataset import collect_head_point
        if self.mode=='train':
            start, end = 636, 1236
        else:
            start, end = 1236, 1636
        
        '''
            load image set
        '''
        self.image_lists = []
        for image_id in range(start, end, 2):
            image_name = 'frame_'+str(image_id).zfill(4)+'.jpg'
            self.image_lists.append([
                osp.join(self.root, 'image_frames', 'camera1', image_name),
                osp.join(self.root, 'image_frames', 'camera3', image_name),
                osp.join(self.root, 'image_frames', 'camera4', image_name),
            ])

        '''
            load head points in image space
        '''
        import json
        self.pt_images   = []
        view1_points = json.load(open(osp.join(self.root, 'labels/via_region_data_view1.json'), 'r'))
        view2_points = json.load(open(osp.join(self.root, 'labels/via_region_data_view2.json'), 'r'))
        view3_points = json.load(open(osp.join(self.root, 'labels/via_region_data_view3.json'), 'r'))
        for image_id in range(start, end, 2):
            image_name = 'frame_'+str(image_id).zfill(4)+'.jpg'
            self.pt_images.append([
                collect_head_point(view1_points[image_name], space='image'),
                collect_head_point(view2_points[image_name], space='image'),
                collect_head_point(view3_points[image_name], space='image'),
            ])
        
        '''
            load head points in bev space
        '''
        self.pt_bev      = []
        if not self.use_refined_lable: #use original labels
            gp_pmap = np.array(h5py.File(osp.join(self.root, 'labels', 'Street_groundplane_pmap.h5'), 'r')['v_pmap_GP'])
            for image_id in range(start, end, 2):
                bev_point        = gp_pmap[gp_pmap[:, 0]==image_id][:, 2:]
                bev_point[:, :2] = (bev_point[:, :2] - np.array([352*0.8, 522*0.8]).reshape(-1, 2)) * 76.25 # unnormalize
                bev_point[:, -1] = -bev_point[:, -1] #neg the height of the CityStreet dataset
                self.pt_bev.append(bev_point)
        else:
            for image_id in range(start, end, 2):
                label = json.load(open(osp.join(self.root, 'labels', 'bev_labels', 'frame_'+str(image_id).zfill(4)+'.json'), 'r'))
                self.pt_bev.append(np.array([[_pts[0][0], _pts[0][1], -_pts[0][2]] for (_id, _pts) in label.items()]))

        '''
            load extrinsic from world to camera
        '''
        self.k_w2cs = np.array([
            [
                [-3.61535900e-02,  9.95963230e-01, -8.21594000e-02,  1.27594012e+03],
                [-3.42514740e-01,  6.48842700e-02,  9.37269270e-01,  2.26524738e+02],
                [ 9.38816580e-01,  6.20264600e-02,  3.38786280e-01,  2.42302403e+04],
            ],
            [
                [-9.97306880e-01,  6.32474700e-02,  3.71314700e-02,  9.68047674e+03],
                [ 2.43989700e-02, -1.91327680e-01,  9.81222900e-01,  2.27339720e+03],
                [ 6.91641500e-02,  9.79486330e-01,  1.89269230e-01,  3.14367631e+04],
            ],
            [
                [ 9.95455020e-01,  7.57878200e-02, -5.76672600e-02,  3.30955016e+03],
                [ 1.67508000e-02,  4.56756280e-01,  8.89434150e-01, -4.60390351e+03],
                [ 9.37481600e-02, -8.86357660e-01,  4.53410830e-01,  2.92641938e+04],
            ]
        ])

        '''
            load intrinsics of each camera
        '''
        self.k_ints = np.array([
            [
                [1.24340000e+03, 0.00000000e+00, 1.38455000e+03],
                [0.00000000e+00, 1.24890000e+03, 7.47846499e+02],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            ],
            [
                [1.25818922e+03, 0.00000000e+00, 1.29253760e+03],
                [0.00000000e+00, 1.25988567e+03, 7.70518094e+02],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
            ],
            [
                [1.255900e+03, 0.000000e+00, 1.374100e+03],
                [0.000000e+00, 1.257200e+03, 7.857148e+02],
                [0.000000e+00, 0.000000e+00, 1.000000e+00],
            ],
        ])

        '''
            distortion coefficients
        '''
        if self.unchanged:
            self.k_distCoeffs = np.array([
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ])
        else:
            self.k_distCoeffs = np.array([
                [    -0.2090,     0.0382, 0, 0, 0],
                [-0.23977946, 0.05901853, 0, 0, 0],
                [    -0.2489, 0.07147814, 0, 0, 0],
            ])

        '''
            load mask 
        '''
        self.k_masks = [
            cv2.imread(osp.join(self.root, 'ROI_maps/ROIs/camera_view/mask1.jpg'))[...,0],
            cv2.imread(osp.join(self.root, 'ROI_maps/ROIs/camera_view/mask2.jpg'))[...,0],
            cv2.imread(osp.join(self.root, 'ROI_maps/ROIs/camera_view/mask3.jpg'))[...,0],
        ]