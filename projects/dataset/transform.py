
'''
    file:   transform.py
    author: zhangxiong (1025679612@qq.com)
    date:   2023/06/23
'''


from ..modules import HELPERS
from .register import PIPELINE

import scipy
import numpy as np
import random
import math
import cv2
import numbers
from easydict import EasyDict as edict
import torch
import itertools
import copy

'''
    scale -> crop ->rotate -> flip, the points resides on the right of the operators.
    scale: a value, or [lower_bound, upper_bound], or [[lower_bound, upper_bound], [lower_bound, upper_bound]]
    angle: a value, or [lower_bound, upper_bound]
    center: [center_x, center_y], or [[lower_bound, upper_bound], [lower_bound, upper_bound]]
    crop: [x, y, w, h] or [[lower_bound, upper_bound], [lower_bound, upper_bound], w, h]
    p_hf: a probability
    p_vf: a probability
    center_coord: pixel or ratio
    crop_coord: pixel or ratio
'''
@PIPELINE.register_module()
class ImageAffineTransform(object):
    def __init__(self, scale, angle, center, crop, p_hf=0.5, p_vf=0.0, iso_aff=True, center_coord='pixel', crop_coord='pixel', *args, **kwargs):
        support_coord_types = ['pixel', 'ratio']
        assert center_coord in support_coord_types, f'center_coord must in {support_coord_types}'
        assert crop_coord in support_coord_types, f'crop_coord must in {support_coord_types}'
        self.scale, self.angle, self.center, self.crop, self.iso_aff, self.p_hf, self.p_vf, self.center_coord, self.crop_coord = scale, angle, center, crop, iso_aff, p_hf, p_vf, center_coord, crop_coord
        self.inter_flg = kwargs.get('inter_flg', [cv2.INTER_LINEAR, cv2.INTER_NEAREST, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])

    def _gen_aug_cfgs_(self, *args, **kwargs):
        def _get_crop_info_(cp_info, crop_length, max_length):
            if cp_info == 'center':
                start = (max_length-crop_length)//2
            elif cp_info in ['left', 'top']:
                start = 0
            elif cp_info in ['right', 'bottom']:
                start = max_length - crop_length
            elif cp_info == 'random':
                start = HELPERS.get('range_float_value')([0, max_length-crop_length])
            else:
                start = HELPERS.get('range_float_value')(cp_info)
                if self.crop_coord != 'pixel':
                    start = start * max_length
            return start

        h, w = kwargs.get('h'), kwargs.get('w')
        if not isinstance(self.scale, numbers.Number): #a range
            if isinstance(self.scale[0], numbers.Number) and isinstance(self.scale[1], numbers.Number): # iso scale
                scale = HELPERS.get('range_float_value')(self.scale)
                scale_x, scale_y = scale, scale
            else: # anti-iso scale
                scale_x, scale_y = HELPERS.get('range_float_value')(self.scale[0]), HELPERS.get('range_float_value')(self.scale[1])
        else:# a fixed scale
            scale_x, scale_y = self.scale, self.scale
        h, w  = h * scale_y, w * scale_x

        '''
            set the crop range
        '''
        if self.crop_coord != 'pixel':
            croph, cropw = int(self.crop[3] * h), int(self.crop[2] * w)
        else:
            croph, cropw = int(self.crop[3]), int(self.crop[2])

        cropx = _get_crop_info_(cp_info=self.crop[0], crop_length=cropw, max_length=w)
        cropy = _get_crop_info_(cp_info=self.crop[1], crop_length=croph, max_length=h)

        h, w = croph, cropw

        '''
            set the rotation
        '''
        angle = HELPERS.get('range_float_value')(self.angle)
        ctx   = HELPERS.get('range_float_value')(self.center[0])
        cty   = HELPERS.get('range_float_value')(self.center[1])
        if self.center_coord != 'pixel':
            ctx, cty = ctx * w, cty * h

        b_hf  = random.random() < self.p_hf #left-right flip or not.
        b_vf  = random.random() < self.p_vf #top-bottom flip or not.

        return edict(
            scale=[scale_x,scale_y], angle=angle, center=[ctx, cty],crop=[cropx, cropy, cropw, croph], b_hf=b_hf, b_vf=b_vf,
        )

    def __call__(self, R, *args, **kwargs):
        cam_rigs  = range(len(R.input_data.image_set))
        aug_cfg   = None
        inter_flg = kwargs.get('inter_flg', self.inter_flg)
        R.metas.img_shape = []
        for cam_rig in cam_rigs:
            h, w, _ = R.input_data.image_set[cam_rig].shape
            aff_mat = R.aug_cfg.img_aug_aff[cam_rig]
            if not self.iso_aff or aug_cfg is None: #aug each cam rig with different augmentation parameters
                aug_cfg = self._gen_aug_cfgs_(*args, h=h, w=w, **kwargs)
            aug_aff_mat = HELPERS.get('get_affine_matrix')(scale=aug_cfg.scale, angle=aug_cfg.angle, center=aug_cfg.center, crop=aug_cfg.crop)
            if aug_cfg.get('b_hf', False):
                hf_mat = HELPERS.get('get_horizontal_flip_matrix_2d')(w=aug_cfg.crop[2])
                aug_aff_mat = hf_mat @ aug_aff_mat
            if aug_cfg.get('b_vf', False):
                vf_mat = HELPERS.get('get_vertical_flip_matrix_2d')(h=aug_cfg.crop[3])
                aug_aff_mat = vf_mat @ aug_aff_mat
            R.input_data.image_set[cam_rig]  = cv2.warpAffine(R.input_data.image_set[cam_rig],  aug_aff_mat[:2], (aug_cfg.crop[2], aug_cfg.crop[3]), flags=np.random.choice(inter_flg, 1)[0])
            R.input_data.image_masks[cam_rig] = cv2.warpAffine(R.input_data.image_masks[cam_rig], aug_aff_mat[:2], (aug_cfg.crop[2], aug_cfg.crop[3]), flags=np.random.choice(inter_flg, 1)[0])
            if R.labels.pt_image[cam_rig].size > 0: #pay attention
                R.labels.pt_image[cam_rig] = np.einsum('ij,nj->ni', aug_aff_mat[:2, :2], R.labels.pt_image[cam_rig]) + aug_aff_mat[:2,2:3].T
            R.aug_cfg.img_aug_aff[cam_rig] = aug_aff_mat @ aff_mat
            R.metas.img_shape.append([aug_cfg.crop[2], aug_cfg.crop[3]])
        return R

    def __repr__(self):
        format_str = self.__class__.__name__ + \
            f'(scale={self.scale}, ' + \
            f'angle={self.angle}, ' + \
            f'center={self.center},' + \
            f'crop={self.crop}, ' + \
            f'iso_aff={self.iso_aff}, ' + \
            f'p_hf={self.p_hf}, ' + \
            f'p_vf={self.p_vf}, ' + \
            f'center_coord={self.center_coord}, ' + \
            f'crop_coord={self.crop_coord})'
        return format_str

@PIPELINE.register_module()
class FilterOutlier(object):
    def __init__(self, world_range=None, employ_mask=True, filter_z=False, eps=0.0, *args, **kwargs):
        self.world_range, self.employ_mask, self.filter_z, self.eps = world_range, employ_mask, filter_z, eps
    
    def _filter_point_image_(self, R):
        h, w, _ = R.input_data.image_set[0].shape
        if self.employ_mask:
            R.labels.pt_image = [
                np.array([pt for pt in pt_image if pt[0]>=0 and pt[1]>=0 and pt[0]<w and pt[1]<h and mask[int(pt[1]), int(pt[0])]>=127.5]) for (pt_image, mask) in zip(R.labels.pt_image, R.input_data.image_masks)
            ]
        else:
            R.labels.pt_image = [
                np.array([pt for pt in pt_image if pt[0]>=0 and pt[1]>=0 and pt[0]<w and pt[1]<h ]) for pt_image in R.labels.pt_image
            ]
        return R

    def _filter_point_bev_(self, R):
        world_range = R.input_data.get('world_range', self.world_range)
        if world_range is None:
            return
        pts_bev, eps = R.labels.pt_bev[:, :3], self.eps
        if not self.filter_z:
            R.labels.pt_bev = np.array([
                pt for pt in pts_bev \
                    if  pt[0]>= world_range[0] + eps \
                    and pt[1]>= world_range[1] + eps \
                    and pt[0] < world_range[3] - eps \
                    and pt[1] < world_range[4] - eps
            ]).reshape(-1, 3)
        else:
            R.labels.pt_bev = np.array([
                pt for pt in pts_bev \
                    if  pt[0]>= world_range[0] + eps \
                    and pt[1]>= world_range[1] + eps \
                    and pt[2]>= world_range[2] + eps \
                    and pt[0] < world_range[3] - eps \
                    and pt[1] < world_range[4] - eps \
                    and pt[2] < world_range[5] - eps
            ]).reshape(-1, 3)

    def __call__(self, R, *args, **kwargs):
        self._filter_point_image_(R) #image space
        self._filter_point_bev_(R)   #bev space

        return R

    def __repr__(self):
        format_str = self.__class__.__name__ + \
            f'(world_range={self.world_range}, ' + \
            f'employ_mask={self.employ_mask})'
        return format_str

@PIPELINE.register_module()
class GammaAdj(object):
    def __init__(self, scale=[0.7,1.3], iso_color=True, iso_cam_rigs=True, *args, **kwargs):
        self.scale, self.iso_color, self.iso_cam_rigs = scale, iso_color, iso_cam_rigs

    def __call__(self, R, *args, **kwargs):
        def _gamma_adj_impl_(img, gamma):
            return ((img.astype(np.float32) / 255.0)**gamma)*255.0
        
        cam_rigs   = range(len(R.input_data.image_set))
        fb, fg, fr = HELPERS.get('range_float_value')(self.scale), HELPERS.get('range_float_value')(self.scale), HELPERS.get('range_float_value')(self.scale)
        for cam_rig in cam_rigs:
            if self.iso_cam_rigs and self.iso_color:
                f = np.array([fb, fb, fb]).reshape(1, 1, 3)
            elif self.iso_cam_rigs and not self.iso_color:
                f = np.array([fb, fg, fr]).reshape(1, 1, 3)
            elif not self.iso_cam_rigs and self.iso_color:
                f = HELPERS.get('range_float_value')(self.scale)
            else:
                f = np.array([HELPERS.get('range_float_value')(self.scale), HELPERS.get('range_float_value')(self.scale), HELPERS.get('range_float_value')(self.scale)]).reshape(1, 1, 3)
            R.input_data.image_set[cam_rig] = _gamma_adj_impl_(R.input_data.image_set[cam_rig], f)
        return R

    def __repr__(self):
        format_str = self.__class__.__name__ + \
            f'(scale={self.scale}, ' + \
            f'iso_color={self.iso_color}, ' + \
            f'iso_cam_rigs={self.iso_cam_rigs})'
        return format_str

@PIPELINE.register_module()
class Gray(object):
    def __init__(self, iso=True, b=[0.1, 0.2], g=[0.55,0.65], r=[0.25,0.35], *args, **kwargs):
        self.b, self.g, self.r, self.iso = b, g, r, iso
    
    def _gen_parameter_(self, mode='BGR'):
        b,g,r = HELPERS.get('range_float_value')(self.b), HELPERS.get('range_float_value')(self.g), HELPERS.get('range_float_value')(self.r)
        if mode in ['BGR', 'bgr']:
            coefs = np.array([b,g,r])
        elif mode in ['RGB', 'rgb']:
            coefs = np.array([r,g,b])
        else:
            raise Exception('unsupported color mode.')

        coefs = coefs/sum(coefs) #normalize the coefficient
        return np.tile(coefs[None], (3, 1))
    
    def __call__(self, R, *args, **kwargs):
        color_mode='BGR'
        cam_rigs  = range(len(R.input_data.image_set))
        coefs = self._gen_parameter_(mode=color_mode)
        for cam_rig in cam_rigs:
            if not self.iso:
                coefs = self._gen_parameter_(mode=color_mode)
            img_gray = np.einsum('hwi,ci->hwc', R.input_data.image_set[cam_rig], coefs)
            R.input_data.image_set[cam_rig] = img_gray
        return R

    def __repr__(self):
        format_str = self.__class__.__name__ + \
            f'(self.b={self.b}, ' + \
            f'self.g={self.g}, ' + \
            f'self.r={self.r}, ' + \
            f'self.iso={self.iso})'
        return format_str

@PIPELINE.register_module()
class ChannelShuffle(object):
    def __init__(self, iso_cam_rigs=True, *args, **kwargs):
        self.iso_cam_rigs = iso_cam_rigs
    
    def __call__(self, R, *args, **kwargs):
        cam_rigs  = range(len(R.input_data.image_set))
        b,g,r=np.random.choice([0,1,2],3,replace=False)
        for cam_rig in cam_rigs:
            if not self.iso_cam_rigs:
                b,g,r=np.random.choice([0,1,2],3,replace=False)
            R.input_data.image_set[cam_rig] = R.input_data.image_set[cam_rig][...,[b,g,r]].copy()
        return R

    def __repr__(self):
        format_str = self.__class__.__name__ + \
            f'(iso_cam_rigs={self.iso_cam_rigs})'
        return format_str

@PIPELINE.register_module()
class BEVAffineTransform(object):
    def __init__(self, scale, angle, trans=None, p_xf=0.5, p_yf=0.5, p_zf=0.0, world_range=None, *args, **kwargs):
        self.scale, self.angle, self.p_xf, self.p_yf, self.p_zf, self.trans, self.world_range = scale, angle, p_xf, p_yf, p_zf, trans, world_range

    def __gen_rot_mat__(self):
        if self.angle is None:
            return np.eye(4)
        if isinstance(self.angle[0], numbers.Number):
            r_z = math.radians(HELPERS.get('range_float_value')(self.angle))
            return HELPERS.get('get_rot_matrix_z')(theta=r_z, is_angle=False)
        r_x = math.radians(HELPERS.get('range_float_value')(self.angle[0])) if self.angle[0] is not None else 0.0
        r_y = math.radians(HELPERS.get('range_float_value')(self.angle[1])) if self.angle[1] is not None else 0.0
        r_z = math.radians(HELPERS.get('range_float_value')(self.angle[2])) if self.angle[2] is not None else 0.0
        
        return HELPERS.get('get_rot_matrix_x')(theta=r_x, is_angle=False) @ \
            HELPERS.get('get_rot_matrix_y')(theta=r_y, is_angle=False) @ \
            HELPERS.get('get_rot_matrix_z')(theta=r_z, is_angle=False)


    def __call__(self, R):
        #generate augmentation parameters
        b_xf  = random.random() < self.p_xf
        b_yf  = random.random() < self.p_yf
        b_zf  = random.random() < self.p_zf
        scale = HELPERS.get('range_float_value')(self.scale)

        world_range = R.input_data.get('world_range', self.world_range)
        #shift&unshift the center matrix
        if world_range is not None:
            ct_x = (world_range[0]+world_range[3])/2.0
            ct_y = (world_range[1]+world_range[4])/2.0
            ct_z = (world_range[2]+world_range[5])/2.0
            aff_shift_cener, aff_recover_center = HELPERS.get('get_trans_matrix_3d')(-ct_x, -ct_y, -ct_z), HELPERS.get('get_trans_matrix_3d')(ct_x, ct_y, ct_z)
        else:
            aff_shift_cener, aff_recover_center = np.eye(4), np.eye(4)

        #do shift the center
        aff_bev_aug_mat       = aff_shift_cener @ np.eye(4)
        aff_bev_aug_rot_mat   = self.__gen_rot_mat__()
        aff_bev_aug_mat       = aff_bev_aug_rot_mat @ aff_bev_aug_mat

        aff_bev_aug_scale_mat = HELPERS.get('get_scale_matrix_3d')(scale, scale, scale)
        aff_bev_aug_mat       = aff_bev_aug_scale_mat @ aff_bev_aug_mat
        
        aff_bev_aug_xflip_mat = HELPERS.get('get_scale_matrix_3d')(-1.0 if b_xf else 1.0, 1.0, 1.0)
        aff_bev_aug_mat       = aff_bev_aug_xflip_mat @ aff_bev_aug_mat

        aff_bev_aug_yflip_mat = HELPERS.get('get_scale_matrix_3d')(1.0, -1.0 if b_yf else 1.0, 1.0)
        aff_bev_aug_mat       = aff_bev_aug_yflip_mat @ aff_bev_aug_mat

        aff_bev_aug_zflip_mat = HELPERS.get('get_scale_matrix_3d')(1.0, 1.0, -1.0 if b_zf else 1.0)
        aff_bev_aug_mat       = aff_bev_aug_zflip_mat @ aff_bev_aug_mat

        '''
            translation
        '''
        trans = R.aug_cfg.get('trans', self.trans)
        if trans is not None:
            tx = HELPERS.get('range_float_value')(trans[0])
            ty = HELPERS.get('range_float_value')(trans[1])
            tz = HELPERS.get('range_float_value')(trans[2])
            aff_bev_aug_mat   = HELPERS.get('get_trans_matrix_3d')(tx, ty, tz) @ aff_bev_aug_mat

        #recover the center
        aff_bev_aug_mat = aff_recover_center @ aff_bev_aug_mat
        
        '''
            update the bev points
        '''
        pt_bev = R.labels.pt_bev[:, :3] # N x 3
        pt_bev = np.concatenate([pt_bev, np.ones((pt_bev.shape[0], 1))], -1) # N x 4
        R.labels.pt_bev[...,:3] = np.einsum('ij,nj->ni', aff_bev_aug_mat, pt_bev)[..., :3]

        R.aug_cfg.bev_aug_aff    = aff_bev_aug_mat @ R.aug_cfg.bev_aug_aff

        return R

    def __repr__(self):
        format_str = self.__class__.__name__ + \
            f'(self.scale={self.scale}, ' + \
            f'self.angle={self.angle}, ' + \
            f'self.p_xf={self.p_xf}, ' + \
            f'self.world_range={self.world_range}, ' + \
            f'self.p_yf={self.p_yf})'
        return format_str

@PIPELINE.register_module()
class Compose(object):
    def __init__(self, ops=[], *args, **kwargs):
        self.ops = ops
    
    def __call__(self, R, *args, **kwargs):
        for op in self.ops:
            R = op(R)
        return R
    
    def append(self, op):
        self.ops.append(op)

    def __getitem__(self, index):
        assert index >= 0 and index < len(self), f'please make sulre that 0 <= index <= {len(self)}'
        return self.ops(index)

    def __len__(self):
        return len(self.ops)
    
    def __repr__(self):
        format_str = self.__class__.__name__ + '=['
        for op in self.ops:
            format_str += f'\n    {op},'
        format_str += '\n]'
        return format_str

@PIPELINE.register_module()
class ProbOp(object):
    def __init__(self, op, p=0.5, *args, **kwargs):
        self.op, self.p = op, p
    
    def __call__(self, R, *args, **kwargs):
        return self.op(R) if random.random() < self.p else R

    def __repr__(self):
        format_str = self.__class__.__name__ + \
            f'(op={self.op}, ' + \
            f'prob={self.p})'
        return format_str

'''
    random select k camera views to conduct the data augmentation
'''
@PIPELINE.register_module()
class RandomCamRigOp(object):
    def __init__(self, op, n_cam_rig, *args, **kwargs):
        self.op, self.n_cam_rig = op, n_cam_rig
    
    def __call__(self, R, *args, **kwargs):
        cam_rigs  = range(len(R.input_data.image_set))
        n_rigs   = min(max(int(HELPERS.get('range_float_value')(self.n_cam_rig)), 0), len(cam_rigs))
        kwargs['cam_rigs'] = np.random.choice(cam_rigs, n_rigs, replace=False)
        return self.op(R, *args, **kwargs)

    def __repr__(self):
        format_str = self.__class__.__name__ + \
            f'(op={self.op}, ' + \
            f'n_cam_rig={self.n_cam_rig})'
        return format_str

@PIPELINE.register_module()
class BaseDensityMapGenerator(object):
    def __init__(self, normalize, *args, **kwargs):
        self.normalize = normalize

    def __call__(self, *args, **kwargs):
        raise "You shall call a concrete subclass of BaseDensityMapGenerator"
    
@PIPELINE.register_module()
class KDTreeDensetyMapGenerator(BaseDensityMapGenerator):
    def __init__(self, leafsize=2048, k=4, normalize=True, scale=1.0):
        super(KDTreeDensetyMapGenerator, self).__init__(normalize=normalize)
        self.leafsize, self.k, self.scale = leafsize, k, scale

    def __call__(self, pts, size):
        pts, gt_count = pts.astype(np.int32), pts.shape[0]
        density = np.zeros((size[1], size[0]), dtype=np.float32)
        if gt_count <= 0:
            return density
        
        tree = scipy.spatial.KDTree(pts.copy(), leafsize=self.leafsize)
        distances, _ = tree.query(pts, k=self.k)
        distances[np.isinf(distances)]=0
        
        for i, pt in enumerate(pts):
            pt2d = np.zeros((size[1], size[0]), dtype=np.float32)
            pt2d[pt[1],pt[0]] = 1.
            if gt_count > 1:
                sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
            else:
                sigma = np.average(size)/4.0 #case: 1 point
            density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
        if self.normalize:
            density = density * (gt_count / density.sum())
        return density * self.scale

@PIPELINE.register_module()
class GaussianFilterDensityMapGenerator(BaseDensityMapGenerator):
    def __init__(self, sigma=10, normalize=True, scale=1.0, *args, **kwargs):
        super(GaussianFilterDensityMapGenerator, self).__init__(normalize=normalize)
        self.sigma, self.scale = sigma, scale
    
    def __call__(self, pts, size):
        pts, gt_count = pts.astype(np.int32), pts.shape[0]
        density = np.zeros(size[::-1], dtype=np.float32) # xyz -> zyx | xy -> yx
        if gt_count <= 0:
            return density
    
        for pt in pts:
            if len(pt)==2: #bev density or 2d density
                density[pt[1], pt[0]] += 1.0
            else: #3d density
                density[pt[2], pt[1], pt[0]] += 1.0
        density = scipy.ndimage.filters.gaussian_filter(density, self.sigma, mode='constant')
        if self.normalize:
            density = density * (gt_count / density.sum())
        return density * self.scale
        # for pt in pts:
        #     d_pt = np.zeros(size[::-1], dtype=np.float32) # xyz -> zyx | xy -> yx
        #     if len(pt)==2: #bev density or 2d density
        #         d_pt[pt[1], pt[0]] = 1.0
        #     else: #3d density
        #         d_pt[pt[2], pt[1], pt[0]] = 1.0
        #     d_pt = scipy.ndimage.filters.gaussian_filter(d_pt, self.sigma, mode='constant')
        #     if self.normalize:
        #         d_pt = d_pt * (1.0 / d_pt.sum())
        #     density = density + d_pt
        # return density * self.scale

@PIPELINE.register_module()
class GenDensity(object):
    def __init__(self, world_range=None, bev_hm_h=None, bev_hm_w=None, down_scale=4, gen_image_density_map=False, gen_bev_density_map=False, img_density_generator=None, bev_density_generator=None, *args, **kwargs):
        self.world_range, self.bev_hm_h, self.bev_hm_w, self.down_scale = world_range, bev_hm_h, bev_hm_w, down_scale
        self.gen_image_density_map, self.gen_bev_density_map      = gen_image_density_map, gen_bev_density_map
        self.img_density_generator, self.bev_density_generator    = img_density_generator, bev_density_generator

    def _gen_image_density_map_(self, R, *args, **kwargs):
        assert self.img_density_generator is not None
        R.labels.density_image = []
        for (pts, image) in zip(R.labels.pt_image, R.input_data.image_set):
            h, w, _ = image.shape
            R.labels.density_image.append(self.img_density_generator(pts/self.down_scale, (w//self.down_scale, h//self.down_scale)))

    def _gen_bev_density_map_(self, R, *args, **kwargs):
        world_range = R.input_data.get('world_range', self.world_range)
        assert self.bev_density_generator is not None
        res_x, res_y = (world_range[3]-world_range[0]) / self.bev_hm_w, (world_range[4]-world_range[1]) / self.bev_hm_h
        pts_bev = ((R.labels.pt_bev[...,:2] - np.array([world_range[0], world_range[1]]).reshape(-1, 2)) / np.array([res_x, res_y]).reshape(-1, 2)).astype(np.int32)
        R.labels.density_bev = self.bev_density_generator(pts_bev, (self.bev_hm_w, self.bev_hm_h))

    def __call__(self, R, *args, **kwargs):
        if self.gen_image_density_map:
            self._gen_image_density_map_(R)
        if self.gen_bev_density_map:
            assert self.bev_hm_h is not None and self.bev_hm_w is not None
            self._gen_bev_density_map_(R)
        return R
        
    def __repr__(self):
        format_str = self.__class__.__name__ + \
            f'(world_range={self.world_range}, ' + \
            f'bev_hm_h={self.bev_hm_h}, ' + \
            f'bev_hm_w={self.bev_hm_w}, ' + \
            f'down_scale={self.down_scale}, ' + \
            f'gen_image_density_map={self.gen_image_density_map}, ' + \
            f'gen_bev_density_map={self.gen_bev_density_map})'
            
        return format_str
    
@PIPELINE.register_module()
class GenDensity3D(object):
    def __init__(self, world_range=None, vox_hm_z=None, vox_hm_h=None, vox_hm_w=None, down_scale=4, gen_image_density_map=False, gen_vox_density_map=False, img_density_generator=None, vox_density_generator=None, *args, **kwargs):
        self.world_range, self.vox_hm_z, self.vox_hm_h, self.vox_hm_w, self.down_scale = world_range, vox_hm_z, vox_hm_h, vox_hm_w, down_scale
        self.gen_image_density_map, self.gen_vox_density_map      = gen_image_density_map, gen_vox_density_map
        self.img_density_generator, self.vox_density_generator    = img_density_generator, vox_density_generator

    def _gen_image_density_map_(self, R, *args, **kwargs):
        assert self.img_density_generator is not None
        R.labels.density_image = []
        for (pts, image) in zip(R.labels.pt_image, R.input_data.image_set):
            h, w, _ = image.shape
            R.labels.density_image.append(self.img_density_generator(pts/self.down_scale, (w//self.down_scale, h//self.down_scale)))

    def _gen_vox_density_map_(self, R, *args, **kwargs):
        world_range = R.input_data.get('world_range', self.world_range)
        assert self.vox_density_generator is not None
        res_x, res_y, res_z = (world_range[3]-world_range[0]) / self.vox_hm_w, (world_range[4]-world_range[1]) / self.vox_hm_h, (world_range[5]-world_range[2]) / self.vox_hm_z
        pts_bev = ((R.labels.pt_bev[...,:3] - np.array([world_range[0], world_range[1], world_range[2]]).reshape(-1, 3)) / np.array([res_x, res_y, res_z]).reshape(-1, 3)).astype(np.int32)
        R.labels.density_vox = self.vox_density_generator(pts_bev, (self.vox_hm_w, self.vox_hm_h, self.vox_hm_z))

    def __call__(self, R, *args, **kwargs):
        if self.gen_image_density_map:
            self._gen_image_density_map_(R)
        if self.gen_vox_density_map:
            assert self.vox_hm_h is not None and self.vox_hm_w is not None and self.vox_hm_z is not None
            self._gen_vox_density_map_(R)
        return R
        
    def __repr__(self):
        format_str = self.__class__.__name__ + \
            f'(world_range={self.world_range}, ' + \
            f'vox_hm_z={self.vox_hm_z}, ' + \
            f'vox_hm_h={self.vox_hm_h}, ' + \
            f'vox_hm_w={self.vox_hm_w}, ' + \
            f'down_scale={self.down_scale}, ' + \
            f'gen_image_density_map={self.gen_image_density_map}, ' + \
            f'gen_vox_density_map={self.gen_vox_density_map})'
            
        return format_str

@PIPELINE.register_module()
class Oneof(object):
    def __init__(self, ops, probs=None, *args, **kwargs):
        if probs is None:
            probs = np.ones(len(ops))
        probs = np.array(probs)/sum(probs)
        self.ops, self.probs = ops, probs
    
    def __call__(self, R, *args, **kwargs):
        op = np.random.choice(a=self.ops, size=1, replace=False, p=self.probs)[0]
        return op(R)

    def __repr__(self):
        format_str = self.__class__.__name__ + \
            f'(ops={self.ops}, ' + \
            f'probs={self.probs})'
        return format_str

@PIPELINE.register_module()
class BEVTTA(object):
    def __init__(self, image_scales, bev_rots, bev_scales, bev_xf, bev_yf, crop_info, density_generator, pt_filter):
        self.ttas = []
        for (s_img, s_bev, r_bev, x_f, y_f) in itertools.product(image_scales, bev_scales, bev_rots, bev_xf, bev_yf):
            self.ttas.append(
                Compose([
                    ImageAffineTransform(scale=s_img, angle=0, center=[0.5, 0.5], crop=crop_info, p_hf=0.0, p_vf=0.0, iso_aff=True, center_coord='ratio', crop_coord='pixel'),
                    BEVAffineTransform(scale=s_bev, angle=r_bev, p_xf=x_f, p_yf=y_f, trans=None),
                    copy.deepcopy(pt_filter),
                    copy.deepcopy(density_generator),
                ])
            )

    def __call__(self, R, *args, **kwargs):
        return [op(copy.deepcopy(R)) for op in self.ttas]

@PIPELINE.register_module()
class Load_Images(object):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, R, *args, **kwargs):
        R.input_data = edict(image_set=[cv2.imread(path) for path in R.image_path])
        return R

    def __repr__(self):
        format_str = self.__class__.__name__ 
        return format_str

@PIPELINE.register_module()
def tta_batch_collect_fn(batches):
    assert(len(batches)) == 1
    return batch_collect_fn(batches[0])

@PIPELINE.register_module()
def batch_collect_fn(batches):
    from einops import rearrange
    be = edict(
        input_data= edict(
            image_set   = torch.tensor(
                rearrange([normalize_image(image) for batch in batches for image in batch.input_data.image_set], '(n k) ... -> n k ...', n=len(batches)), dtype=torch.float32
            ),
            image_masks    = torch.tensor(
                np.stack([batch.input_data.image_masks for batch in batches], 0), dtype=torch.float32,
            ),
            cam_rigs = torch.tensor(
                np.stack([batch.input_data.cam_rigs for batch in batches], 0), dtype=torch.int
            ),
            k_int = torch.tensor(
                np.stack([batch.input_data.k_int for batch in batches], 0), dtype=torch.float32
            ),
            k_ext  = torch.tensor(
                np.stack([batch.input_data.k_ext for batch in batches], 0), dtype=torch.float32,
            ),
            world_range = torch.tensor(
                np.stack([batch.input_data.world_range for batch in batches], 0), dtype=torch.float32,
            ) if 'world_range' in batches[0].input_data else None,
        ),
        aug_cfg = edict(
            img_aug_aff = torch.tensor(
                np.stack([batch.aug_cfg.img_aug_aff for batch in batches], 0), dtype=torch.float32
            ),
            bev_aug_aff = torch.tensor(
                np.stack([batch.aug_cfg.bev_aug_aff for batch in batches], 0), dtype=torch.float32
            ),
        ),
        labels = edict(
            density_image = torch.tensor(
                np.stack([batch.labels.density_image for batch in batches], 0), dtype=torch.float32
            ) if 'density_image' in batches[0].labels else None,  #2D density-map
            density_bev = torch.tensor(
                np.stack([batch.labels.density_bev for batch in batches], 0), dtype=torch.float32
            ) if 'density_bev' in batches[0].labels else None,    #bev density-map
            density_vox = torch.tensor(
                np.stack([batch.labels.density_vox for batch in batches], 0), dtype=torch.float32
            ) if 'density_vox' in batches[0].labels else None,    #3D density-map
        ),
        metas = edict(
            image_paths = [batch.metas.image_paths for batch in batches],
            img_shape   = [batch.metas.img_shape   for batch in batches],
            pt_bev      = [batch.labels.pt_bev     for batch in batches],
            pt_image    = [batch.labels.pt_image   for batch in batches],
        ),
    )

    return be


# img: np.ndarray with shape [H x W x 3], type np.uint8
# return: 3 x H x W
@PIPELINE.register_module()
def normalize_image(img):
    return (img.astype(np.float32)/127.5 - 1.0).transpose(2, 0, 1)

# img: np.ndarray with shape [3x H x W], type np.float32
# return: H x W x 3
@PIPELINE.register_module()
def unnormalize_image(img):
    return ((img+1.0)*127.5).astype(np.uint8).transpose(1, 2, 0)