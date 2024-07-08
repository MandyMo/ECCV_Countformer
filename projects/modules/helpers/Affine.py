
'''
    file:   Affine.py
    author: zhangxiong (1025679612@qq.com)
    date:   2023/06/23
'''

from .register import HELPERS
import math
import numpy as np
import random
import numbers

@HELPERS.register_module()
def get_rot_matrix_2d(theta, is_angle=True):
    return get_rot_matrix_z(theta=theta, is_angle=is_angle)[:3, :3]

@HELPERS.register_module()
def get_rot_matrix_z(theta, is_angle=True):
    radian = math.radians(theta) if is_angle else theta
    cv, sv = math.cos(radian), math.sin(radian)
    return np.array([
        [ cv, -sv, 0.0, 0.0], 
        [ sv,  cv, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

@HELPERS.register_module()
def batch_get_rot_matrix_z(theta, is_angle=True):
    radian = theta*math.pi/180.0 if is_angle else theta
    cv, sv = np.cos(radian), np.sin(radian)
    k_ones, k_zeros = np.ones(radian.shape[0]), np.zeros(radian.shape[0])
    return np.stack([
             cv,     -sv, k_zeros, k_zeros, 
             sv,      cv, k_zeros, k_zeros,
        k_zeros, k_zeros,  k_ones, k_zeros,
        k_zeros, k_zeros, k_zeros, k_ones
    ], -1).reshape(-1, 4, 4)

@HELPERS.register_module()
def get_rot_matrix_x(theta, is_angle=True):
    radian = math.radians(theta) if is_angle else theta
    cv, sv = math.cos(radian), math.sin(radian)
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0,  cv, -sv, 0.0],
        [0.0,  sv,  cv, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

@HELPERS.register_module()
def get_rot_matrix_y(theta, is_angle=True):
    radian = math.radians(theta) if is_angle else theta
    cv, sv = math.cos(radian), math.sin(radian)
    return np.array([
        [ cv, 0.0,  sv, 0.0],
        [0.0,   1,   0, 0.0],
        [-sv,   0,  cv, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

@HELPERS.register_module()
def get_scale_matrix_2d(sx, sy):
    return np.array([
        [sx, 0, 0.0],
        [0, sy, 0.0],
        [0,  0, 1.0],
    ])

@HELPERS.register_module()
def get_scale_matrix_3d(sx, sy, sz):
    return np.array([
        [ sx, 0.0, 0.0, 0.0],
        [0.0,  sy, 0.0, 0.0],
        [0.0, 0.0,  sz, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

@HELPERS.register_module()
def get_trans_matrix_2d(tx, ty):
    return np.array([
        [1.0, 0.0,  tx],
        [0.0, 1.0,  ty],
        [0.0, 0.0, 1.0],
    ])

@HELPERS.register_module()
def get_trans_matrix_3d(tx, ty, tz):
    return np.array([
        [1.0, 0.0, 0.0,  tx],
        [0.0, 1.0, 0.0,  ty],
        [0.0, 0.0, 1.0,  tz],
        [0.0, 0.0, 0.0, 1.0],
    ])

@HELPERS.register_module()
def get_horizontal_flip_matrix_2d(w):
    return np.array([
        [-1.0,  0.0,   w],
        [ 0.0,  1.0, 0.0],
        [ 0.0,  0.0, 1.0],
    ])

@HELPERS.register_module()
def get_vertical_flip_matrix_2d(h):
    return np.array([
        [1.0,  0.0, 0.0],
        [0.0, -1.0,   h],
        [0.0,  0.0, 1.0],
    ])

@HELPERS.register_module()
def range_float_value(a):
    return a if isinstance(a, numbers.Number) else random.uniform(a[0], a[1])

@HELPERS.register_module()
def get_affine_matrix(scale, angle, center, crop, mode='scr'):
    funcs = {
        'src' : get_affine_matrix_src,
        'scr' : get_affine_matrix_scr
    }
    return funcs[mode](scale=scale, angle=angle, center=center, crop=crop)

@HELPERS.register_module() # scale -> crop -> rotate
def get_affine_matrix_scr(scale, angle, center, crop=[0, 0, 0, 0]):
    sx, sy, ctx, cty, cpx, cpy = scale[0], scale[1], center[0], center[1], crop[0], crop[1]
    R = get_rot_matrix_2d(theta=angle, is_angle=True)
    S = get_scale_matrix_2d(sx=sx, sy=sy)
    C = -R @ np.array([cpx+ctx, cpy+cty, 0.0]).reshape(3, 1) + np.array([ctx, cty, 0.0]).reshape(3, 1)
    T = get_trans_matrix_2d(C[0][0], C[1][0])
    return T @ R @ S

@HELPERS.register_module() # scale -> rotate -> crop
def get_affine_matrix_src(scale, angle, center, crop=[0, 0, 0, 0]):
    sx, sy, ctx, cty, cpx, cpy = scale[0], scale[1], center[0], center[1], crop[0], crop[1]
    R = get_rot_matrix_2d(theta=angle, is_angle=True)
    S = get_scale_matrix_2d(sx=sx, sy=sy)
    C = (np.eye(3) - R) @ np.array([ctx, cty, 1.0])
    T = get_trans_matrix_2d(C[0]-cpx, C[1]-cpy)
    aff_mat = T @ R @ S # T * R * S * p
    return aff_mat