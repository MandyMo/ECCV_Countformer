
'''
    file:   helper.py
    author: zhangxiong (1025679612@qq.com)
    date:   2023/06/18
'''

from .register import HELPERS

import torch.nn as nn
import numpy as np
import random
import torch
import os

@HELPERS.register_module()
def build_convs(cs, ks, ss, ps, last_relu=False, last_bn=False):
    ops = []
    n_layer = len(cs)-1
    for _ in range(n_layer-1):
        c_in, c_out, k, s, p = cs[_], cs[_+1], ks[_], ss[_], ps[_]
        ops.append(nn.Conv2d(c_in, c_out, k, s, padding=p, bias=False))
        ops.append(nn.BatchNorm2d(c_out))
        ops.append(nn.ReLU(inplace=True))
    
    c_in, c_out, k, s, p = cs[-2], cs[-1], ks[-1], ss[-1], ps[-1]
    
    ops.append(nn.Conv2d(c_in, c_out, k, s, padding=p, bias=(not last_bn)))
    if last_bn:
        ops.append(nn.BatchNorm2d(c_out))
    if last_relu:
        ops.append(nn.ReLU(inplace=True))

    return nn.Sequential(*ops)

@HELPERS.register_module()
def copy_state_dict(cur_state_dict, pre_state_dict, prefix = ''):
	def _get_params(key):
		key = prefix + key
		if key in pre_state_dict:
			return pre_state_dict[key]
		return None
	
	for k in cur_state_dict.keys():
		v = _get_params(k)
		try:
			if v is None:
				print('parameter {} not found'.format(k))
				continue
			try:
				cur_state_dict[k].copy_(v)
			except:
				cur_state_dict[k].copy_(v.view(cur_state_dict[k].shape))
		except:
			print('copy param {} failed'.format(k))
			continue

@HELPERS.register_module()
def build_config(cfg_path):
    import importlib
    cfg =importlib.import_module(cfg_path)
    return cfg

@HELPERS.register_module()
def seed_training_process(seed=42,deter=True):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    # torch.set_deterministic(deter)  # avoiding nondeterministic algorithms (see https://pytorch.org/docs/stable/notes/randomness.html)
    torch.use_deterministic_algorithms(deter)

@HELPERS.register_module()
def all_reduce(tensor_set, op=torch.distributed.ReduceOp.SUM):
    if isinstance(tensor_set, (dict,)):
        for (k, v) in tensor_set.items():
            all_reduce(v)
    elif isinstance(tensor_set, (list,)):
        for v in tensor_set:
            all_reduce(v)
    elif isinstance(tensor_set, (torch.Tensor,)):
        torch.distributed.all_reduce(tensor_set, op=op)
    else:
        pass        

@HELPERS.register_module()
def limit_opencv_threads(num_thread=0):
    import cv2
    cv2.setNumThreads(num_thread)
    cv2.ocl.setUseOpenCL(False)
