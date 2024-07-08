

'''
    file:   register.py
    author: zhangxiong (1025679612@qq.com)
    date:   2023/06/18
'''

__all__ = ['MODELS']

from ..registry import Registry

def build_model(cfg, registry):
    import copy
    import torch.nn as nn
    if isinstance(cfg, (list, tuple)):
        return nn.Sequential([build_model(sub_cfg, registry=registry) for sub_cfg in cfg])
    cfg = copy.deepcopy(cfg)
    cls_name = cfg.pop('type')
    return registry.get(cls_name)(**cfg, cfg=cfg)

MODELS = Registry('models', build_func=build_model)