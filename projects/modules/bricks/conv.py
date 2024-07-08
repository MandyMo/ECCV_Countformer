
# Copyright (c) OpenMMLab. All rights reserved.
from .register import BRICKS

import inspect
from typing import Dict, Optional

from torch import nn

BRICKS.register_module('Conv1d',   module=nn.Conv1d)
BRICKS.register_module('Conv2d',   module=nn.Conv2d)
BRICKS.register_module('Conv3d',   module=nn.Conv3d)
BRICKS.register_module('Conv',     module=nn.Conv2d)
BRICKS.register_module('deconv3d', module=nn.ConvTranspose3d)
BRICKS.register_module('deconv2d', module=nn.ConvTranspose2d)
BRICKS.register_module('deconv1d', module=nn.ConvTranspose1d)
BRICKS.register_module('Deconv3d', module=nn.ConvTranspose3d)
BRICKS.register_module('Deconv2d', module=nn.ConvTranspose2d)
BRICKS.register_module('Deconv1d', module=nn.ConvTranspose1d)



def build_conv_layer(cfg: Optional[Dict], *args, **kwargs) -> nn.Module:
    """Build convolution layer.

    Args:
        cfg (None or dict): The conv layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an conv layer.
        args (argument list): Arguments passed to the `__init__`
            method of the corresponding conv layer.
        kwargs (keyword arguments): Keyword arguments passed to the `__init__`
            method of the corresponding conv layer.

    Returns:
        nn.Module: Created conv layer.
    """
    if cfg is None:
        cfg_ = dict(type='Conv2d')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    cls_name= cfg_.pop('type')

    return BRICKS.get(cls_name)(*args, **kwargs, **cfg_) 