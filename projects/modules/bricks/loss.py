
# Copyright (c) OpenMMLab. All rights reserved.
from .register import BRICKS

import copy
from typing import Dict, Optional

from torch import nn

BRICKS.register_module('MSELoss',      module=nn.MSELoss)
BRICKS.register_module('SmoothL1Loss', module=nn.SmoothL1Loss)

def build_loss_layer(cfg, *args, **kwargs):
    cfg_ = copy.deepcopy(cfg)
    cls_name = cfg_.pop('type')
    return BRICKS.get(cls_name)(*args, **kwargs, **cfg_)