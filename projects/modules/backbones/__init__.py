
'''
    file:   __init__.py
    author: zhangxiong (1025679612@qq.com)
    date:   2023/06/18
'''

__all__ = ['BACKBONES']

from .register import (
    BACKBONES
)

from .resnet import *
from .vovnet import *
from .swin   import *