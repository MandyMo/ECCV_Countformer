
'''
    file:   __init__.py
    author: zhangxiong (1025679612@qq.com)
    date:   2023/07/20
'''

__all__ = ['OPTIMIZATION']

from .register import (
    OPTIMIZATION
)

from .PolyScheduler       import *
from .LinearStepScheduler import *
from .CosineScheduler     import *