
'''
    file:   register.py
    author: zhangxiong (1025679612@qq.com)
    date:   2023/06/18
'''

try:
    from torch.hub import load_state_dict_from_url 
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from ..register import (
    MODELS
)

BACKBONES = MODELS