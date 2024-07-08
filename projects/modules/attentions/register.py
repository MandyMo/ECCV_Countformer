
'''
    file:   register.py
    author: zhangxiong (1025679612@qq.com)
    date:   2023/06/18
'''

from ..register import (
    MODELS
)
from ...registry import build_from_cfg

ATTENTION = MODELS

def build_attention(cfg, default_args=None):
    """Builder for attention."""
    return build_from_cfg(cfg, ATTENTION, default_args)

def build_feedforward_network(cfg, default_args=None):
    """Builder for feed-forward network (FFN)."""
    return build_from_cfg(cfg, ATTENTION, default_args)

def build_transformer_layer(cfg, default_args=None):
    """Builder for transformer layer."""
    return build_from_cfg(cfg, ATTENTION, default_args)