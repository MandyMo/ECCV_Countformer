

'''
    file:   naive_embedding.py
    author: zhangxiong (1025679612@qq.com)
    date:   2023/08/21
'''

from .register import EMBEDDINGS

import torch
import torch.nn as nn

@EMBEDDINGS.register_module()
class NaiveEmbedding(nn.Module):
    def __init__(self, embed_dims, *args, **kwargs):
        super(NaiveEmbedding, self).__init__()
        self.register_parameter('embedding', nn.Parameter(torch.randn(*embed_dims)))
    
    def forward(self, *args, **kwargs):
        return self.embedding