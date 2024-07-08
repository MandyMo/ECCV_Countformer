
'''
    file:   CosineScheduler.py
    author: zhangxiong (1025679612@qq.com)
    date:   2023/07/26
'''

from .register import OPTIMIZATION
import math

@OPTIMIZATION.register_module()
class CosineScheduler(object):
    def __init__(self, optimizer, base_lr, max_iters, warmup_ratio, warmup_lr_ratio, total_cycles=5, min_lr_ratio=1e-4):
        self.optimizer        = optimizer
        self.base_lr          = base_lr
        self.min_lr_ratio     = min_lr_ratio
        self.total_cycles     = total_cycles
        self.max_iters        = max_iters
        self.c_iters          = 0
        self.warmup_lr_ratio  = warmup_lr_ratio
        self.warmup_ratio     = warmup_ratio
        self.warmup_end_iter  = int(max_iters * warmup_ratio) if warmup_ratio < 1.0 else int(warmup_ratio)
        self.warmup_lr        = base_lr * warmup_lr_ratio
        self.T                = (max_iters - self.warmup_end_iter)  / (total_cycles+0.5)
    
    def state_dict(self):
        return {"base_lr": self.base_lr, "cur_iter": self.c_iters}

    def load_state_dict(self, st):
        self.c_iters = st['cur_iter']

    def get_lr(self, nstep):
        if nstep < self.warmup_end_iter:
            return (self.warmup_lr + (self.base_lr-self.warmup_lr) * nstep / self.warmup_end_iter) / self.base_lr
        else:
            t = math.fmod(nstep-self.warmup_end_iter, self.T) / self.T
            s = math.cos(2*t*math.pi)
            s = (s+1.0)/2.0
            return s*(1-self.min_lr_ratio) + self.min_lr_ratio

    def __repr__(self) -> str:
        format_str = self.__class__.__name__ + \
            f'(optimizer={self.optimizer}, ' + \
            f'base_lr={self.base_lr}, ' + \
            f'min_lr_ratio={self.min_lr_ratio}, ' + \
            f'total_cycles={self.total_cycles}, ' + \
            f'max_iters={self.max_iters}, ' + \
            f'T={self.T})'
        return format_str

    def step(self):
        scale = self.get_lr(self.c_iters)
        self.c_iters += 1
        for group in self.optimizer.param_groups:
            group['lr'] = group['_lr'] * scale
        return scale
