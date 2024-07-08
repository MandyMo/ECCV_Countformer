
'''
    file:   PolyScheduler.py
    author: zhangxiong (1025679612@qq.com)
    date:   2023/07/20
'''

from .register import OPTIMIZATION

@OPTIMIZATION.register_module()
class PolyScheduler(object):
    def __init__(self, optimizer, base_lr, max_iters, power, warmup_ratio, warmup_lr_ratio):
        self.optimizer        = optimizer
        self.base_lr          = base_lr
        self.power            = power
        self.warmup_lr_ratio  = warmup_lr_ratio
        self.warmup_ratio     = warmup_ratio
        self.max_iters        = max_iters

        self.warmup_end_iter  = int(max_iters * warmup_ratio)
        self.warmup_lr        = base_lr * warmup_lr_ratio
        self.c_iters          = 0

    def state_dict(self):
        return {"base_lr": self.base_lr, "cur_iter": self.c_iters}

    def get_lr(self, nstep):
        if nstep < self.warmup_end_iter:
            ratio = nstep / self.warmup_end_iter
            return (self.warmup_lr + (ratio ** self.power) * (self.base_lr - self.warmup_lr)) / self.base_lr
        else:
            n_total_step = self.max_iters - self.warmup_end_iter
            ratio = (nstep - self.warmup_end_iter) / n_total_step
            return (1-ratio) ** self.power

    def load_state_dict(self, st):
        self.c_iters = st['cur_iter']

    def __repr__(self) -> str:
        format_str = self.__class__.__name__ + \
            f'(optimizer={self.optimizer}, ' + \
            f'base_lr={self.base_lr}, ' + \
            f'power={self.power}, ' + \
            f'warmup_lr_ratio={self.warmup_lr_ratio}, ' + \
            f'warmup_ratio={self.warmup_ratio}, ' + \
            f'max_iters={self.max_iters}, ' + \
            f'warmup_end_iter={self.warmup_end_iter}, ' + \
            f'warmup_lr={self.warmup_lr})'
        return format_str

    def step(self):
        scale = self.get_lr(self.c_iters)
        self.c_iters += 1
        for group in self.optimizer.param_groups:
            group['lr'] = group['_lr'] * scale
        return scale
