
'''
    file:   LinearStepScheduler.py
    author: zhangxiong (1025679612@qq.com)
    date:   2023/07/20
'''

from .register import OPTIMIZATION

@OPTIMIZATION.register_module()
class LinearStepScheduler(object):
    def __init__(self, optimizer, base_lr, max_iters, gamma, warmup_ratio, warmup_lr_ratio, mile_stones=[0.3, 0.7, 0.9]):
        self.optimizer        = optimizer
        self.base_lr          = base_lr
        self.gamma            = gamma
        self.warmup_lr_ratio  = warmup_lr_ratio
        self.warmup_ratio     = warmup_ratio
        self.max_iters        = max_iters
        self.mile_stones      = mile_stones

        self.warmup_end_iter  = int(max_iters * warmup_ratio) if warmup_ratio < 1.0 else int(warmup_ratio)
        self.warmup_lr        = base_lr * warmup_lr_ratio
        self.c_iters          = 0
    
    def state_dict(self):
        return {"base_lr": self.base_lr, "cur_iter": self.c_iters}

    def load_state_dict(self, st):
        self.c_iters = st['cur_iter']

    def get_lr(self, nstep):
        if nstep < self.warmup_end_iter:
            return (self.warmup_lr + (self.base_lr-self.warmup_lr) * nstep / self.warmup_end_iter) / self.base_lr
        else:
            for stage in range(len(self.mile_stones)):
                if nstep < self.mile_stones[stage] * self.max_iters:
                    break
                else:
                    stage = len(self.mile_stones)
            return self.gamma**stage

    def __repr__(self) -> str:
        format_str = self.__class__.__name__ + \
            f'(optimizer={self.optimizer}, ' + \
            f'base_lr={self.base_lr}, ' + \
            f'gamma={self.gamma}, ' + \
            f'warmup_lr_ratio={self.warmup_lr_ratio}, ' + \
            f'warmup_ratio={self.warmup_ratio}, ' + \
            f'max_iters={self.max_iters}, ' + \
            f'mile_stones={self.mile_stones}, ' + \
            f'warmup_end_iter={self.warmup_end_iter}, ' + \
            f'warmup_lr={self.warmup_lr})'
        return format_str

    def step(self):
        scale = self.get_lr(self.c_iters)
        self.c_iters += 1
        for group in self.optimizer.param_groups:
            group['lr'] = group['_lr'] * scale
        return scale
