
'''
    file:   AverageMeter.py
    author: zhangxiong (1025679612@qq.com)
    date:   2023/06/18
'''

from .register import HELPERS

@HELPERS.register_module()
class AverageMeter(object):
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.initialized = True

    def update(self, val, weight=0.01):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = (1 - weight) * self.val + weight * val
        self.avg = self.val

    def value(self):
        return self.val

    def average(self):
        return self.avg

    def reset(self):
        self.initialized = False
