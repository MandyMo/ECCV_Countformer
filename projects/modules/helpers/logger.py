
'''
    file:   logger.py
    author: zhangxiong (1025679612@qq.com)
    date:   2023/07/20
'''

from .register import HELPERS
import os
from os import path as osp
import json

@HELPERS.register_module()
class Logger(object):
    def __init__(self, abs_path, mode='w'):
        self.abs_path = abs_path
        if self.abs_path is not None:
            os.makedirs(osp.dirname(self.abs_path), exist_ok=True)
        self.writer = open(self.abs_path, 'w') 
    
    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.abs_path = None
        self.writer, self.abs_path = None, None

    def flush(self):
        if self.writer is not None:
            self.writer.flush()

    def time(self):
        import time
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))

    def reset(self, abs_path, mode='w'):
        self.close()
        self.abs_path = abs_path
        os.makedirs(osp.basename(self.abs_path), exist_ok=True)
        self.writer = open(self.abs_path, mode)
    
    def log_str(self, msg, nl=True):
        if isinstance(msg, (dict, )):
            msg = json.dumps(msg, indent=2)
        else:
            msg = str(msg)
        self.writer.write(f'[{self.time()}] ' + \
            msg + '\n' if nl else ''
        )
        return self
