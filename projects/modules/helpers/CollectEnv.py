
'''
    file:   CollectEnv.py
    author: zhangxiong (1025679612@qq.com)
    date:   2023/06/18
'''

from .register import HELPERS

import os.path as osp
import subprocess
import sys
from collections import defaultdict
import cv2
import torch

def get_build_config():
    return torch.__config__.show()

@HELPERS.register_module()
def collect_env():
    env_info = {}
    env_info['sys.platform'] = sys.platform
    env_info['Python'] = sys.version.replace('\n', '')

    cuda_available = torch.cuda.is_available()
    env_info['CUDA available'] = cuda_available
    if cuda_available:
        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        for name, device_ids in devices.items():
            env_info['GPU ' + ','.join(device_ids)] = name

        CUDA_HOME = torch.utils.cpp_extension.CUDA_HOME
        env_info['CUDA_HOME'] = CUDA_HOME
        if CUDA_HOME is not None and osp.isdir(CUDA_HOME):
            try:
                nvcc = osp.join(CUDA_HOME, 'bin/nvcc')
                nvcc = subprocess.check_output(f'"{nvcc}" -V', shell=True)
                nvcc = nvcc.decode('utf-8').strip()
                release = nvcc.rfind('Cuda compilation tools')
                build = nvcc.rfind('Build ')
                nvcc = nvcc[release:build].strip()
            except subprocess.SubprocessError:
                nvcc = 'Not Available'
            env_info['NVCC'] = nvcc

    try:
        import sysconfig
        cc = sysconfig.get_config_var('CC')
        if cc:
            cc = osp.basename(cc.split()[0])
            cc_info = subprocess.check_output(f'{cc} --version', shell=True)
            env_info['GCC'] = cc_info.decode('utf-8').partition(
                '\n')[0].strip()
        else:
            import locale
            import os
            from distutils.ccompiler import new_compiler
            ccompiler = new_compiler()
            ccompiler.initialize()
            cc = subprocess.check_output(
                f'{ccompiler.cc}', stderr=subprocess.STDOUT, shell=True)
            encoding = os.device_encoding(
                sys.stdout.fileno()) or locale.getpreferredencoding()
            env_info['MSVC'] = cc.decode(encoding).partition('\n')[0].strip()
            env_info['GCC'] = 'n/a'
    except subprocess.CalledProcessError:
        env_info['GCC'] = 'n/a'

    env_info['PyTorch'] = torch.__version__
    env_info['PyTorch compiling details'] = get_build_config()

    try:
        import torchvision
        env_info['TorchVision'] = torchvision.__version__
    except ModuleNotFoundError:
        pass
    env_info['OpenCV'] = cv2.__version__
    return env_info
