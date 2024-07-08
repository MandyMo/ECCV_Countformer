

'''
    file:   register.py
    author: zhangxiong (1025679612@qq.com)
    date:   2023/06/23
'''

import numbers
import copy
from easydict import EasyDict as edict

def build_data_transforms(trans, registry):
    if isinstance(trans, (tuple, list)): # a compose transform
        return [build_data_transforms(tran, registry=registry) for tran in trans]
    if isinstance(trans, numbers.Number) or isinstance(trans, (str,)):
        return trans
    trans = copy.deepcopy(trans)
    cls_name = trans.pop('type')
    for (k, v) in trans.items():
        if isinstance(v, (dict,)) and 'type' in v: # a sub-transform component
            trans[k] = build_data_transforms(v, registry=registry)
        if isinstance(v, (list,)):
            trans[k] = build_data_transforms(v, registry=registry)
    return registry.get(cls_name)(**trans)

def build_dataset(cfg, registry):
    if isinstance(cfg, (tuple, list)):
        return [build_dataset(sub_dataset, registry=registry) for sub_dataset in cfg]
    elif cfg['type'] == 'CBGSDataset':
        return registry.get(cfg['type'])(stub_data=build_dataset(cfg['dataset'], registry=registry))
    else:
        cfg.update(dict(trans = build_data_transforms(cfg.pop('data_pipeline'), registry=registry.children['projects'])))
        return registry.get(cfg.pop('type'))(**cfg)

def build_dataloader(cfg, datasets=['train', 'val'], registry=None):
    import torch
    def _create_loader_(cfg, dataset, sampler):
        if 'collate_fn' in cfg:
            cfg['collate_fn'] = registry.children['projects'].get(cfg.collate_fn)
        return torch.utils.data.DataLoader(
            dataset=dataset,
            sampler=sampler,
            **cfg,
        )
    R = edict()
    ddp =  torch.cuda.is_available() and torch.cuda.device_count()>1
    for part in datasets:
        data_cfg = cfg[part]
        dataset  = build_dataset(cfg=data_cfg.pop('dataset'), registry=registry)
        sampler  = torch.utils.data.distributed.DistributedSampler(dataset=dataset) if ddp else None
        R[part] = edict(
            dataset=dataset,
            sampler=sampler,
            loader=_create_loader_(cfg=data_cfg, dataset=dataset, sampler=sampler)
        )
    return R

from ..registry import Registry

DATASET  = Registry('dataset', build_func=build_dataloader)
PIPELINE = Registry('pipeline', build_func=build_data_transforms, parent=DATASET)

import torch
@DATASET.register_module()
class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()
        
    def preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return

        with torch.cuda.stream(self.stream):
            self.next_batch = self.to_cuda(self.next_batch, non_blocking=True)

    def pool(self):
        '''
            start the iterator
        '''
        R = self.next()
        while R is not None:
            yield R
            R = self.next()

    def to_cuda(self, data_dict, non_blocking=True, ignore_keys=['metas']):
        for k in data_dict.keys():
            if k in ignore_keys:
                continue
            if isinstance(data_dict[k], (dict,)):
                data_dict[k] = self.to_cuda(data_dict=data_dict[k], non_blocking=non_blocking,ignore_keys=ignore_keys)
            elif data_dict[k] is None or data_dict[k][0] is None: #ignore none type
                continue
            else:
                data_dict[k] = data_dict[k].cuda(non_blocking=True)
        return data_dict

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        next_batch = self.next_batch
        self.preload()
        return next_batch