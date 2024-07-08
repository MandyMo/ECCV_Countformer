
'''
    file:   register.py
    author: zhangxiong (1025679612@qq.com)
    date:   2023/07/20
'''

from ...registry import Registry

def build_optimizer(cfg, registry, model, epoch_iter):
    def _get_scaled_lr_(base_lr, cfg):
        if 'world_size' in cfg.optimizer:
            base_lr = base_lr * cfg.optimizer.world_size
        if 'batch_size' in cfg.optimizer:
            base_lr = base_lr * cfg.optimizer.batch_size
        return base_lr

    optim_cfg = cfg.optimizer.get(cfg.optimizer.activate)

    if 'scale_lr' in cfg.optimizer: #scale lr based on the world_size & batch_size
        optim_cfg.lr = _get_scaled_lr_(optim_cfg.lr, cfg)

    optimizer = build_parameter_optimizer(model=model, opt_name=cfg.optimizer.activate, cfg=cfg)
    sche_cfg  = cfg.lr_schedule.get(cfg.lr_schedule.activate)

    sche_cfg  = cfg.lr_schedule.get(cfg.lr_schedule.activate)
    scheduler = build_lr_schedule(
        sche_name = cfg.lr_schedule.activate, 
        optimizer = optimizer,
        max_iters=epoch_iter*cfg.train_cfg.epoches,
        base_lr=optim_cfg.lr,
        cfg=sche_cfg
    )

    return optimizer, scheduler

def build_lr_schedule(sche_name, optimizer, max_iters, base_lr, cfg):
    if sche_name == 'PolyScheduler':
        from .PolyScheduler import PolyScheduler
        scheduler = PolyScheduler(optimizer=optimizer, max_iters=max_iters, base_lr=base_lr, **cfg)
    elif sche_name == 'LinearStepScheduler':
        from .LinearStepScheduler import LinearStepScheduler
        scheduler = LinearStepScheduler(optimizer=optimizer, max_iters=max_iters, base_lr=base_lr, **cfg)
    elif sche_name == 'CosineScheduler':
        from .CosineScheduler import CosineScheduler
        scheduler = CosineScheduler(optimizer=optimizer, base_lr=base_lr, max_iters=max_iters, **cfg)
    else:
        scheduler = None
    return scheduler

def build_parameter_optimizer(opt_name, model, cfg):
    import torch
    def build_param_groups(model, groups_info, base_lr):
        import itertools
        if groups_info is None:
            return [{
                'params' : model.parameters(), 
                'lr' : base_lr,
                '_lr': base_lr,
            }]
        all_param_ids = list(itertools.chain(*[list(map(id, model.module._modules.get(module).parameters())) for (module, lr_scale) in groups_info.items()]))
        return [
            {'params' : getattr(model.module, module).parameters(), 'lr' : base_lr * lr_scale, '_lr' : base_lr * lr_scale} for (module, lr_scale) in groups_info.items()
        ] + [
            {'params' : filter(lambda p : id(p) not in all_param_ids, model.parameters()), 'lr' : base_lr, '_lr' : base_lr}
        ]
    
    return getattr(torch.optim, opt_name)(build_param_groups(model, groups_info=cfg.optimizer.get('param_group_cfg', None), base_lr=cfg.optimizer.get(opt_name).lr), **cfg.optimizer.get(opt_name))

OPTIMIZATION = Registry('optimization', build_func=build_optimizer)
