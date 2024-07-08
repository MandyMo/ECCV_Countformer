
'''
    file:   train.py
    author: zhangxiong (1025679612@qq.com)
    date:   2023/07/20
'''

def debug_mode():
    import os
    import sys
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(cur_dir, '..'))
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    print('finished setting debug mode')

def parse_args():
    import argparse
    import random
    parser = argparse.ArgumentParser(description = 'neofusion')
    parser.add_argument('--cfg_path', type=str, default='projects/configs/cross_view/voxel/swin_benchmark_lr_cam_embed_img_vox.py')
    parser.add_argument('--apply_image_mask', type=int, default=0)
    parser.add_argument('--nccl_port', type=int, default=random.randint(50000, 65535))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deter', type=int, default=0)
    parser.add_argument('--resume',type=int, default=0)
    parser.add_argument('--chkp_path', type=str, default=None)
    parser.add_argument('--only_weight', type=int, default=0)
    parser.add_argument('--amp', type=int, default=0)
    args = parser.parse_args()
    return args

def limit_cv_thread():
    import cv2
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)

def seed_random(seed=42,deter=True):
    import random
    import os
    import torch
    import numpy as np

    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.use_deterministic_algorithms(deter>=1)

'''
    pay attention
'''
# debug_mode() #debug mode
args = parse_args()
limit_cv_thread()
seed_random(args.seed, args.deter)
#===========================================================================

import os
from os import path as osp
import torch
from projects import MODELS, DATASET, HELPERS, OPTIMIZATION
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast as autocast

class Trainer(object):
    def __init__(self, args, rank=0, world_size=1, amp=False):
        self.args, self.device_count, self.rank, self.world_size, self.amp = args, torch.cuda.device_count(), rank, world_size, amp
        self.device = torch.device('cuda:{}'.format(self.rank)) if torch.cuda.is_available() else torch.device('cpu')
        
        self.glb_step = -1
        self.start_epoch = 0

        self.__init_multi_gpu__()
        self.__load_configs__()
        self.__create_logger__()
        self._create_dataset()
        self._create_model()
        self.__dump_infos__()

    def __create_logger__(self):
        from torch.utils.tensorboard.writer import SummaryWriter
        self.tf_logger = SummaryWriter(self.cfg.optimization.logger_cfg.log_dir) if self.is_master_node else None
        self.tx_logger = HELPERS.get('Logger')(osp.join(self.cfg.optimization.logger_cfg.log_dir, osp.basename(self.args.cfg_path).replace('.py', '.txt'))) if self.is_master_node else None

    def __dump_infos__(self):
        if not self.is_master_node:
            return
        import shutil
        def copytree(src, dst, symlinks=False, ignore=None):
            if not osp.exists(dst):
                os.makedirs(dst)
            for item in os.listdir(src):
                s = osp.join(src, item)
                d = osp.join(dst, item)
                if osp.isdir(s):
                    copytree(s, d, symlinks, ignore)
                elif s.endswith('.py') or s.endswith('.sh'):
                    shutil.copy2(s, d)
        cfg_path = self.args.cfg_path
        shutil.copy2(cfg_path, osp.join(self.cfg.optimization.logger_cfg.log_dir, osp.basename(cfg_path)))
        [copytree(_dir, osp.join(self.cfg.optimization.logger_cfg.log_dir, 'src', _dir)) for _dir in ['projects', 'tools']]
        self.tx_logger.log_str(HELPERS.get('collect_env')()).log_str(self.args).log_str(self.cfg)
        self.tx_logger.log_str(self.model).log_str(self.optimizer).log_str(self.opt_scheduler)
        self.tx_logger.log_str(f'[sample: {len(self.dataset.train.dataset)}, total_iters:{len(self.dataset.train.loader) * self.cfg.optimization.train_cfg.epoches}, iter_per_epoch: {len(self.dataset.train.loader)}]')
        self.tx_logger.flush()

    def __load_configs__(self):
        from projects import Config
        import time
        self.cfg = Config.fromfile(self.args.cfg_path)
        time_suffix = time.strftime("[%Y-%m-%d::%H:%M:%S]", time.localtime(time.time()))
        self.cfg.optimization.logger_cfg.log_dir      = osp.join(self.cfg.optimization.logger_cfg.log_dir,      time_suffix)
        self.cfg.optimization.logger_cfg.chkp_sv_path = osp.join(self.cfg.optimization.logger_cfg.chkp_sv_path, time_suffix, 'checkpoints')
        self.cfg.optimization.optimizer.world_size    = self.world_size
        self.cfg.optimization.optimizer.batch_size    = self.cfg.data.train.batch_size
        self.cfg.model.amp                            = self.cfg.optimization.get('amp', self.amp)

    def __init_multi_gpu__(self):
        self.is_master_node=True if self.rank==0 else False
        if self.device_count < 2:
            return
        torch.cuda.set_device(self.rank)
        dist.init_process_group(
            backend='nccl', 
            init_method='tcp://127.0.0.1:{}'.format(self.args.nccl_port), 
            world_size=self.world_size, 
            rank=self.rank
        )

    def _create_model(self):
        model = MODELS.build(self.cfg.model).to(self.device)
        #parallel training
        if self.device_count <= 1:
            self.model = nn.DataParallel(model).to(self.device)
        else:
            self.model = nn.parallel.DistributedDataParallel(model, device_ids=[self.rank], output_device=self.rank, broadcast_buffers=False, find_unused_parameters=True).to(self.device)

        if self.cfg.optimization.train_cfg.sync_bn and self.device_count > 1:
            self.model=torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

        self.optimizer, self.opt_scheduler = OPTIMIZATION.build(cfg=self.cfg.optimization, model=self.model, epoch_iter=len(self.dataset.train.loader))

        chkp_path = self.args.chkp_path if self.args.resume > 0 else None

        if self.cfg.optimization.train_cfg.resume and osp.isfile(self.cfg.optimization.train_cfg.chkp_path):
            chkp_path = self.cfg.optimization.train_cfg.chkp_path

        if chkp_path is not None:
            self.load_chkp(chkp_path)
            [self.tx_logger.log_str('load pretrained state_dict {}'.format(chkp_path)) if self.rank==0 else None]
        
        #mixed-precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
            
    def _create_dataset(self):
        self.dataset = DATASET.build(self.cfg.data)
        self.k_input = ['images', 'extrins', 'intrins', 'pcd_points', 'image_aug_aff', 'bev_aug_aff']

    def dump_loss(self, loss_dict, prefix='bev_counting'):
        if self.world_size > 1:
            HELPERS.get('all_reduce')(loss_dict)
        if not self.is_master_node: #not master node
            return
        for (k, v) in loss_dict.items():
            if isinstance(v, (dict,)):
                self.dump_loss(loss_dict=v, prefix=prefix+'.'+k)
            else:
                self.tf_logger.add_scalar(prefix+'/'+k, v/self.world_size if isinstance(v, (torch.Tensor,)) else v, global_step=self.glb_step)

    def dump_record(self, input_dict, p, step=None, prefix='bev_counting'):
        dump_fn = HELPERS.get(self.cfg.optimization.logger_cfg.log_record_fn)
        step    = self.glb_step if step is None else step
        dump_fn(logger=self.tf_logger, batch=input_dict, p=p, step=step, prefix=prefix)

    def close_logger(self):
        if self.rank != 0:
            return
        self.tf_logger.close()
        self.tx_logger.close()

    def evaluate(self, apply_mask=True):
        def gather_tensor(tensor):
            if self.world_size > 1:
                tensor_list = [torch.zeros_like(tensor) for _ in range(self.world_size)]
                dist.all_gather(tensor_list, tensor)
                return sum(tensor_list)
            else:
                return tensor
        
        import tqdm
        from easydict import EasyDict as edict
        self.model.eval()
        total_mae_bev, total_mae_image, total_nae_bev, total_nae_image = 0, 0, 0, 0
        start_step = self.epoch * len(self.dataset.val.loader)
        img_density_map_scale, scene_density_map_scale = self.cfg.get('img_density_map_scale', 1.0), self.cfg.get('scene_density_map_scale', 1.0)

        for (step, input_dict) in enumerate(tqdm.tqdm(DATASET.get('DataPrefetcher')(self.dataset.val.loader).pool(), total=len(self.dataset.val.loader)) if self.rank==0 else DATASET.get('DataPrefetcher')(self.dataset.val.loader).pool()):
            input_dict = edict(input_dict)
            p = self.model(input_dict)

            #get mask
            mask = nn.functional.interpolate(input_dict.input_data.image_masks, p.image_density_map.shape[3:], mode='bilinear')/255.0
            mask[mask>=0.3] = 1.0
            mask[mask< 0.3] = 0

            if self.is_master_node:
                self.dump_record(input_dict=input_dict, p=p, step=step+start_step, prefix='bev_counting_test')
            batch_mae_bev   = torch.tensor(0, device=self.device, dtype=torch.float32)
            batch_mae_image = torch.tensor(0, device=self.device, dtype=torch.float32)
            batch_nae_bev   = torch.tensor(0, device=self.device, dtype=torch.float32)
            batch_nae_image = torch.tensor(0, device=self.device, dtype=torch.float32)

            bs = input_dict.input_data.image_set.shape[0]
            n_view = input_dict.input_data.image_set.shape[1]

            if apply_mask:
                p.image_density_map = p.image_density_map * mask[:,:,None]
                input_dict.labels.density_image = input_dict.labels.density_image * mask
            
            for b in range(bs):
                g_bev = input_dict.metas.pt_bev[b].shape[0]
                if 'bev_density_map' in p: #bev pillar representation
                    p_bev = p.bev_density_map[b].sum().clamp(min=0) / scene_density_map_scale
                else: #voxel representation
                    p_bev = p.vox_density_map[-1][b].sum().clamp(min=0) / scene_density_map_scale
                mae_bev = (p_bev-g_bev).abs()
                nae_bev = (mae_bev / (g_bev+1e-3)).clamp(max=1.0)
                batch_mae_bev = batch_mae_bev + mae_bev
                batch_nae_bev = batch_nae_bev + nae_bev
                for(p_image, g_image) in zip(p.image_density_map[b], input_dict.labels.density_image[b]):
                    p_image   = p_image.sum().clamp(min=0) / img_density_map_scale
                    g_image   = g_image.sum() / img_density_map_scale
                    mae_image = (p_image-g_image).abs()
                    nae_image = (mae_image / (g_image+1e-3)).clamp(max=1.0)
                    batch_mae_image = batch_mae_image + mae_image
                    batch_nae_image = batch_nae_image + nae_image
            #gather from other gpus
            total_nae_bev    += float(gather_tensor(batch_nae_bev))
            total_mae_bev    += float(gather_tensor(batch_mae_bev))
            total_nae_image  += float(gather_tensor(batch_nae_image))
            total_mae_image  += float(gather_tensor(batch_mae_image))
        total_samples = len(self.dataset.val.dataset)

        if not self.is_master_node:
            return
        
        loss_dict = edict(
            nae_bev=total_nae_bev/total_samples,
            mae_bev=total_mae_bev/total_samples,
            nae_image=total_nae_image/total_samples/n_view,
            mae_image=total_mae_image/total_samples/n_view,
        )
        for (k, v) in loss_dict.items():
            self.tf_logger.add_scalar('bev_counting_test'+'/'+k, v, global_step=self.epoch)
        return loss_dict

    def train(self, epoch):
        import tqdm
        from easydict import EasyDict as edict
        self.epoch = epoch + 1
        def clip_grad_norm(loss_dict):
            if self.cfg.optimization.optimizer.clip_norm:
                loss_dict.max_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfg.optimization.optimizer.max_norm)

        def get_max_norm(loss_dict):
            loss_dict.max_grad_norm = max([p.grad.data.norm(2) for p in list(filter(lambda p: p.grad is not None, self.model.parameters()))])
        
        self.model.train()
        if self.device_count > 1:
            self.dataset.train.sampler.set_epoch(epoch)
        optimizer, scheduler = self.optimizer, self.opt_scheduler

        for input_dict in tqdm.tqdm(DATASET.get('DataPrefetcher')(self.dataset.train.loader).pool(), desc=f"epoch{epoch}", total=len(self.dataset.train.loader)) if self.rank==0 else DATASET.get('DataPrefetcher')(self.dataset.train.loader).pool():
            self.glb_step += 1
            input_dict = edict(input_dict)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.amp):
                p = self.model(input_dict)
                loss_dict = self.model.module.loss_fn(p=p, input_dict=input_dict)

            self.scaler.scale(loss_dict.total_loss).backward()
            clip_grad_norm(loss_dict=loss_dict)
            get_max_norm(loss_dict=loss_dict)
            self.scaler.step(optimizer)
            self.scaler.update()

            loss_dict.learning_rate = scheduler.step()

            if self.glb_step % self.cfg.optimization.logger_cfg.tf_log_loss_period == 0:
                with torch.no_grad():
                    self.dump_loss(loss_dict, prefix='bev_counting_train')
                if self.is_master_node:
                    self.tx_logger.log_str({k:round(float(v), 4) for (k, v) in loss_dict.items()})
            
            if self.is_master_node and self.glb_step % self.cfg.optimization.logger_cfg.tf_log_record_period == 0:
                with torch.no_grad():
                    self.dump_record(input_dict=input_dict, p=p, prefix='bev_counting_train')

        #release the gpu memory
        del input_dict
        del p
        del loss_dict
        torch.cuda.empty_cache()

    def dump_chkp(self, sv_path):
        st = {
            'net':self.model.module.state_dict(),
            'optimizer':self.optimizer.state_dict(),
            'lr_scheduler':self.opt_scheduler.state_dict(),
            'metas': {
                'epoch':self.epoch,
                'glb_step':self.glb_step,
            },
        }
        torch.save(st, sv_path)

    def load_chkp(self, sv_path):
        st = torch.load(sv_path, map_location='cpu')
        if 'net' in st:
            HELPERS.get('copy_state_dict')(self.model.module.state_dict(), st['net'])

        if self.args.only_weight > 0: #only load the parameters of the neural network
            return
        
        if 'optimizer' in st:
            self.optimizer.load_state_dict(st['optimizer'])
        if 'lr_scheduler' in st:
            self.opt_scheduler.load_state_dict(st['lr_scheduler'])
        if 'metas' in st and 'epoch' in st['metas']:
            self.start_epoch = st['metas']['epoch']
        if 'metas' in st and 'glb_step' in st['metas']:
            self.glb_step = st['metas']['glb_step']

    def do_train(self):
        import math
        sv_folder = self.cfg.optimization.logger_cfg.chkp_sv_path
        os.makedirs(sv_folder, exist_ok=True)
        best_nae_bev, best_mae_bev, best_nae_img, best_mae_img=math.inf, math.inf, math.inf, math.inf
        for epoch in range(self.start_epoch, self.cfg.optimization.train_cfg.epoches):
            self.train(epoch)
            chkp_path = osp.join(sv_folder, f'{epoch+1}.pth')
            if (epoch+1) % self.cfg.optimization.logger_cfg.chkp_sv_period!=0:
                continue

            if self.is_master_node and self.epoch % 10 == 0:
                self.dump_chkp(chkp_path)

            with torch.no_grad():
                eval_result = self.evaluate(apply_mask = True if self.args.apply_image_mask > 0 else False)

            if self.is_master_node:
                self.tx_logger.log_str(eval_result)
                if best_mae_bev > float(eval_result.mae_bev):
                    best_chkp_path = osp.join(sv_folder, 'best_mae_bev.pth')
                    best_mae_bev = float(eval_result.mae_bev)
                    self.tx_logger.log_str(f'obtain a better mae_bev {round(best_mae_bev, 2)} checkpoint {epoch+1}.pth')
                    self.dump_chkp(best_chkp_path)

                if best_nae_bev > float(eval_result.nae_bev):
                    best_chkp_path = osp.join(sv_folder, 'best_nae_bev.pth')
                    best_nae_bev = float(eval_result.nae_bev)
                    self.tx_logger.log_str(f'obtain a better nae_bev {round(best_nae_bev, 2)} checkpoint {epoch+1}.pth')
                    self.dump_chkp(best_chkp_path)
                
                if best_mae_img > float(eval_result.mae_image):
                    best_chkp_path = osp.join(sv_folder, 'best_mae_image.pth')
                    best_mae_img = float(eval_result.mae_image)
                    self.tx_logger.log_str(f'obtain a better mae_image {round(best_mae_img, 2)} checkpoint {epoch+1}.pth')
                    self.dump_chkp(best_chkp_path)
                    
                if best_nae_img > float(eval_result.nae_image):
                    best_chkp_path = osp.join(sv_folder, 'best_nae_image.pth')
                    best_nae_img = float(eval_result.nae_image)
                    self.tx_logger.log_str(f'obtain a better nae_image {round(best_nae_img, 2)} checkpoint {epoch+1}.pth')
                    self.dump_chkp(best_chkp_path)
                
                self.tx_logger.flush()

        self.close_logger()
        import sys
        sys.exit(0)
 
def worker(rank, world_size, args, **kargs):
    trainer = Trainer(args=args, rank=rank, world_size=world_size, amp=True if args.amp > 0 else False)
    trainer.do_train()

def main():
    world_size = torch.cuda.device_count()
    if world_size <= 1:
        worker(rank=0, world_size=1, args=args)
    else:
        import torch.multiprocessing as mp
        mp.spawn(worker, nprocs=world_size, args=(world_size,args,))

if __name__ == '__main__':
    main()
