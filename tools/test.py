
'''
    file:   train.py
    author: zhangxiong (1025679612@qq.com)
    date:   2023/07/20
'''

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import os
# os.environ['CUDA_VISIBLE_DEVICES']='0'
from os import path as osp
import sys
sys.path.append(osp.join(osp.dirname(osp.abspath(__file__)), '..'))


import torch
from projects import MODELS, DATASET, HELPERS
import torch.nn as nn
import torch.distributed as dist

class Tester(object):
    def __init__(self, args, rank=0, world_size=1, mixed_precision=False):
        self.args, self.device_count, self.rank, self.world_size = args, torch.cuda.device_count(), rank, world_size
        self.device = torch.device('cuda:{}'.format(self.rank)) if torch.cuda.is_available() else torch.device('cpu')
        
        self.glb_step = -1
        self.start_epoch = 0

        self.__init_multi_gpu__()
        self.__load_configs__()
        self._create_dataset()
        self._create_model()

    def __load_configs__(self):
        from projects import Config
        self.cfg = Config.fromfile(self.args.cfg_path)

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
        if self.device_count <= 1:
            self.model = nn.DataParallel(model).to(self.device)
        else:
            self.model = nn.parallel.DistributedDataParallel(model, device_ids=[self.rank], output_device=self.rank, broadcast_buffers=False, find_unused_parameters=True).to(self.device)
        
        if self.is_master_node:
            print('loading ' + self.args.chkp_path)
        self.load_chkp(self.args.chkp_path)
            
    def _create_dataset(self):
        dataset = DATASET.build(self.cfg.data, datasets=[self.args.part])
        self.dataset, self.loader = dataset.get(self.args.part).dataset, dataset.get(self.args.part).loader

    def load_chkp(self, sv_path):
        st = torch.load(sv_path, map_location='cpu')
        if 'net' in st:
            HELPERS.get('copy_state_dict')(self.model.module.state_dict(), st['net'])

    def do_test(self, chunk_size=8, apply_mask=True):
        img_density_map_scale=self.cfg.model.get('img_density_map_scale', 1.0)
        scene_density_map_scale=self.cfg.model.get('scene_density_map_scale', 1.0)
        def gather_tensor(tensor):
            if self.world_size > 1:
                tensor_list = [torch.zeros_like(tensor) for _ in range(self.world_size)]
                dist.all_gather(tensor_list, tensor)
                return sum(tensor_list)
            else:
                return tensor
            
        def _chunk_input_(_input, s, e):
            if isinstance(_input, (list, torch.Tensor,)):
                return _input[s:e]
            elif isinstance(_input, (dict, edict,)):
                chnuk_input_dict=edict()
                for (k, v) in _input.items():
                    chnuk_input_dict[k] = _chunk_input_(v, s, e)
                return chnuk_input_dict
            else:
                raise 'unsupported type.'

        def _merge_output_(_output):
            if isinstance(_output, (torch.Tensor,)):
                return _output
            elif isinstance(_output[0], (dict,)):
                output_dict=edict()
                for (k, _) in _output[0].items():
                    output_dict[k] = _merge_output_([o[k] for o in _output])
                return output_dict
            elif isinstance(_output[0], (list,)):
                merged_output = []
                [merged_output.extend(o) for o in _output]
                return merged_output
            elif isinstance(_output[0], (torch.Tensor,)):
                return torch.cat(_output, 0)
            else:
                raise 'unsupported type'
            


        import tqdm
        from easydict import EasyDict as edict
        self.model.eval()
        total_mae_bev, total_mae_image, total_nae_bev, total_nae_image = 0, 0, 0, 0
        total_mae_image_view = [0 for _ in range(self.cfg.n_sel_view)]
        total_nae_image_view = [0 for _ in range(self.cfg.n_sel_view)]
        for input_dict in tqdm.tqdm(DATASET.get('DataPrefetcher')(self.loader).pool(), total=len(self.loader)) if self.rank==0 else DATASET.get('DataPrefetcher')(self.loader).pool():
            bs, n_view = input_dict.input_data.image_set.shape[:2]
            input_dict = edict(input_dict)
            p = self.model(input_dict)

            #get mask
            mask = nn.functional.interpolate(input_dict.input_data.image_masks, p.image_density_map.shape[3:], mode='bilinear')/255.0
            mask[mask>=0.3] = 1.0
            mask[mask< 0.3] = 0

            batch_mae_bev   = torch.tensor(0, device=self.device, dtype=torch.float32)
            batch_mae_image = torch.tensor(0, device=self.device, dtype=torch.float32)
            batch_nae_bev   = torch.tensor(0, device=self.device, dtype=torch.float32)
            batch_nae_image = torch.tensor(0, device=self.device, dtype=torch.float32)
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
                for (view, (p_image, g_image)) in enumerate(zip(p.image_density_map[b], input_dict.labels.density_image[b])):
                    p_image   = p_image.sum().clamp(min=0) / img_density_map_scale
                    g_image   = g_image.sum() / img_density_map_scale
                    mae_image = (p_image-g_image).abs() 
                    nae_image = (mae_image / (g_image+1e-3)).clamp(max=1.0)
                    batch_mae_image = batch_mae_image + mae_image
                    batch_nae_image = batch_nae_image + nae_image

                    #each view
                    total_mae_image_view[view] += gather_tensor(mae_image)
                    total_nae_image_view[view] += gather_tensor(nae_image)

            #gather from other gpus
            total_nae_bev    += gather_tensor(batch_nae_bev)
            total_mae_bev    += gather_tensor(batch_mae_bev)
            total_nae_image  += gather_tensor(batch_nae_image)
            total_mae_image  += gather_tensor(batch_mae_image)

        total_samples = len(self.dataset) * bs
        loss_dict = edict(
            nae_bev=float(total_nae_bev/total_samples),
            mae_bev=float(total_mae_bev/total_samples),
            nae_image=float(total_nae_image/total_samples/n_view),
            mae_image=float(total_mae_image/total_samples/n_view),
            total_mae_image=[float(mae_image/total_samples) for mae_image in total_mae_image_view],
            total_nae_image=[float(nae_image/total_samples) for nae_image in total_nae_image_view],
        )
        if self.is_master_node:
            import json
            # print({k:round(float(v), 4) for (k,v) in loss_dict.items()})
            print(json.dumps(loss_dict, indent=2))
 
def worker(rank, world_size, args, **kargs):
    tester = Tester(args=args, rank=rank, world_size=world_size)
    with torch.no_grad():
        tester.do_test()

def parse_args():
    import argparse
    import random
    parser = argparse.ArgumentParser(description = 'neofusion')
    parser.add_argument('--cfg_path', type=str, default='projects/configs/citystreet/bev/resnet_benchmark.py')
    parser.add_argument('--nccl_port', type=int, default=random.randint(50000, 65535))
    parser.add_argument('--part', type=str, default='val')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deter', type=int, default=0)
    parser.add_argument('--chkp_path', type=str, default='workdirs/CityStreet/bev/resnet/[2023-10-03::11:18:00]/checkpoints/best_mae_bev.pth')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    world_size = torch.cuda.device_count()
    if world_size <= 1:
        worker(rank=0, world_size=1, args=args)
    else:
        import torch.multiprocessing as mp
        mp.spawn(worker, nprocs=world_size, args=(world_size,args,))

if __name__ == '__main__':
    main()
