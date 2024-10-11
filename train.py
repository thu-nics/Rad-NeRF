import torch
from torch import nn
from opt import get_opts
import os
import glob
import imageio
import numpy as np
import cv2
from einops import rearrange
import tinycudann as tcnn

# data
from torch.utils.data import DataLoader
from datasets import dataset_dict
from datasets.ray_utils import axisangle_to_R, get_rays
from datasets.geometry import _process_points3d, get_bbox_from_points, filter_outliers_by_boxplot, normalize_points

# models
from kornia.utils.grid import create_meshgrid3d
from models.networks import NGP, NGP_zoo, Ray_Gate
from models.rendering import render, moe_render, MAX_SAMPLES

# optimizer, losses
from apex.optimizers import FusedAdam
from torch.optim.lr_scheduler import CosineAnnealingLR
from losses import NeRFLoss

# metrics
from torchmetrics import (
    PeakSignalNoiseRatio, 
    StructuralSimilarityIndexMeasure
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# pytorch-lightning
# from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.utilities.distributed import all_gather_ddp_if_available
from pytorch_lightning.profilers import SimpleProfiler

from utils.util import slim_ckpt, load_ckpt, init_global_logger, get_global_logger

import warnings; warnings.filterwarnings("ignore")


def depth2img(depth):
    depth = (depth-depth.min())/(depth.max()-depth.min())
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return depth_img


class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.warmup_steps = self.hparams.warmup_steps
        self.update_interval = 16

        self.loss = NeRFLoss()
        self.train_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1)
        if self.hparams.eval_lpips:
            self.val_lpips = LearnedPerceptualImagePatchSimilarity('vgg')
            for p in self.val_lpips.net.parameters():
                p.requires_grad = False

        rgb_act = 'Sigmoid'
        if hparams.moe_training:
            self.model = NGP_zoo(scale=self.hparams.scale, 
                                 rgb_act=rgb_act, 
                                 size=self.hparams.model_zoo_size, 
                                 t=self.hparams.hash_table_size)
            self.gating_net = Ray_Gate(out_dim=self.hparams.model_zoo_size, 
                                       num_topk=self.hparams.num_topk, 
                                       overlap_ratio=self.hparams.overlap_ratio)
        else:
            self.model = NGP(scale=self.hparams.scale, rgb_act=rgb_act, t=self.hparams.hash_table_size)
            G = self.model.grid_size
            self.model.register_buffer('density_grid',
                torch.zeros(self.model.cascades, G**3))
            self.model.register_buffer('grid_coords',
                create_meshgrid3d(G, G, G, False, dtype=torch.int32).reshape(-1, 3))
        
        self.global_logger = init_global_logger(f'logs/{hparams.dataset_name}/{hparams.scene_name}/{hparams.exp_name}/log.txt')
        self.validation_step_outputs = []
        

    def forward(self, batch, split, extra_data):
        if split=='train':
            poses = self.poses[batch['img_idxs']]
            directions = self.directions[batch['pix_idxs']]
        else:
            poses = batch['pose']
            directions = self.directions

        if self.hparams.optimize_ext:
            dR = axisangle_to_R(self.dR[batch['img_idxs']])
            poses[..., :3] = dR @ poses[..., :3]
            poses[..., 3] += self.dT[batch['img_idxs']]

        rays_o, rays_d = get_rays(directions, poses)
        kwargs = {'test_time': split!='train',
                  'random_bg': self.hparams.random_bg,
                  'moe_training': self.hparams.moe_training,
                  'warmup': self.global_step<self.warmup_steps}
        if self.hparams.scale > 0.5:
            kwargs['exp_step_factor'] = 1/256
        
        if self.hparams.moe_training:
            return moe_render(self.model, self.gating_net, rays_o, rays_d, **kwargs)
        else:
            return render(self.model, rays_o, rays_d, **kwargs)

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_type]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'downsample': self.hparams.downsample,
                  'load_depth': self.hparams.depth_loss_w > 0,
                  'num_view': self.hparams.num_view}
        self.train_dataset = dataset(split=self.hparams.split, **kwargs)
        self.train_dataset.batch_size = self.hparams.batch_size

        # update model size range
        # if self.hparams.dataset_type == 'colmap':
        #     self.model.register_bbox(self.train_dataset.bbox)
        # import ipdb; ipdb.set_trace()
        
        self.test_dataset = dataset(split='test', **kwargs)

        self.global_logger.info(f'traindataset size={len(self.train_dataset)}')

    def configure_optimizers(self):
        # define additional parameters
        self.register_buffer('directions', self.train_dataset.directions.to(self.device))
        self.register_buffer('poses', self.train_dataset.poses.to(self.device))

        if self.hparams.optimize_ext:
            N = len(self.train_dataset.poses)
            self.register_parameter('dR',
                nn.Parameter(torch.zeros(N, 3, device=self.device)))
            self.register_parameter('dT',
                nn.Parameter(torch.zeros(N, 3, device=self.device)))

        load_ckpt(self.model, self.hparams.weight_path)

        net_params = []
        for n, p in self.named_parameters():
            if n not in ['dR', 'dT']: net_params += [p]

        opts = []
        self.net_opt = FusedAdam(net_params, self.hparams.lr, eps=1e-15)
        opts += [self.net_opt]
        if self.hparams.optimize_ext:
            opts += [FusedAdam([self.dR, self.dT], 1e-8)] # learning rate is hard-coded

        eps = self.hparams.lr / 30
        net_sch = CosineAnnealingLR(self.net_opt,
                                    self.hparams.num_epochs,
                                    eta_min=eps)

        return opts, [net_sch]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          num_workers=0,
                          persistent_workers=False,
                          batch_size=None,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset,
                          num_workers=8,
                          batch_size=None,
                          pin_memory=True)

    def on_train_start(self):
        pass

    def training_step(self, batch, batch_nb, *args):
        warmup=self.global_step<self.warmup_steps
        if self.global_step%self.update_interval == 0:
            self.model.update_density_grid(0.01*MAX_SAMPLES/3**0.5,
                                           warmup=warmup,
                                           erode=False)                     
                       
        extra_data = {}
        batch['rgb'] = batch['rays'][:,:3]
        batch['grad'] = batch['rays'][:,3:]

        results = self(batch, split='train', extra_data=extra_data)

        loss_d = self.loss(results, batch,
                           lambda_opacity=self.hparams.opacity_loss_w)

        loss = sum(lo.mean() for lo in loss_d.values())

        with torch.no_grad():
            self.train_psnr(results['rgb'], batch['rgb'])
        self.log('lr', self.net_opt.param_groups[0]['lr'])
        self.log('train/loss', loss)
        self.log('train/psnr', self.train_psnr, True)
        
        return loss
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        pass
    
    def on_train_epoch_end(self):
        pass

    def on_validation_start(self):
        torch.cuda.empty_cache()
        if not self.hparams.no_save_test:
            self.val_dir = f'results/{self.hparams.dataset_name}/{self.hparams.scene_name}/{self.hparams.exp_name}'
            os.makedirs(self.val_dir, exist_ok=True)

    def validation_step(self, batch, batch_nb):
        extra_data = {}
        rgb_gt = batch['rgb']
        results = self(batch, split='test', extra_data=extra_data)

        logs = {}
        # compute each metric per image
        rgb_results = results['rgb']
        self.val_psnr(rgb_results, rgb_gt)
        logs['psnr'] = self.val_psnr.compute()
        self.val_psnr.reset()

        w, h = self.train_dataset.img_wh
        rgb_pred = rearrange(rgb_results, '(h w) c -> 1 c h w', h=h)
        rgb_gt = rearrange(rgb_gt, '(h w) c -> 1 c h w', h=h)
        self.val_ssim(rgb_pred, rgb_gt)
        logs['ssim'] = self.val_ssim.compute()
        self.val_ssim.reset()
        if self.hparams.eval_lpips:
            self.val_lpips(torch.clip(rgb_pred*2-1, -1, 1),
                           torch.clip(rgb_gt*2-1, -1, 1))
            logs['lpips'] = self.val_lpips.compute()
            self.val_lpips.reset()

        if not self.hparams.no_save_test: # save test image to disk
            idx = batch['img_idxs']
            rgb_pred = rearrange(rgb_results.cpu().numpy(), '(h w) c -> h w c', h=h)
            rgb_pred = (rgb_pred*255).astype(np.uint8)
            if self.hparams.moe_training:
                depth = torch.sum(results['depth']*results['gating_code'], 1).cpu().numpy()
                depth_vis = depth2img(rearrange(depth, '(h w) -> h w', h=h))
            else:  
                depth_vis = depth2img(rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h))
            imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}epoch{self.current_epoch}.png'), rgb_pred)
            imageio.imsave(os.path.join(self.val_dir, f'{idx:03d}epoch{self.current_epoch}_d.png'), depth_vis)

        self.validation_step_outputs.append(logs)

        return logs

    def on_validation_epoch_end(self):
        psnrs = torch.stack([x['psnr'] for x in self.validation_step_outputs])
        # mean_psnr = all_gather_ddp_if_available(psnrs).mean()
        mean_psnr = psnrs.mean()
        self.log('test/psnr', mean_psnr, True)
        self.global_logger.info('test/psnr={}'.format(mean_psnr))

        ssims = torch.stack([x['ssim'] for x in self.validation_step_outputs])
        # mean_ssim = all_gather_ddp_if_available(ssims).mean()
        mean_ssim = ssims.mean()
        self.log('test/ssim', mean_ssim)
        self.global_logger.info('test/ssim={}'.format(mean_ssim))

        if self.hparams.eval_lpips:
            lpipss = torch.stack([x['lpips'] for x in self.validation_step_outputs])
            # mean_lpips = all_gather_ddp_if_available(lpipss).mean()
            mean_lpips = lpipss.mean()
            self.log('test/lpips_vgg', mean_lpips)
            self.global_logger.info('test/lpips={}'.format(mean_lpips))

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


if __name__ == '__main__':
    hparams = get_opts()
    if hparams.val_only and (not hparams.ckpt_path):
        raise ValueError('You need to provide a @ckpt_path for validation!')
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(hparams.gpu_id)

    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.dataset_name}/{hparams.scene_name}/{hparams.exp_name}',
                              filename='{epoch:d}',
                              save_weights_only=True,
                              every_n_epochs=hparams.num_epochs,
                              save_on_train_epoch_end=True,
                              save_top_k=-1)
    callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1)]

    logger = TensorBoardLogger(save_dir=f"logs/{hparams.dataset_name}/{hparams.scene_name}",
                               name=hparams.exp_name,
                               default_hp_metric=False)

    profiler = SimpleProfiler(dirpath=f'logs/{hparams.dataset_name}/{hparams.scene_name}/{hparams.exp_name}', filename='profile')
    trainer = Trainer(max_epochs=hparams.num_epochs,
                      check_val_every_n_epoch=hparams.num_epochs,
                      callbacks=callbacks,
                      logger=logger,
                      enable_model_summary=False,
                      accelerator='gpu',
                      profiler=profiler,
                    #   strategy=None,
                      num_sanity_val_steps=-1 if hparams.val_only else 0,
                      precision=16)

    system = NeRFSystem(hparams)
    # 正向传播时：开启自动求导的异常侦测
    # torch.autograd.set_detect_anomaly(True)
    trainer.fit(system, ckpt_path=hparams.ckpt_path)

    if not hparams.val_only: # save slimmed ckpt for the last epoch
        ckpt_ = \
            slim_ckpt(f'ckpts/{hparams.dataset_name}/{hparams.scene_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}.ckpt',
                      save_poses=hparams.optimize_ext)
        torch.save(ckpt_, f'ckpts/{hparams.dataset_name}/{hparams.scene_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}_slim.ckpt')

    if (not hparams.no_save_test) and \
       hparams.dataset_type=='nsvf' and \
       'Synthetic' in hparams.root_dir: # save video
        imgs = sorted(glob.glob(os.path.join(system.val_dir, '*.png')))
        imageio.mimsave(os.path.join(system.val_dir, 'rgb.mp4'),
                        [imageio.imread(img) for img in imgs[::2]],
                        fps=30, macro_block_size=1)
        imageio.mimsave(os.path.join(system.val_dir, 'depth.mp4'),
                        [imageio.imread(img) for img in imgs[1::2]],
                        fps=30, macro_block_size=1)
