import torch
import time
import os
import numpy as np
from metrics import psnr
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import dataset_dict
from datasets.ray_utils import get_rays
from train import depth2img
import imageio
import cv2

from opt import get_opts
from kornia.utils.grid import create_meshgrid3d
from models.networks import NGP, NGP_zoo, MNGP, Ray_Gate
from utils.util import slim_ckpt, load_ckpt
# from models.rendering import render, moe_render, MAX_SAMPLES
from models.ml_rendering import ml_render, MAX_SAMPLES
from models.rendering import render

# load params
hparams = get_opts()

os.environ['CUDA_VISIBLE_DEVICES'] = str(hparams.gpu_id)

# load dataset
dataset = dataset_dict[hparams.dataset_type]
kwargs = {'root_dir': hparams.root_dir,
          'downsample': hparams.downsample}

if 'Synthetic' in hparams.root_dir:
    dataset = dataset(split='test', **kwargs)
else:
    dataset = dataset(split='test_traj', **kwargs)

# load model
if hparams.moe_training:
    model = MNGP(scale=hparams.scale, 
                      rgb_act='Sigmoid', 
                      size=hparams.model_zoo_size, 
                      t=hparams.hash_table_size)
    if hparams.dataset_type == 'colmap':
        model.register_bbox(dataset.bbox)
    model = model.cuda()
    gating_net = Ray_Gate(out_dim=hparams.model_zoo_size, type=hparams.gate_type).cuda()
    load_ckpt(model, hparams.ckpt_path, 'model')
    load_ckpt(gating_net, hparams.ckpt_path, 'gating_net')
else:
    model = NGP(scale=hparams.scale, rgb_act='Sigmoid', t=hparams.hash_table_size)
    G = model.grid_size
    model.register_buffer('density_grid', torch.zeros(model.cascades, G**3))
    model.register_buffer('grid_coords', create_meshgrid3d(G, G, G, False, dtype=torch.int32).reshape(-1, 3))
    if hparams.dataset_type == 'colmap':
        model.register_bbox(dataset.bbox)
    model = model.cuda()
    load_ckpt(model, hparams.ckpt_path, 'model')

imgs = []; depths = []
psnrs = []; psnrs_0 = []; psnrs_1 = []; psnrs_2 = []; psnrs_3 = []; psnrs_fuse = []
kwargs = {'test_time': True,
          'random_bg': hparams.random_bg,
          'moe_training': hparams.moe_training,
          'warmup': False}
gc_imgs_0 = []; gc_imgs_1 = []
if hparams.scale > 0.5:
    kwargs['exp_step_factor'] = 1/256
# os.makedirs(f'render_results/{hparams.dataset_name}/{hparams.scene_name}/{hparams.exp_name}/', exist_ok=True)
for img_idx in tqdm(range(len(dataset))):
    rays_o, rays_d = get_rays(dataset.directions.cuda(), dataset[img_idx]['pose'].cuda())
    imgs_d = get_rays(torch.mean(dataset.directions,0,keepdim=True), dataset.poses)[1]
    rays_data = torch.cat((rays_o, rays_d), 1)

    if hparams.moe_training:
        results = ml_render(model, gating_net, rays_o, rays_d, imgs_d, **kwargs)
    else:
        results = render(model, rays_o, rays_d, **kwargs)
    
    import ipdb; ipdb.set_trace()
    
    # imageio.imwrite(f'render_results/{hparams.dataset_name}/{hparams.scene_name}/{hparams.exp_name}/{img_idx:03d}.png', pred)
    # imageio.imwrite(f'render_results/{hparams.dataset_name}/{hparams.scene_name}/{hparams.exp_name}/{img_idx:03d}_d.png', depth_)