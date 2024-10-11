import torch
import json
import numpy as np
import os
from tqdm import tqdm
import glob

from .ray_utils import get_ray_directions
from .color_utils import read_image

from .base import BaseDataset
from .geometry import inter_poses

SCANNET_FAR = 2.0


class ScanNetDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

        self.unpad = 24

        self.read_intrinsics()

        if kwargs.get('read_meta', True):
            self.read_meta(split)

    def read_intrinsics(self):
        w, h = int(1296*self.downsample), int(968*self.downsample)
        K = np.loadtxt(os.path.join(self.root_dir, 'intrinsics.txt'), dtype=np.float32)
        K[:2] *= self.downsample

        self.K = torch.FloatTensor(K)
        self.directions = get_ray_directions(h, w, self.K[:3,:3])
        self.img_wh = (w, h)

    def read_meta(self, split):
        self.rays = []
        self.poses = []
        all_img_paths = sorted(glob.glob(os.path.join(self.root_dir, 'images', '*.jpg')))
        all_poses = sorted(glob.glob(os.path.join(self.root_dir, 'poses', '*.txt')))

        img_paths = []
        poses = []
        for img_path, pose in tqdm(zip(all_img_paths, all_poses)):
            c2w = np.loadtxt(pose)[:3]
            if np.isinf(c2w).sum()==0:
                img_paths.append(img_path)
                poses.append(pose)
                self.poses += [c2w]

                img = read_image(img_path, self.img_wh, unpad=self.unpad)
                self.rays += [img]
        
        self.rays = torch.FloatTensor(np.stack(self.rays)) # (N_images, hw, ?)
        self.poses = np.stack(self.poses)
        
        xyz_min = self.poses[...,3].min(0)
        xyz_max = self.poses[...,3].max(0)

        sbbox_scake = (xyz_max-xyz_min).max() + 2 * SCANNET_FAR
        sbbox_shift = (xyz_min+xyz_max)/2

        self.poses[...,3] -= sbbox_shift
        self.poses[...,3] /= sbbox_scake
        
        if split=='train':
            ind = [i for i in range(len(img_paths)) if i%16!=0]
            self.poses = self.poses[ind]
            self.rays = self.rays[ind]
            img_paths = [x for i, x in enumerate(img_paths) if i%16!=0]
        elif split=='test':
            ind = [i for i in range(len(img_paths)) if i%16==0]
            self.poses = self.poses[ind]
            self.rays = self.rays[ind]
            img_paths = [x for i, x in enumerate(img_paths) if i%16==0]
        elif split == 'test_traj':
            self.poses = inter_poses(self.poses, 1000, 20)
            
        self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)
        print(f'Loading {self.poses.shape[0]} {split} images ...')
    
