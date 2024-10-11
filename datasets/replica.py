import torch
import json
import numpy as np
import os
from tqdm import tqdm
import glob

from .ray_utils import get_ray_directions
from .color_utils import read_image

from .base import BaseDataset

class ReplicaDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, load_depth=False, **kwargs):
        super().__init__(root_dir, split, downsample, load_depth)

        self.read_intrinsics()

        if kwargs.get('read_meta', True):
            self.read_meta(split)

    def read_intrinsics(self):
        with open(os.path.join(self.root_dir, 'transforms.json'), 'r') as fp:
            metas = json.load(fp)
        w, h = metas['w'], metas['h']
        w, h = int(w*self.downsample), int(h*self.downsample)
        fx, fy = metas['fl_x'] * self.downsample, metas['fl_y'] * self.downsample
        K = np.float32([[fx, 0, w/2],
                        [0, fy, h/2],
                        [0,  0,   1]])

        self.K = torch.FloatTensor(K)
        self.directions = get_ray_directions(h, w, self.K[:3,:3])
        self.img_wh = (w, h)
        # xyz_min, xyz_max = np.array(metas['aabb'])[0], np.array(metas['aabb'])[1]
        # self.shift = (xyz_max+xyz_min)/2
        # self.bbox = np.array(metas['aabb'])

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

                img = read_image(img_path, self.img_wh)
                self.rays += [img]
        
        self.rays = torch.FloatTensor(np.stack(self.rays)) # (N_images, hw, ?)
        self.poses = np.stack(self.poses)
        
        if split=='train':
            ind = [i for i in range(len(img_paths)) if i%2==0]
            self.poses = self.poses[ind]
            self.rays = self.rays[ind]
            img_paths = [x for i, x in enumerate(img_paths) if i%8!=0]
        elif split=='test':
            ind = [i for i in range(len(img_paths)) if i%2!=0]
            self.poses = self.poses[ind]
            self.rays = self.rays[ind]
            img_paths = [x for i, x in enumerate(img_paths) if i%8==0]
        elif split == 'test_traj':
            poses_path = os.path.join(self.root_dir, 'traj.txt')
            self.poses = np.loadtxt(poses_path).reshape(-1,4,4)[:,:3]

        # self.poses[...,3] -= self.shift
        # self.bbox -= self.shift
            
        self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)
        print(f'Loading {self.poses.shape[0]} {split} images ...')
    
