import torch
import glob
import numpy as np
import os
from tqdm import tqdm

from .ray_utils import get_ray_directions
from .color_utils import read_image

from .base import BaseDataset


class Mill19Dataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

        self.read_intrinsics()

        if kwargs.get('read_meta', True):
            self.read_meta(split)

    def read_intrinsics(self):

        exam = torch.load(os.path.join(self.root_dir, './train/metadata/000001.pt'))
        w, h = int(exam['W']*self.downsample), int(exam['H']*self.downsample)
        fx, fy = exam['intrinsics'][0].item()*self.downsample, exam['intrinsics'][1].item()*self.downsample
        K = np.float32([[fx, 0, w/2],
                        [0, fy, h/2],
                        [0,  0,   1]])


        self.K = torch.FloatTensor(K)
        self.directions = get_ray_directions(h, w, self.K)
        self.img_wh = (w, h)

        if 'building' in self.root_dir:
            self.ray_altitude_range = [8,50]
        elif 'rubble' in self.root_dir:
            self.ray_altitude_range = [11,38]
        self.origin_drb = torch.load(os.path.join(self.root_dir, 'coordinates.pt'))['origin_drb']
        self.pose_scale_factor = torch.load(os.path.join(self.root_dir, 'coordinates.pt'))['pose_scale_factor']

    def read_meta(self, split):
        self.rays = []
        self.poses = []


        train_img_paths = sorted(glob.glob(os.path.join(self.root_dir, 'train', 'rgbs/*')))
        train_poses = sorted(glob.glob(os.path.join(self.root_dir, 'train', 'metadata/*')))
        test_img_paths = sorted(glob.glob(os.path.join(self.root_dir, 'train', 'rgbs/*')))
        test_poses = sorted(glob.glob(os.path.join(self.root_dir, 'train', 'metadata/*')))

        all_poses = train_poses + test_poses
        self.all_poses = []
        for pose in tqdm(all_poses):
            c2w = torch.load(pose)['c2w']
            self.all_poses += [c2w]
        self.all_poses = torch.stack(self.all_poses)

        self.all_poses[...,3] = self.all_poses[...,3]*self.pose_scale_factor + self.origin_drb
        self.all_poses[...,3][...,0] += self.ray_altitude_range[1]

        self.scale = torch.norm(self.all_poses[..., 3], dim=-1).min()
        # import ipdb; ipdb.set_trace()

        if split == 'train': 
            img_paths = train_img_paths
            poses = train_poses
        elif split == 'test': # test set for real scenes
            img_paths = test_img_paths
            poses = test_poses
        else: 
            raise ValueError(f'{split} split not recognized!')

        print(f'Loading {len(img_paths)} {split} images ...')
        for img_path, pose in tqdm(zip(img_paths, poses)):

            c2w = torch.load(pose)['c2w']
            c2w[:,3] = c2w[:,3]*self.pose_scale_factor + self.origin_drb
            c2w[:,3][0] += self.ray_altitude_range[1]
            c2w[:, 3] /= self.scale

            self.poses += [c2w]
            img = read_image(img_path, self.img_wh)
            self.rays += [img]

        self.rays = torch.FloatTensor(np.stack(self.rays)) # (N_images, hw, ?)]
        self.poses = torch.stack(self.poses) # (N_images, 3, 4)