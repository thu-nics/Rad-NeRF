import torch
import numpy as np
import os
import glob
from tqdm import tqdm
import math
from pathlib import Path
import torchvision.models as models

from .ray_utils import *
from .color_utils import read_image
from .colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary, read_model
from .geometry import _process_points3d, get_bbox_from_points, filter_outliers_by_boxplot, inter_poses

from .base import BaseDataset


class ColmapDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

        self.read_intrinsics()

        if kwargs.get('read_meta', True):
            self.read_meta(split, **kwargs)

    def read_intrinsics(self):
        # Step 1: read and scale intrinsics (same for all images)
        camdata = read_cameras_binary(os.path.join(self.root_dir, 'sparse/0/cameras.bin'))
        h = int(camdata[1].height*self.downsample)
        w = int(camdata[1].width*self.downsample)
        self.img_wh = (w, h)

        if camdata[1].model in ['SIMPLE_RADIAL', 'SIMPLE_PINHOLE']:
            fx = fy = camdata[1].params[0]*self.downsample
            cx = camdata[1].params[1]*self.downsample
            cy = camdata[1].params[2]*self.downsample
        elif camdata[1].model in ['PINHOLE', 'OPENCV']:
            fx = camdata[1].params[0]*self.downsample
            fy = camdata[1].params[1]*self.downsample
            cx = camdata[1].params[2]*self.downsample
            cy = camdata[1].params[3]*self.downsample
        else:
            raise ValueError(f"Please parse the intrinsics for camera model {camdata[1].model}!")
        self.fx = fx
        self.fy = fy
        self.K = torch.FloatTensor([[fx, 0, cx, 0],
                                    [0, fy, cy, 0],
                                    [0,  0,  1, 0],
                                    [0,  0,  0, 1]])
        self.directions = get_ray_directions(h, w, self.K[:3, :3])

    def read_meta(self, split, **kwargs):
        # Step 2: correct poses
        # read extrinsics (of successfully reconstructed images)
        imdata = read_images_binary(os.path.join(self.root_dir, 'sparse/0/images.bin'))
        img_names = [imdata[k].name for k in imdata]
        if '360_v2' in self.root_dir and self.downsample<1: # mipnerf360 data
            folder = f'images_{int(1/self.downsample)}'
        else:
            folder = 'images'
        # read successfully reconstructed images and ignore others
        img_paths = [os.path.join(self.root_dir, folder, name)
                     for name in img_names]
        img_paths = []
        self.exist_ind = np.zeros((len(img_names),))
        for i in range(len(img_names)):
            name = img_names[i]
            img_path = os.path.join(self.root_dir, folder, name)
            if os.path.exists(img_path):
                self.exist_ind[i] = 1
                img_paths.append(img_path)

        w2c_mats = []
        bottom = np.array([[0, 0, 0, 1.]])
        for k in imdata:
            im = imdata[k]
            R = im.qvec2rotmat(); t = im.tvec.reshape(3, 1)
            w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
        w2c_mats = np.stack(w2c_mats, 0)
        poses = np.linalg.inv(w2c_mats)[:, :3] # (N_images, 3, 4) cam2world matrices

        pts3d = read_points3d_binary(os.path.join(self.root_dir, 'sparse/0/points3D.bin'))
        self.bds, self.vis_arr = self.cal_bds(poses, pts3d, imdata)

        pts3d = np.array([pts3d[k].xyz for k in pts3d]) # (N, 3)
        self.poses, self.pts3d = center_poses(poses, pts3d)
        
        self.scale = np.linalg.norm(self.poses[..., 3], axis=-1).min()
        self.poses[..., 3] /= self.scale
        self.pts3d /= self.scale

        effect_point_ind = self.vis_arr[:,self.exist_ind==1].sum(-1)
        effect_points = self.pts3d[:,:3][effect_point_ind>=1]
        self.bbox = get_bbox_from_points(effect_points)

        self.rays = []
        if split == 'test_traj': # use precomputed test poses
            if '360_v2' in self.root_dir:
                self.poses = create_spheric_poses(1.2, self.poses[:, 1, 3].mean())
            elif 'free' in self.root_dir:
                self.poses = inter_poses(self.poses, 200, 10)
            self.poses = torch.FloatTensor(self.poses)
            return 

        # use every 8th image as test set
        if split=='train':
            img_paths = [x for i, x in enumerate(img_paths) if i%8!=0]
            self.train_img_list = img_paths
            self.total_poses = self.poses
            self.poses = np.array([x for i, x in enumerate(self.poses[self.exist_ind==1]) if i%8!=0])
            # TODO: add few-shot setting
            # import ipdb; ipdb.set_trace()
            if kwargs['num_view']>0:
                index = np.random.choice(np.arange(len(img_paths)), kwargs['num_view'], replace=False)
                img_paths = [img_paths[i]for i in index]
                self.poses = self.poses[index]
                self.train_img_list = img_paths
                # import ipdb; ipdb.set_trace()
        elif split=='test':
            img_paths = [x for i, x in enumerate(img_paths) if i%8==0]
            self.test_img_list = img_paths
            self.total_poses = self.poses
            self.poses = np.array([x for i, x in enumerate(self.poses[self.exist_ind==1]) if i%8==0])
        
        print(f'Loading {len(img_paths)} {split} images ...')
        for img_path in tqdm(img_paths):
            buf = [] # buffer for ray attributes: rgb, etc

            img = read_image(img_path, self.img_wh, blend_a=False)
            img = torch.FloatTensor(img)
            buf += [img]
            img = img.reshape(self.img_wh[1], self.img_wh[0],-1).permute(2,0,1)

            self.rays += [torch.cat(buf, 1)]

        self.rays = torch.stack(self.rays) # (N_images, hw, ?)
        self.poses = torch.FloatTensor(self.poses) # (N_images, 3, 4)
    
    def cal_bds(self, poses, pts3d, imdata):
        pts_arr = []
        vis_arr = []
        id_list = list(imdata.keys())
        for k in pts3d:
            pts_arr.append(pts3d[k].xyz)
            cams = [0] * poses.shape[0]
            for ind in pts3d[k].image_ids:
                act_ind = id_list.index(ind)
                cams[act_ind-1] = 1
            vis_arr.append(cams)

        pts_arr = np.array(pts_arr)
        vis_arr = np.array(vis_arr)
        print( 'Points', pts_arr.shape, 'Visibility', vis_arr.shape )

        poses = poses.transpose(1,2,0)
        poses = np.concatenate([poses[:, 1:2, :], poses[:, 0:1, :], -poses[:, 2:3, :], poses[:, 3:4, :]], 1)
        zvals = np.sum(-(pts_arr[:, np.newaxis, :].transpose([2,0,1]) - poses[:, 3:4, :]) * poses[:, 2:3, :], 0)
        valid_z = zvals[vis_arr==1]
        print( 'Depth stats', valid_z.min(), valid_z.max(), valid_z.mean())
        
        bds = []
        img_names = [imdata[k].name for k in imdata]
        valid_mask = torch.ones(len(img_names),)
        for i in range(len(img_names)):
            vis = vis_arr[:, i]
            zs = zvals[:, i]
            zs = zs[vis==1]
            close_depth, inf_depth = np.percentile(zs, .5), np.percentile(zs, 99.5)
            if close_depth > 0 and inf_depth > 0:
                bds.append(np.array([close_depth, inf_depth]))
            else:
                valid_mask[i]=0
                bds.append(np.array([1, 100]))

        bds = np.array(bds)
        return bds, vis_arr