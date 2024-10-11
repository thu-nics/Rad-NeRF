import torch
import glob
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import math
import json
from xml.dom.minidom import parse

from .ray_utils import get_ray_directions
from .color_utils import read_image

from .base import BaseDataset
SCANNET_FAR = 2.0

class EyefulDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

        self.read_intrinsics()

        if kwargs.get('read_meta', True):
            self.read_meta(split, **kwargs)

    def read_intrinsics(self):
        with open(os.path.join(self.root_dir, "cameras.json"), 'r') as f:
            meta = json.load(f)['KRT']
        
        origin_width = meta[0]['width']
        w = 684
        h = 1024
        downsample = origin_width / w

        K = np.array(meta[0]['K']).T
        K[:2] /= downsample

        K[:2] *= self.downsample
        w, h = int(w*self.downsample), int(h*self.downsample)
        self.K = torch.FloatTensor(K)
        self.directions = get_ray_directions(h, w, self.K)
        self.img_wh = (w, h)

    def read_meta(self, split, **kwargs):
        self.rays = []
        self.poses = []

        with open(os.path.join(self.root_dir, "splits.json"), 'r') as f:
            splits = json.load(f)
        if split=='train':
            splits = splits['train']
        else:
            splits = splits['test']

        with open(os.path.join(self.root_dir, "cameras.json"), 'r') as f:
            meta = json.load(f)['KRT']

        for i, frame in enumerate(meta):
            if frame['cameraId'] in splits:
                w2c = np.array(frame['T']).T
                c2w = np.linalg.inv(w2c)[:3]
                
                # c2w[:, 0] *= -1
                # c2w[:, 2] *= -1
                image_name = frame['cameraId']
                image_path = os.path.join(self.root_dir, 'images', '{}.jpg'.format(image_name))

                self.poses += [c2w]

                img = read_image(image_path, self.img_wh)
                self.rays += [img]

        self.rays = torch.FloatTensor(np.stack(self.rays)) # (N_images, hw, ?)
        self.poses = np.stack(self.poses)

        self.poses = torch.FloatTensor(self.poses)