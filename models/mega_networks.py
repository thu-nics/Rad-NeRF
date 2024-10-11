import torch
from torch import nn
import torch.nn.functional as F
import tinycudann as tcnn
import vren
from einops import rearrange
from .custom_functions import TruncExp
import numpy as np
from kornia.utils.grid import create_meshgrid3d
import math
import copy
import open3d as o3d

from .rendering import NEAR_DISTANCE
from datasets.geometry import get_bbox_from_points


class mega_NGP(nn.Module):
    def __init__(self, scale, rgb_act='Sigmoid', size=2, t=19):
        super().__init__()

        self.rgb_act = rgb_act

        # scene bounding box
        self.scale = scale # scale only controls bounding bbox, not hash grid trainable params
        self.register_buffer('center', torch.zeros(1, 3))
        self.register_buffer('xyz_min', -torch.ones(1, 3)*scale)
        self.register_buffer('xyz_max', torch.ones(1, 3)*scale)
        self.register_buffer('half_size', (self.xyz_max-self.xyz_min)/2)

        # each density grid covers [-2^(k-1), 2^(k-1)]^3 for k in [0, C-1]
        self.cascades = max(1+int(np.ceil(np.log2(2*scale))), 1)
        # self.cascades = 1
        self.grid_size = 128
        self.register_buffer('density_bitfield',
            torch.zeros(self.cascades*self.grid_size**3//8, dtype=torch.uint8))

        # constants
        L = 16; F = 2; log2_T = t; N_min = 16
        b = np.exp(np.log(2048*scale/N_min)/(L-1))
        print(f'GridEncoding: Nmin={N_min} b={b:.5f} F={F} T=2^{log2_T} L={L}')

        self.xyz_encoder = \
            tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "Grid",
	                "type": "Hash",
                    "n_levels": L,
                    "n_features_per_level": F,
                    "log2_hashmap_size": log2_T,
                    "base_resolution": N_min,
                    "per_level_scale": b,
                    "interpolation": "Linear"
                },
            )
        
        self.size = size

        self.geo_net = \
            tcnn.Network(
                n_input_dims=self.xyz_encoder.n_output_dims, 
                n_output_dims=16+1,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                },
            )

        self.dir_encoder = \
            tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "SphericalHarmonics",
                    "degree": 4,
                },
            )

        for i in range(self.size):
            setattr(self, 'rgb_net_{}'.format(i), tcnn.Network(
                n_input_dims=32, n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": self.rgb_act,
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                },
            ))
    
    def density(self, x, return_feat=False):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            return_feat: whether to return intermediate feature

        Outputs:
            sigmas: (N)
        """
        x = (x-self.xyz_min)/(self.xyz_max-self.xyz_min)
        x = x.clip(0.0, 1.0)
        xyz_feature = self.xyz_encoder(x)

        h = self.geo_net(xyz_feature)

        sigmas = TruncExp.apply(h[:, 0])
        if return_feat: return sigmas, h[...,1:]
        return sigmas

    def forward(self, x, d, ind, **kwargs):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            d: (N, 3) directions

        Outputs:
            sigmas: (N)
            rgbs: (N, 3)
        """
        sigmas, h = self.density(x, return_feat=True)
        d = d/torch.norm(d, dim=1, keepdim=True)
        d = self.dir_encoder((d+1)/2)

        rgb_net = getattr(self, 'rgb_net_{}'.format(ind))
        rgbs = rgb_net(torch.cat([d, h], 1))

        return sigmas, rgbs

    @torch.no_grad()
    def get_all_cells(self):
        """
        Get all cells from the density grid.
        
        Outputs:
            cells: list (of length self.cascades) of indices and coords
                   selected at each cascade
        """
        indices = vren.morton3D(self.grid_coords).long()
        cells = [(indices, self.grid_coords)] * self.cascades

        return cells

    @torch.no_grad()
    def sample_uniform_and_occupied_cells(self, M, density_threshold):
        """
        Sample both M uniform and occupied cells (per cascade)
        occupied cells are sample from cells with density > @density_threshold
        
        Outputs:
            cells: list (of length self.cascades) of indices and coords
                   selected at each cascade
        """
        cells = []
        for c in range(self.cascades):
            # uniform cells
            coords1 = torch.randint(self.grid_size, (M, 3), dtype=torch.int32,
                                    device=self.density_grid.device)
            indices1 = vren.morton3D(coords1).long()
            # occupied cells
            indices2 = torch.nonzero(self.density_grid[c]>density_threshold)[:, 0]
            if len(indices2)>0:
                rand_idx = torch.randint(len(indices2), (M,),
                                         device=self.density_grid.device)
                indices2 = indices2[rand_idx]
                coords2 = vren.morton3D_invert(indices2.int())
                cells += [(torch.cat([indices1, indices2]), torch.cat([coords1, coords2]))]
            else:
                cells += [(indices1, coords1)]

        return cells

    @torch.no_grad()
    def update_density_grid(self, density_threshold, warmup=False, decay=0.95, erode=False):
        density_grid_tmp = torch.zeros_like(self.density_grid)
        if warmup: # during the first steps
            cells = self.get_all_cells()
        else:
            cells = self.sample_uniform_and_occupied_cells(self.grid_size**3//4,
                                                           density_threshold)
        # infer sigmas
        for c in range(self.cascades):
            indices, coords = cells[c]
            s = min(2**(c-1), self.scale)
            half_grid_size = s/self.grid_size
            xyzs_w = (coords/(self.grid_size-1)*2-1)*(s-half_grid_size)
            
            # pick random position in the cell by adding noise in [-hgs, hgs]
            xyzs_w += (torch.rand_like(xyzs_w)*2-1) * half_grid_size
            density_grid_tmp[c, indices] = self.density(xyzs_w)
            # import ipdb; ipdb.set_trace()

        if erode:
            # My own logic. decay more the cells that are visible to few cameras
            decay = torch.clamp(decay**(1/self.count_grid), 0.1, 0.95)
        self.density_grid = \
            torch.where(self.density_grid<0,
                        self.density_grid,
                        torch.maximum(self.density_grid*decay, density_grid_tmp))

        mean_density = self.density_grid[self.density_grid>0].mean().item()
        vren.packbits(self.density_grid, min(mean_density, density_threshold),
                      self.density_bitfield)
    
    @torch.no_grad()
    def register_bbox(self, bbox):
        self.xyz_min = torch.from_numpy(bbox[0][None,:]).float().to(self.xyz_min.device)
        self.xyz_max = torch.from_numpy(bbox[1][None,:]).float().to(self.xyz_min.device)
        self.half_size = torch.from_numpy((bbox[1] - bbox[0])/2)[None,:].float().to(self.xyz_min.device)
        self.center = torch.from_numpy(np.mean(bbox, axis=0))[None,:].float().to(self.xyz_min.device)
        self.cascades = max(1+int(np.ceil(np.log2(2*self.half_size.max()))), 1)
        self.density_bitfield = torch.zeros(self.cascades*self.grid_size**3//8, dtype=torch.uint8).to(self.xyz_min.device)
        self.density_grid = torch.zeros(self.cascades, self.grid_size**3)