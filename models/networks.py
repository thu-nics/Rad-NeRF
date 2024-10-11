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

from .rendering import NEAR_DISTANCE
from datasets.geometry import get_bbox_from_points


class NGP(nn.Module):
    def __init__(self, scale, rgb_act='Sigmoid', t=19):
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

        self.geo_net = \
            tcnn.Network(
                n_input_dims=self.xyz_encoder.n_output_dims, 
                n_output_dims=16+1,
                network_config={
                    "otype": "CUTLASSMLP",
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

        self.rgb_net = \
            tcnn.Network(
                n_input_dims=32, n_output_dims=3,
                network_config={
                    "otype": "CUTLASSMLP",
                    "activation": "ReLU",
                    "output_activation": self.rgb_act,
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                },
            )

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

    def forward(self, x, d, **kwargs):
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

        rgbs = self.rgb_net(torch.cat([d, h], 1))

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


class MNGP(nn.Module):
    def __init__(self, scale, rgb_act='Sigmoid', size=2, t=19):
        super().__init__()

        self.rgb_act = rgb_act

        # scene bounding box
        self.scale = scale # scale only controls bounding bbox, not hash grid trainable params
        self.register_buffer('center', torch.zeros(1, 3))
        self.register_buffer('xyz_min', -torch.ones(1, 3)*scale)
        self.register_buffer('xyz_max', torch.ones(1, 3)*scale)
        self.register_buffer('half_size', (self.xyz_max-self.xyz_min)/2)

        self.size = size
        # constants
        L = 16; F = 2; log2_T = t; N_min = 16
        b = np.exp(np.log(2048*scale/N_min)/(L-1))
        print(f'GridEncoding: Nmin={N_min} b={b:.5f} F={F} T=2^{log2_T} L={L}')

        # import ipdb; ipdb.set_trace()
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

        self.dir_encoder = \
            tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "SphericalHarmonics",
                    "degree": 4,
                },
            )
        
        # each density grid covers [-2^(k-1), 2^(k-1)]^3 for k in [0, C-1]
        self.cascades = max(1+int(np.ceil(np.log2(2*scale))), 1)
        self.grid_size = 128
        self.register_buffer('grid_coords',
                create_meshgrid3d(self.grid_size, self.grid_size, self.grid_size, False, dtype=torch.int32).reshape(-1, 3))
        for i in range(size):
            self.register_buffer('density_bitfield_{}'.format(i),
                torch.zeros(self.cascades*self.grid_size**3//8, dtype=torch.uint8))

            self.register_buffer('density_grid_{}'.format(i),
                torch.zeros(self.cascades, self.grid_size**3))
            setattr(self, 'geo_net_{}'.format(i), tcnn.Network(
                n_input_dims=self.xyz_encoder.n_output_dims, 
                n_output_dims=16+1,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                },
            ))
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

    def density(self, x, ind, return_feat=False):
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

        geo_net = getattr(self, 'geo_net_{}'.format(ind))
        h = geo_net(xyz_feature)

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
        sigmas, h = self.density(x, ind, return_feat=True)
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
    def sample_uniform_and_occupied_cells(self, M, density_threshold, ind):
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
            density_grid = getattr(self, 'density_grid_{}'.format(ind))
            coords1 = torch.randint(self.grid_size, (M, 3), dtype=torch.int32,
                                    device=density_grid.device)
            indices1 = vren.morton3D(coords1).long()
            # occupied cells
            indices2 = torch.nonzero(density_grid[c]>density_threshold)[:, 0]
            if len(indices2)>0:
                rand_idx = torch.randint(len(indices2), (M,),
                                         device=density_grid.device)
                indices2 = indices2[rand_idx]
                coords2 = vren.morton3D_invert(indices2.int())
                cells += [(torch.cat([indices1, indices2]), torch.cat([coords1, coords2]))]
            else:
                cells += [(indices1, coords1)]

        return cells

    @torch.no_grad()
    def update_density_grid(self, density_threshold, warmup=False, decay=0.95, erode=False):
        for i in range(self.size):
            density_grid = getattr(self, 'density_grid_{}'.format(i))
            density_bitfield = getattr(self, 'density_bitfield_{}'.format(i))
            density_grid_tmp = torch.zeros_like(density_grid)
            if warmup: # during the first steps
                cells = self.get_all_cells()
            else:
                cells = self.sample_uniform_and_occupied_cells(self.grid_size**3//4,
                                                            density_threshold, i)
            # infer sigmas
            for c in range(self.cascades):
                indices, coords = cells[c]
                s = min(2**(c-1), self.scale)
                half_grid_size = s/self.grid_size
                xyzs_w = (coords/(self.grid_size-1)*2-1)*(s-half_grid_size)
                
                # pick random position in the cell by adding noise in [-hgs, hgs]
                xyzs_w += (torch.rand_like(xyzs_w)*2-1) * half_grid_size
                density_grid_tmp[c, indices] = self.density(xyzs_w, i)

            if erode:
                # My own logic. decay more the cells that are visible to few cameras
                decay = torch.clamp(decay**(1/self.count_grid), 0.1, 0.95)
            
            density_grid = \
                torch.where(density_grid<0,
                            density_grid,
                            torch.maximum(density_grid*decay, density_grid_tmp))

            mean_density = density_grid[density_grid>0].mean().item()
            vren.packbits(density_grid, min(mean_density, density_threshold),
                        density_bitfield)
            setattr(self, 'density_grid_{}'.format(i), density_grid)
            setattr(self, 'density_bitfield_{}'.format(i), density_bitfield)
    

    @torch.no_grad()
    def register_bbox(self, bbox):
        self.xyz_min = torch.from_numpy(bbox[0][None,:]).float().to(self.xyz_min.device)
        self.xyz_max = torch.from_numpy(bbox[1][None,:]).float().to(self.xyz_min.device)
        self.half_size = torch.from_numpy((bbox[1] - bbox[0])/2)[None,:].float().to(self.xyz_min.device)
        self.center = torch.from_numpy(np.mean(bbox, axis=0))[None,:].float().to(self.xyz_min.device)
        self.cascades = max(1+int(np.ceil(np.log2(2*self.half_size.max()))), 1)
        for i in range(self.size):
            setattr(self, 'density_bitfield_{}'.format(i), torch.zeros(self.cascades*self.grid_size**3//8, dtype=torch.uint8).to(self.xyz_min.device))
            setattr(self, 'density_grid_{}'.format(i), torch.zeros(self.cascades, self.grid_size**3))


class unshared_MNGP(nn.Module):
    def __init__(self, scale, rgb_act='Sigmoid', size=2, t=19):
        super().__init__()

        self.rgb_act = rgb_act

        # scene bounding box
        self.scale = scale # scale only controls bounding bbox, not hash grid trainable params
        self.register_buffer('center', torch.zeros(1, 3))
        self.register_buffer('xyz_min', -torch.ones(1, 3)*scale)
        self.register_buffer('xyz_max', torch.ones(1, 3)*scale)
        self.register_buffer('half_size', (self.xyz_max-self.xyz_min)/2)

        self.size = size
        # constants
        L = 16; F = 2; log2_T = t; N_min = 16
        b = np.exp(np.log(2048*scale/N_min)/(L-1))
        print(f'GridEncoding: Nmin={N_min} b={b:.5f} F={F} T=2^{log2_T} L={L}')
        
        # each density grid covers [-2^(k-1), 2^(k-1)]^3 for k in [0, C-1]
        self.cascades = max(1+int(np.ceil(np.log2(2*scale))), 1)
        self.grid_size = 128
        self.register_buffer('grid_coords',
                create_meshgrid3d(self.grid_size, self.grid_size, self.grid_size, False, dtype=torch.int32).reshape(-1, 3))
        for i in range(size):
            self.register_buffer('density_bitfield_{}'.format(i),
                torch.zeros(self.cascades*self.grid_size**3//8, dtype=torch.uint8))

            self.register_buffer('density_grid_{}'.format(i),
                torch.zeros(self.cascades, self.grid_size**3))
            setattr(self, 'xyz_encoder_{}'.format(i), tcnn.Encoding(
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
            ))
            setattr(self, 'dir_encoder_{}'.format(i), tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "SphericalHarmonics",
                    "degree": 4,
                },
            ))
            setattr(self, 'geo_net_{}'.format(i), tcnn.Network(
                n_input_dims=self.xyz_encoder_0.n_output_dims, 
                n_output_dims=16+1,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                },
            ))
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

    def density(self, x, ind, return_feat=False):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            return_feat: whether to return intermediate feature

        Outputs:
            sigmas: (N)
        """
        x = (x-self.xyz_min)/(self.xyz_max-self.xyz_min)
        x = x.clip(0.0, 1.0)
        xyz_encoder = getattr(self, 'xyz_encoder_{}'.format(ind))
        xyz_feature = xyz_encoder(x)

        geo_net = getattr(self, 'geo_net_{}'.format(ind))
        h = geo_net(xyz_feature)

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
        sigmas, h = self.density(x, ind, return_feat=True)
        d = d/torch.norm(d, dim=1, keepdim=True)
        dir_encoder = getattr(self, 'dir_encoder_{}'.format(ind))
        d = dir_encoder((d+1)/2)

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
    def sample_uniform_and_occupied_cells(self, M, density_threshold, ind):
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
            density_grid = getattr(self, 'density_grid_{}'.format(ind))
            coords1 = torch.randint(self.grid_size, (M, 3), dtype=torch.int32,
                                    device=density_grid.device)
            indices1 = vren.morton3D(coords1).long()
            # occupied cells
            indices2 = torch.nonzero(density_grid[c]>density_threshold)[:, 0]
            if len(indices2)>0:
                rand_idx = torch.randint(len(indices2), (M,),
                                         device=density_grid.device)
                indices2 = indices2[rand_idx]
                coords2 = vren.morton3D_invert(indices2.int())
                cells += [(torch.cat([indices1, indices2]), torch.cat([coords1, coords2]))]
            else:
                cells += [(indices1, coords1)]

        return cells

    @torch.no_grad()
    def update_density_grid(self, density_threshold, warmup=False, decay=0.95, erode=False):
        for i in range(self.size):
            density_grid = getattr(self, 'density_grid_{}'.format(i))
            density_bitfield = getattr(self, 'density_bitfield_{}'.format(i))
            density_grid_tmp = torch.zeros_like(density_grid)
            if warmup: # during the first steps
                cells = self.get_all_cells()
            else:
                cells = self.sample_uniform_and_occupied_cells(self.grid_size**3//4,
                                                            density_threshold, i)
            # infer sigmas
            for c in range(self.cascades):
                indices, coords = cells[c]
                s = min(2**(c-1), self.scale)
                half_grid_size = s/self.grid_size
                xyzs_w = (coords/(self.grid_size-1)*2-1)*(s-half_grid_size)
                
                # pick random position in the cell by adding noise in [-hgs, hgs]
                xyzs_w += (torch.rand_like(xyzs_w)*2-1) * half_grid_size
                density_grid_tmp[c, indices] = self.density(xyzs_w, i)

            if erode:
                # My own logic. decay more the cells that are visible to few cameras
                decay = torch.clamp(decay**(1/self.count_grid), 0.1, 0.95)
            
            density_grid = \
                torch.where(density_grid<0,
                            density_grid,
                            torch.maximum(density_grid*decay, density_grid_tmp))

            mean_density = density_grid[density_grid>0].mean().item()
            vren.packbits(density_grid, min(mean_density, density_threshold),
                        density_bitfield)
            setattr(self, 'density_grid_{}'.format(i), density_grid)
            setattr(self, 'density_bitfield_{}'.format(i), density_bitfield)
    

    @torch.no_grad()
    def register_bbox(self, bbox):
        self.xyz_min = torch.from_numpy(bbox[0][None,:]).float().to(self.xyz_min.device)
        self.xyz_max = torch.from_numpy(bbox[1][None,:]).float().to(self.xyz_min.device)
        self.half_size = torch.from_numpy((bbox[1] - bbox[0])/2)[None,:].float().to(self.xyz_min.device)
        self.center = torch.from_numpy(np.mean(bbox, axis=0))[None,:].float().to(self.xyz_min.device)
        self.cascades = max(1+int(np.ceil(np.log2(2*self.half_size.max()))), 1)
        for i in range(self.size):
            setattr(self, 'density_bitfield_{}'.format(i), torch.zeros(self.cascades*self.grid_size**3//8, dtype=torch.uint8).to(self.xyz_min.device))
            setattr(self, 'density_grid_{}'.format(i), torch.zeros(self.cascades, self.grid_size**3))



class PMOE_NGP(nn.Module):
    def __init__(self, scale, rgb_act='Sigmoid', size=3, num_topk=2, t=19):
        super().__init__()

        self.rgb_act = rgb_act

        # scene bounding box
        self.scale = scale
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
        self.register_buffer('grid_coords',
            create_meshgrid3d(self.grid_size, self.grid_size, self.grid_size, False, dtype=torch.int32).reshape(-1, 3))

        # constants
        self.size = size
        self.num_topk = num_topk
        L = 16; F = 2; log2_T = t; N_min = 16
        b = np.exp(np.log(2048*scale/N_min)/(L-1))
        print(f'GridEncoding: Nmin={N_min} b={b:.5f} F={F} T=2^{log2_T} L={L}')

        self.gate_net = Point_Gate(out_dim=self.size)

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

        self.dir_encoder = \
            tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "SphericalHarmonics",
                    "degree": 4,
                },
            )
        
        # each density grid covers [-2^(k-1), 2^(k-1)]^3 for k in [0, C-1]
        for i in range(size):
            setattr(self, 'geo_net_{}'.format(i), tcnn.Network(
                n_input_dims=self.xyz_encoder.n_output_dims, 
                n_output_dims=16+1,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                },
            ))
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
    
    def gate_forward(self, x, warmup=False):
        x = (x-self.xyz_min)/(self.xyz_max-self.xyz_min)
        x = x.clip(0.0, 1.0)
        gating_code, gating_importance, gating_indices = self.gate_net(x, warmup=warmup)
        gate_results = {}
        gate_results['code'] = gating_code
        gate_results['importance'] = gating_importance
        gate_results['indice'] = gating_indices

        return gate_results

    def density(self, x, gating_code, return_feat=False):
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
        h = torch.zeros((x.shape[0], self.geo_net_0.n_output_dims)).to(x.device)
        for i in range(self.size):
            geo_net = getattr(self, 'geo_net_{}'.format(i))
            h += gating_code[:,i][:,None] * geo_net(xyz_feature)
        sigmas = TruncExp.apply(h[:, 0])
        geo_features = h[...,1:]

        if return_feat: return sigmas, geo_features
        return sigmas

    def forward(self, x, d, **kwargs):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            d: (N, 3) directions

        Outputs:
            sigmas: (N)
            rgbs: (N, 3)
        """
        gate_results = self.gate_forward(x, warmup=kwargs['warmup'])
        gating_code = gate_results['code']
        
        d = d/torch.norm(d, dim=1, keepdim=True)
        d = self.dir_encoder((d+1)/2)

        sigmas, h = self.density(x, gating_code, return_feat=True)
        rgbs = torch.zeros_like(x).to(x.device)
        for i in range(self.size):
            rgb_net = getattr(self, 'rgb_net_{}'.format(i))
            rgbs += rgb_net(torch.cat([d,h],1)) * gating_code[:,i][:,None]

        return sigmas, rgbs, gate_results

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
            gate_results = self.gate_forward(xyzs_w, warmup=warmup)
            sigmas = self.density(xyzs_w, gate_results['code'])
            density_grid_tmp[c, indices] = sigmas

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


class Embedder:
    def __init__(self, in_dim, multires, **kwargs):
        self.num_topkwargs = {
                'include_input' : True,
                'input_dims' : in_dim,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
        self.create_embedding_fn()

        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.num_topkwargs['input_dims']
        out_dim = 0
        if self.num_topkwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.num_topkwargs['max_freq_log2']
        N_freqs = self.num_topkwargs['num_freqs']
        
        if self.num_topkwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.num_topkwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim


        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class MLP(nn.Module):
    """base mlp
    """
    def __init__(self, num_layers, hidden_dim, in_dims, out_dims, bias=True):
        super().__init__()
        self.D = num_layers
        self.W = hidden_dim
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.activation = nn.ReLU()
        self.ln = nn.LayerNorm(hidden_dim)

        curr_in_dim = self.in_dims

        # mlp layers
        layers = []
        for i in range(self.D):
            out_dim = self.out_dims if i == self.D - 1 else self.W
            layer = nn.Linear(curr_in_dim, out_dim, bias=bias)

            torch.nn.init.kaiming_normal_(layer.weight)
            if bias:
                torch.nn.init.constant_(layer.bias, 0)

            layers.append(layer)
            curr_in_dim = out_dim

        self.mlp = nn.ModuleList(layers)
        

    def forward(self, x) -> torch.Tensor:
        h = x
        for layer_id in range(self.D):
            h = self.mlp[layer_id](h)
            if layer_id < self.D - 2:
                h = self.activation(h)
            elif layer_id == self.D - 2:
                h = self.ln(h)  

        return h


# class Ray_Gate(nn.Module):
#     def __init__(self, out_dim, type='ray', ):
#         super().__init__()
#         self.out_dim = out_dim
#         self.type = type
#         self.encoder = tcnn.Network(
#                             n_input_dims=6,
#                             n_output_dims=out_dim,
#                             network_config={
#                                 "otype": "FullyFusedMLP",
#                                 "activation": "ReLU",
#                                 "output_activation": "None",
#                                 "n_neurons": 64,
#                                 "n_hidden_layers": 4,
#                             }
#                         )

#         self.noisy_module = tcnn.Network(
#                             n_input_dims=6,
#                             n_output_dims=out_dim,
#                             network_config={
#                                 "otype": "FullyFusedMLP",
#                                 "activation": "ReLU",
#                                 "output_activation": "None",
#                                 "n_neurons": 64,
#                                 "n_hidden_layers": 2,
#                             }
#                         )
#         self.softplus = nn.Softplus()
#         self.softmax = nn.Softmax(1)
#         self.noisy_gating = True
#         self.num_topk = 1
    
    # def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
    #     """Noisy top-k gating.
    #       See paper: https://arxiv.org/abs/1701.06538.
    #       Args:
    #         x: input Tensor with shape [batch_size, input_size]
    #         train: a boolean - we only add noise at training time.
    #         noise_epsilon: a float
    #       Returns:
    #         gates: a Tensor with shape [batch_size, num_experts]
    #         load: a Tensor with shape [num_experts]
    #     """
    #     clean_logits = self.encoder(x)
        
    #     if self.noisy_gating and train:
    #         raw_noise_stddev = self.noisy_module(x)
    #         noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
    #         noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
    #         logits = noisy_logits
    #     else:
    #         logits = clean_logits

    #     # calculate topk + 1 that will be needed for the noisy gates
    #     top_logits, top_indices = logits.topk(min(self.num_topk + 1, self.out_dim), dim=1)
    #     top_k_logits = top_logits[:, :self.num_topk]
    #     top_k_indices = top_indices[:, :self.num_topk]
    #     top_k_gates = self.softmax(top_k_logits)

    #     zeros = torch.zeros_like(logits, requires_grad=True)
    #     gates = zeros.scatter(1, top_k_indices, top_k_gates.to(logits.dtype))
    #     if self.noisy_gating and self.num_topk < self.out_dim and train:
    #         load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
    #     else:
    #         load = self._gates_to_load(gates)
    #     return gates, load, top_k_indices
    
    # def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
    #     """Helper function to NoisyTopKGating.
    #     Computes the probability that value is in top k, given different random noise.
    #     This gives us a way of backpropagating from a loss that balances the number
    #     of times each expert is in the top k experts per example.
    #     In the case of no noise, pass in None for noise_stddev, and the result will
    #     not be differentiable.
    #     Args:
    #     clean_values: a `Tensor` of shape [batch, n].
    #     noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
    #       normally distributed noise with standard deviation noise_stddev.
    #     noise_stddev: a `Tensor` of shape [batch, n], or None
    #     noisy_top_values: a `Tensor` of shape [batch, m].
    #        "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
    #     Returns:
    #     a `Tensor` of shape [batch, n].
    #     """
    #     batch = clean_values.size(0) # N
    #     m = noisy_top_values.size(1) # num_topk+1
    #     top_values_flat = noisy_top_values.flatten() # (N*(num_topk+1),)

    #     threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.num_topk
    #     threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1) # (N,1)
    #     is_in = torch.gt(noisy_values, threshold_if_in)
    #     threshold_positions_if_out = threshold_positions_if_in - 1
    #     threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1) # (N,1)
    #     # is each value currently in the top k.
    #     normal = torch.distributions.normal.Normal(
    #         loc=torch.tensor([0.0], device=clean_values.device),
    #         scale=torch.tensor([1.0], device=clean_values.device)
    #     )
    #     prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
    #     prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
    #     prob = torch.where(is_in, prob_if_in, prob_if_out)
    #     return prob

    # def _gates_to_load(self, gates):
    #     """Compute the true load per expert, given the gates.
    #     The load is the number of examples for which the corresponding gate is >0.
    #     Args:
    #     gates: a `Tensor` of shape [batch_size, n]
    #     Returns:
    #     a float32 `Tensor` of shape [n]
    #     """
    #     return (gates > 0).sum(0)
    
    # def forward(self, x, warmup=False):
    #     # x = self.encoder(x)
    #     # gate = F.softmax(x,1)
    #     # top_k_indices = None
    #     # importance = gate.sum(0)
    #     gate, importance, top_k_indices = self.noisy_top_k_gating(x, self.training)

    #     return gate, importance, top_k_indices
    
    # def freeze_dict(self):
    #     for name, param in self.named_parameters():
    #         param.requires_grad_(False)

class Ray_Gate(nn.Module):
    def __init__(self, out_dim, type='ray'):
        super().__init__()
        self.out_dim = out_dim
        self.type = type
        self.encoder = tcnn.Network(
                            n_input_dims=6,
                            n_output_dims=out_dim,
                            network_config={
                                "otype": "FullyFusedMLP",
                                "activation": "ReLU",
                                "output_activation": "None",
                                "n_neurons": 64,
                                "n_hidden_layers": 4,
                            }
                        )
    
    def forward(self, x, warmup=False):
        x = self.encoder(x)
        gate = F.softmax(x,1)
        top_k_indices = None
        importance = gate.sum(0)

        return gate, importance, top_k_indices
    
    def freeze_dict(self):
        for name, param in self.named_parameters():
            param.requires_grad_(False)

            
class NGP_zoo(nn.Module):
    def __init__(self, scale, rgb_act='Sigmoid', size=10, t=19):
        super().__init__()

        self.rgb_act = rgb_act
        self.scale = scale
        self.size = size
        self.register_buffer('center', torch.zeros(1, 3))
        self.register_buffer('xyz_min', -torch.ones(1, 3)*scale)
        self.register_buffer('xyz_max', torch.ones(1, 3)*scale)
        self.register_buffer('half_size', (self.xyz_max-self.xyz_min)/2)

        for i in range(size):
            setattr(self, 'model_{}'.format(i), NGP(scale=scale, rgb_act=rgb_act, t=t))
        
        for i in range(size):
            model = getattr(self, 'model_{}'.format(i))
            G = model.grid_size
            model.register_buffer('density_grid',
                torch.zeros(model.cascades, G**3))
            model.register_buffer('grid_coords',
                create_meshgrid3d(G, G, G, False, dtype=torch.int32).reshape(-1, 3))
    
    @torch.no_grad()
    def update_density_grid(self, density_threshold, warmup=False, decay=0.95, erode=False):
        for i in range(self.size):
            model = getattr(self, 'model_{}'.format(i))
            density_grid_tmp = torch.zeros_like(model.density_grid)
            if warmup: # during the first steps
                cells = model.get_all_cells()
            else:
                cells = model.sample_uniform_and_occupied_cells(model.grid_size**3//4,
                                                        density_threshold)
            # infer sigmas
            for c in range(model.cascades):
                indices, coords = cells[c]
                s = min(2**(c-1), model.scale)
                half_grid_size = s/model.grid_size
                xyzs_w = (coords/(model.grid_size-1)*2-1)*(s-half_grid_size)
                
                # pick random position in the cell by adding noise in [-hgs, hgs]
                xyzs_w += (torch.rand_like(xyzs_w)*2-1) * half_grid_size
                density_grid_tmp[c, indices] = model.density(xyzs_w)

            model.density_grid = \
                torch.where(model.density_grid<0,
                            model.density_grid,
                            torch.maximum(model.density_grid*decay, density_grid_tmp))
                
            mean_density = model.density_grid[model.density_grid>0].mean().item()
            vren.packbits(model.density_grid, min(mean_density, density_threshold),
                        model.density_bitfield)
    
    @torch.no_grad()
    def register_bbox(self, bbox):
        for i in range(self.size):
            model = getattr(self, 'model_{}'.format(i))
            model.xyz_min = torch.from_numpy(bbox[0][None,:]).float()
            model.xyz_max = torch.from_numpy(bbox[1][None,:]).float()
            model.half_size = torch.from_numpy((bbox[1] - bbox[0])/2)[None,:].float()
            model.center = torch.from_numpy(np.mean(bbox, axis=0))[None,:].float()
            model.cascades = max(1+int(np.ceil(np.log2(2*model.half_size.max()))), 1)
            model.density_bitfield = torch.zeros(model.cascades*model.grid_size**3//8, dtype=torch.uint8)
            model.density_grid = torch.zeros(model.cascades, model.grid_size**3)