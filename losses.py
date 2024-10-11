import torch
from torch import nn
import vren


class DistortionLoss(torch.autograd.Function):
    """
    Distortion loss proposed in Mip-NeRF 360 (https://arxiv.org/pdf/2111.12077.pdf)
    Implementation is based on DVGO-v2 (https://arxiv.org/pdf/2206.05085.pdf)

    Inputs:
        ws: (N) sample point weights
        deltas: (N) considered as intervals
        ts: (N) considered as midpoints
        rays_a: (N_rays, 3) ray_idx, start_idx, N_samples
                meaning each entry corresponds to the @ray_idx th ray,
                whose samples are [start_idx:start_idx+N_samples]

    Outputs:
        loss: (N_rays)
    """
    @staticmethod
    def forward(ctx, ws, deltas, ts, rays_a):
        loss, ws_inclusive_scan, wts_inclusive_scan = \
            vren.distortion_loss_fw(ws, deltas, ts, rays_a)
        ctx.save_for_backward(ws_inclusive_scan, wts_inclusive_scan,
                              ws, deltas, ts, rays_a)
        return loss

    @staticmethod
    def backward(ctx, dL_dloss):
        (ws_inclusive_scan, wts_inclusive_scan,
        ws, deltas, ts, rays_a) = ctx.saved_tensors
        dL_dws = vren.distortion_loss_bw(dL_dloss, ws_inclusive_scan,
                                         wts_inclusive_scan,
                                         ws, deltas, ts, rays_a)
        return dL_dws, None, None, None


class NeRFLoss(nn.Module):
    def __init__(self, lambda_opacity=1e-3):
        super().__init__()

    def forward(self, results, target,
                      lambda_opacity=1e-3,
                      lambda_distortion=0,
                      lambda_disp=0,
                      lambda_cv_importance=0,
                      lambda_depth_mutual=0):
        loss = {}

        # rgb MSE Loss
        loss['rgb'] = (results['rgb']-target['rgb'])**2

        o = results['opacity']+1e-10
        # encourage opacity to be either 0 or 1 to avoid floater
        loss['opacity'] = lambda_opacity*(-o*torch.log(o))

        if lambda_disp > 0:
            loss['disp'] = lambda_disp * results['disp']**2

        if lambda_distortion > 0:
            loss['distortion'] = 0
            for i in range(results['gating_code'].shape[-1]):
                loss['distortion'] += lambda_distortion * \
                    (DistortionLoss.apply(results['ws_{}'.format(i)], results['deltas_{}'.format(i)],
                                        results['ts_{}'.format(i)], results['rays_a_{}'.format(i)])).mean()
        
        if lambda_cv_importance > 0 and results['gating_code'].shape[-1]>1:
            cv_squared_importance = results['gating_importance'].float().var() / (results['gating_importance'].float().mean()**2 + 1e-10)
            loss['cv_importance'] = lambda_cv_importance * cv_squared_importance
            
        if lambda_depth_mutual > 0 and results['gating_code'].shape[-1]>1:
            loss['depth_mutual'] = lambda_depth_mutual*((results['depth'] - torch.sum(results['depth']*results['gating_code'], 1, keepdim=True).detach())**2)
            
        return loss
