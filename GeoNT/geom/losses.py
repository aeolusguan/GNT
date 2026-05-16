import math
import numpy as np
import torch
import torch.nn.functional as F
from lietorch import SO3, SE3, Sim3
from .graph_utils import graph_to_edge_list
from .projective_ops import projective_transform, projective_transform_v2


def pose_metrics(dE):
    """ Translation/Rotation/Scaling metrics from Sim3 """
    t, q, s = dE.data.split([3, 4, 1], -1)
    ang = SO3(q).log().norm(dim=-1)

    # convert radians to degrees
    r_err = (180 / np.pi) * ang
    t_err = t.norm(dim=-1)
    s_err = (s - 1.0).abs()
    return r_err, t_err, s_err


def geodesic_loss(gt_pose, pose_est, graph, gt_scale, pr_scale, conf):
    """ Loss function for training network """

    # relative pose
    ii, jj, kk = graph_to_edge_list(graph)
    dP = gt_pose[:,jj] * gt_pose[:,ii].inv()

    # scale the relative poses
    dP = dP.scale(1.0 / gt_scale[:, ii])
    pose_est = pose_est.scale(1.0 / pr_scale[:, ii])

    # pose error
    d = (pose_est * dP.inv()).log()

    tau, phi = d.split([3,3], dim=-1)
    geodesic_loss = tau.norm(dim=-1) * conf[..., 0] - 0.01 * torch.log(conf[..., 0]) + \
        phi.norm(dim=-1) * conf[..., 1] - 0.01 * torch.log(conf[..., 1])
    
    geodesic_loss = geodesic_loss.mean()

    dE = Sim3(pose_est * dP.inv()).detach()
    r_err, t_err, s_err = pose_metrics(dE)

    metrics = {
        'rot_error': r_err.mean().item(),
        'tr_error': t_err.mean().item(),
        'bad_rot': (r_err < .1).float().mean().item(),
        'bad_tr': (t_err < .01).float().mean().item(),
    }

    return geodesic_loss, metrics


def flow_loss(gt_pose, disps, poses_est, disps_est, intrinsics, graph, valid, flow_predictions=None, args=None):
    """ optical flow loss """

    N = gt_pose.shape[1]

    ii, jj, kk = graph_to_edge_list(graph)
    coords0, val0 = projective_transform(gt_pose, disps, intrinsics, ii, jj)
    val0 = val0 * valid[:, ii].float().unsqueeze(dim=-1)

    coords1, val1 = projective_transform_v2(poses_est, disps_est, intrinsics, ii, jj)
    v = (val0 * val1).squeeze(dim=-1)
    epe = v * (coords1 - coords0).norm(dim=-1)
    flow_loss = epe.mean()

    epe = epe.reshape(-1)[v.reshape(-1) > 0.5]
    metrics = {
        'f_error': epe.mean().item(),
        '1px': (epe<1.0).float().mean().item(),
    }

    if flow_predictions is not None:
        # compute flow loss for frontend flow
        ht, wd = flow_predictions['flow'][-1].shape[2:4]
        nf_loss = []
        y, x = torch.meshgrid(
            torch.arange(ht, device=disps.device, dtype=torch.float),
            torch.arange(wd, device=disps.device, dtype=torch.float),
            indexing="ij",
        )
        flow_gt = coords0[0] - torch.stack([x, y], dim=-1)[None]
        flow_gt = flow_gt.permute(0, 3, 1, 2)  # [N, 2, H, W]
        flows = flow_predictions['flow']
        infos = flow_predictions['info']
        prob_up = flow_predictions['prob_up']
        n_predictions = len(flows)

        front_flow_loss = 0.0
        for i, (flow, info) in enumerate(zip(flows, infos)):
            i_weight = args.gamma ** (n_predictions - i - 1)

            raw_b = info[:, 2:]
            log_b = torch.zeros_like(raw_b)
            weight = info[:, :2]
            # Large b Component
            log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=args.var_max)
            # Small b Component
            log_b[:, 1] = torch.clamp(raw_b[:, 1], min=args.var_min, max=0)
            # term2: [N, 2, m, H, W]
            term2 = ((flow_gt - flow).abs().unsqueeze(2)) * (torch.exp(-log_b).unsqueeze(1))
            # term1: [N, m, H, W]
            term1 = weight - math.log(2) - log_b
            nf_loss = torch.logsumexp(weight, dim=1, keepdim=True) - torch.logsumexp(term1.unsqueeze(1) - term2, dim=2)
            final_mask = (~torch.isnan(nf_loss.detach())) & (~torch.isinf(nf_loss.detach())) & (val0[0, :, None].squeeze(-1) > 0.5)

            front_flow_loss += i_weight * ((final_mask * nf_loss).sum() / final_mask.sum())
        
        # Use confidence to weight initialization loss
        flow_gt_ = torch.clamp(flow_gt / 8.0, min=-16, max=16)
        n_bins = prob_up.shape[1] // 2
        idx_bins = torch.linspace(-16, 16, n_bins, device=flow_gt_.device, dtype=flow_gt_.dtype).view(1, n_bins, 1, 1)
        label_x = F.softmax(-torch.abs(flow_gt_[:, :1] - idx_bins), dim=1)
        label_y = F.softmax(-torch.abs(flow_gt_[:, 1:2] - idx_bins), dim=1)
        kl_loss_x = -(torch.log(torch.clamp(F.softmax(prob_up[:, :n_bins], dim=1), min=1e-6)) * label_x).sum(dim=1)
        kl_loss_y = -(torch.log(torch.clamp(F.softmax(prob_up[:, n_bins:], dim=1), min=1e-6)) * label_y).sum(dim=1)
        kl_loss = kl_loss_x + kl_loss_y
        final_mask = (~torch.isnan(kl_loss.detach())) & (~torch.isinf(kl_loss.detach())) & (val0[0,].squeeze(-1) > 0.5)
        info = (torch.exp(-log_b) * torch.softmax(weight, dim=1)).sum(dim=1).detach()
        front_flow_loss += 0.5 * (kl_loss * final_mask * info).sum() / final_mask.sum()
    else:
        front_flow_loss = None

    return flow_loss, front_flow_loss, metrics