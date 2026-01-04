import numpy as np
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


def geodesic_loss(gt_pose, pose_est, graph, pr_scale, gt_scale):
    """ Loss function for training network """

    # relative pose
    ii, jj, kk = graph_to_edge_list(graph)
    dP = gt_pose[:,jj] * gt_pose[:,ii].inv()

    # scale the relative poses
    dP = dP.scale(1.0 / gt_scale[:, ii])
    pose_est = pose_est.scale(1.0 / pr_scale[:, ii])

    # pose error
    d = (pose_est * dP.inv()).log()

    if isinstance(pose_est, SE3):
        tau, phi = d.split([3,3], dim=-1)
        geodesic_loss = tau.norm(dim=-1).mean() + phi.norm(dim=-1).mean()
    elif isinstance(pose_est, Sim3):
        tau, phi, sig = d.split([3,3,1], dim=-1)
        geodesic_loss += tau.norm(dim=-1).mean() + phi.norm(dim=-1).mean() + 0.05 * sig.norm(dim=-1).mean()

    dE = Sim3(pose_est * dP.inv()).detach()
    r_err, t_err, s_err = pose_metrics(dE)

    metrics = {
        'rot_error': r_err.mean().item(),
        'tr_error': t_err.mean().item(),
        'bad_rot': (r_err < .1).float().mean().item(),
        'bad_tr': (t_err < .01).float().mean().item(),
    }

    return geodesic_loss, metrics


def flow_loss(gt_pose, disps, poses_est, disps_est, intrinsics, graph, valid):
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

    return flow_loss, metrics