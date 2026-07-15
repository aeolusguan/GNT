import torch
import logging
from dataclasses import dataclass
from math import ceil, floor

from lietorch import SE3, Sim3
from .geometry.losses import geodesic_loss, flow_loss, pose_metrics
from .geometry.graph_utils import graph_to_edge_list
from .geometry.projective_ops import projective_transform


def check_and_fix_inf_nan(input_tensor, loss_name="default", hard_max=100):
    """
    Checks if 'input_tensor' contains inf or nan values and clamps extreme values.

    Args:
        input_tensor (torch.Tensor): The loss tensor to check and fix.
        loss_name (str): Name of the loss (for diagnostic prints).
        hard_max (float, optional): Maximum absolute value allowed. Value outside
                                  [-hard_max, hard_max] will be clamped. If None,
                                  no clamping is performed. Default is 100.
    """
    if input_tensor is None:
        return input_tensor

    # Check for inf/nan values
    has_inf_nan = torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any()
    if has_inf_nan:
        logging.warning(f"Tensor {loss_name} contains inf or nan values. Replacing with zeros.")
        input_tensor = torch.where(
            torch.isnan(input_tensor) | torch.isinf(input_tensor),
            torch.zeros_like(input_tensor),
            input_tensor
        )

    # Apply hard clamping if specified
    if hard_max is not None:
        input_tensor = torch.clamp(input_tensor, min=-hard_max, max=hard_max)

    return input_tensor


def invalid_to_zeros(arr, valid_mask):
    if valid_mask is not None:
        arr = arr.clone()
        arr[~valid_mask] = 0
        nnz = valid_mask.view(len(valid_mask), -1).sum(1)
    else:
        nnz = arr.numel() // len(arr) if len(arr) else 0  # number of point per image
    return arr, nnz


def normalize_depth(depth, valid, eps=1e-8):
    """  
    depth: [B,H,W]
    valid: [B,H,W]
    """
    nan_depth, nnz = invalid_to_zeros(depth, valid)
    scale_factor = nan_depth.flatten(1).sum(dim=1) / (nnz + eps)
    scale_factor = scale_factor.clip(min=1e-8)
    return nan_depth / scale_factor[:, None, None], scale_factor


def camara_loss(pred_pose_enc, gt_pose_enc, loss_type="l1", conf=None):
    """
    Compute translation, rotation for a batch of pose encodings.

    Args:
        pred_pose_enc: (N, D) predicted pose encoding
        gt_pose_enc: (N, D) ground truth pose encoding
        loss_type: "l1" (abs error) or "l2" (euclidean error)
        conf: (N,2) predicted confidence for the 6D pose residual. Only one translational confidence and 
            one rotational confidence per relative pose.
    Returns:
        loss_T: translation loss (mean)
        loss_R: rotation loss (mean)

        NOTE: The VGGT paper uses smooth l1 loss, but we found l1 loss is more stable than smooth l1 and l2 loss.
        So here we use l1 loss.
    """
    pred_q = pred_pose_enc[..., 3:7]
    gt_q = gt_pose_enc[..., 3:7]
    gt_q = torch.where(gt_q[..., 3:4] < 0, -gt_q, gt_q)
    quat_residual = pred_q - gt_q

    if loss_type == "l1":
        # Translation: first 3 dims; Rotation: next 4 (quaternion)
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).abs().sum(dim=-1)
        loss_R = quat_residual.abs().sum(dim=-1)
    elif loss_type == "l2":
        # L2 norm for each component
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).norm(dim=-1)
        loss_R = quat_residual.norm(dim=-1)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    # Check/fix numerical issues (nan/inf) for each loss component
    loss_T = check_and_fix_inf_nan(loss_T, "loss_T")
    loss_R = check_and_fix_inf_nan(loss_R, "loss_R")

    loss_T = loss_T.clamp(max=100)
    if conf is not None:
        loss_T = loss_T * conf[..., 0] - 0.01 * torch.log(conf[..., 0])
        loss_R = loss_R * conf[..., 1] - 0.05 * torch.log(conf[..., 1])

    # Clamp outlier translation loss to prevent instability, then average
    loss_T = loss_T.mean()
    loss_R = loss_R.mean()

    return loss_T, loss_R


@dataclass(eq=False)
class MultitaskLoss(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, predictions, batch) -> torch.Tensor:
        """
        Compute the total multi-task loss.

        Args:
            predictions: Dict containing model predictions for different tasks
            batch: Dict containing ground truth data and masks

        Returns:
            Dict containing individual losses and total objective
        """
        pr_depth = predictions["depth"].clip(max=100, min=1e-3)
        pr_depth_conf = predictions["depth_conf"]
        edge_measurements = predictions["edge_measurements"]
        # Edge measurement layout: 0:3 translation, 3:7 quaternion,
        # 7:9 pose confidence/log variance, 9 relative log scale, 10 scale confidence.
        pr_rel_poses = SE3(edge_measurements[..., :7])
        pr_rel_poses_log_variance = edge_measurements[..., 7:9]
        pr_relative_log_scale = edge_measurements[..., 9]
        pr_scale_confidence = edge_measurements[..., 10]
        gt_depth = batch["depth"]
        # pr_depth/gt_depth/valid_mask: [B,S,H,W].
        valid_mask = torch.logical_and(predictions["valid"], batch["valid"])

        # Flatten frames to compute one valid-pixel mean depth per frame: [B*S,H,W].
        gt_depth_target, gt_scale = normalize_depth(gt_depth.flatten(0, 1), valid_mask.flatten(0, 1))
        gt_scale = gt_scale.view(*gt_depth.shape[:2])  # [B,S]
        gt_depth_target = gt_depth_target.view(*gt_depth.shape)  # [B,S,H,W]

        graph = batch["graph"]
        gt_pose = SE3(batch["poses"]).inv()  # convert poses w2c -> c2w

        intrinsics = batch["intrinsics"]

        cam_loss, geo_metrics = self.compute_camera_loss(
            gt_pose,
            pr_rel_poses,
            graph,
            gt_scale,
            pr_rel_poses_log_variance,
            pr_relative_log_scale,
            pr_scale_confidence,
        )
        cam_loss = check_and_fix_inf_nan(cam_loss, "cam_loss")
        flo_loss, front_flow_loss, flo_metrics = flow_loss(
            gt_pose,
            1.0 / gt_depth,
            pr_rel_poses,
            1.0 / pr_depth,
            intrinsics,
            graph,
            valid=valid_mask,
            flow_predictions=predictions["flow_predictions"],
            args=self.args,
        )
        flo_loss = check_and_fix_inf_nan(flo_loss, "flo_loss")
        depth_conf_loss, depth_grad_loss, depth_reg_loss = self.compute_depth_loss(pr_depth, gt_depth_target, pr_depth_conf, valid_mask)
        depth_loss = depth_grad_loss + depth_conf_loss
        depth_metrics = {'depth_reg': depth_reg_loss.item(), 'depth_grad': depth_grad_loss.item()}

        total_loss = self.args.w_pose * cam_loss + self.args.w_flow * flo_loss + self.args.w_depth * depth_loss + self.args.w_front_flow * front_flow_loss

        return total_loss, geo_metrics, flo_metrics, depth_metrics
    
    def compute_camera_loss(
        self,
        gt_pose: SE3,
        pr_rel_poses: SE3,
        graph,
        gt_scale,
        pr_rel_pose_log_variance,
        pr_relative_log_scale,
        pr_scale_confidence,
    ):
        # relative pose
        ii, jj, kk = graph_to_edge_list(graph)
        dP = gt_pose[:,jj] * gt_pose[:,ii].inv()

        # gt_scale[:, ii]: [B,E], one source depth scale per relative edge.
        # GT relative translation is supervised in the source normalized-depth gauge.
        # Predictions are already expected to live in this gauge.
        dP = dP.scale(1.0 / gt_scale[:, ii])

        loss_T, loss_R = camara_loss(pr_rel_poses.data, dP.data, conf=pr_rel_pose_log_variance)
        gt_relative_log_scale = torch.log(gt_scale[:, jj]) - torch.log(gt_scale[:, ii])
        scale_residual = (pr_relative_log_scale - gt_relative_log_scale).abs()
        scale_residual = check_and_fix_inf_nan(scale_residual, "relative_log_scale_loss")
        scale_loss = scale_residual * pr_scale_confidence - 0.02 * torch.log(pr_scale_confidence)
        scale_loss = check_and_fix_inf_nan(scale_loss, "relative_log_scale_conf_loss").mean()
        cam_loss = loss_T + loss_R + scale_loss

        dE = Sim3(pr_rel_poses * dP.inv()).detach()
        r_err, t_err, s_err = pose_metrics(dE)
        scale_error = scale_residual.detach()

        metrics = {
            'rot_error': r_err.mean().item(),
            'tr_error': t_err.mean().item(),
            'scale_error': scale_error.mean().item(),
            'bad_rot': (r_err < .1).float().mean().item(),
            'bad_tr': (t_err < .01).float().mean().item(),
        }

        return cam_loss, metrics
    
    def compute_depth_loss(self, pr_depth, gt_depth_target, depth_conf, valid_mask, alpha=0.2):
        # Compute L1 loss between predicted and ground truth points
        depth_reg_loss = torch.abs(pr_depth - gt_depth_target)
        depth_reg_loss = depth_reg_loss[valid_mask]

        # Confidence-weighted loss: loss * conf - alpha * log(conf)
        depth_conf_loss = depth_reg_loss * depth_conf[valid_mask] - alpha * torch.log(depth_conf[valid_mask])
        depth_conf_loss = check_and_fix_inf_nan(depth_conf_loss, "depth_conf_loss")

        depth_grad_loss = gradient_loss_multi_scale_wrapper(
            pr_depth.flatten(0, 1).unsqueeze(-1),
            gt_depth_target.flatten(0, 1).unsqueeze(-1),
            valid_mask.flatten(0, 1),
            gradient_loss_fn=gradient_loss,
        )
        depth_grad_loss = check_and_fix_inf_nan(depth_grad_loss, "depth_grad_loss")

        # Process confidence-weighted loss
        if depth_conf_loss.numel() > 0:
            # Filter out outliers using quantile-based thresholding
            if self.args.depth_valid_range > 0:
                depth_conf_loss = filter_by_quantile(depth_conf_loss, self.args.depth_valid_range)
            
            depth_conf_loss = depth_conf_loss.mean()
        else:
            depth_conf_loss = (0.0 * pr_depth).mean()

        # Process regular regression loss
        if depth_reg_loss.numel() > 0:
            # Filter out outliers using quantile-based thresholding
            if self.args.depth_valid_range > 0:
                depth_reg_loss = filter_by_quantile(depth_reg_loss, self.args.depth_valid_range)

            depth_reg_loss = depth_reg_loss.mean()
        else:
            depth_reg_loss = (0.0 * pr_depth).mean()

        return depth_conf_loss, depth_grad_loss, depth_reg_loss


def gradient_loss_multi_scale_wrapper(prediction, target, mask, scales=4, gradient_loss_fn = None, conf=None):
    """
    Multi-scale gradient loss wrapper. Applies gradient loss at multiple scales by subsampling the input.
    This helps capture both fine and coarse spatial structures.

    Args:
        prediction: (B, H, W, C) predicted values
        target: (B, H, W, C) ground truth values
        mask: (B, H, W) valid pixel mask
        scales: Number of scales to use
        gradient_loss_fn: Gradient loss function to apply
        conf: (B, H, W) confidence weights (optional)
    """
    total = 0
    for scale in range(scales):
        step = pow(2, scale)  # Subsample by 2^scale

        total += gradient_loss_fn(
            prediction[:, ::step, ::step],
            target[:, ::step, ::step],
            mask[:, ::step, ::step],
            None if conf is None else conf[:, ::step, ::step]
        )
    
    total = total / scales
    return total


def gradient_loss(prediction, target, mask, conf=None, gamma=1.0, alpha=0.2):
    """   
    Gradient-based loss. Compute the L1 difference between adjacent pixels in x and y directions.

    Args:
        prediction: (B, H, W, C) predicted values
        target: (B, H, W, C) ground truth values
        mask: (B, H, W) valid pixel mask
        conf: (B, H, W) confidence weights (optional)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
    """
    # Expand mask to match prediction channels
    mask = mask[..., None].expand(-1, -1, -1, prediction.shape[-1])
    M = torch.sum(mask, (1, 2, 3))

    # Compute difference between prediction and target
    diff = prediction - target
    diff = torch.mul(mask, diff)

    # Compute gradients in x direction (horizontal)
    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    # Compute gradients in y direction (vertical)
    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    # Clamp gradients to prevent outliers
    grad_x = grad_x.clamp(max=100)
    grad_y = grad_y.clamp(max=100)

    # Apply confidence weighting if provided
    if conf is not None:
        conf = conf[..., None].expand(-1, -1, -1, prediction.shape[-1])
        conf_x = conf[:, :, 1:]
        conf_y = conf[:, 1:, :]

        grad_x = gamma * grad_x * conf_x - alpha * torch.log(conf_x)
        grad_y = gamma * grad_y * conf_y - alpha * torch.log(conf_y)
    
    # Sum gradients and normalize by number of valid pixels
    grad_loss = torch.sum(grad_x, (1, 2, 3)) + torch.sum(grad_y, (1, 2, 3))
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        grad_loss = torch.sum(grad_loss) / divisor

    return grad_loss


def filter_by_quantile(loss_tensor, valid_range, min_elements=1000, hard_max=100):
    """
    Filter loss tensor by keeping only values below a certain quantile threshold.

    This helps remove outliers that could destabilize training.

    Args:
        loss_tensor: Tensor containing loss values
        valid_range: Float between 0 and 1 indicating the quantile threshold
        min_elements: Minimum number of elements required to apply filtering
        hard_max: Maximum allowed value for any individual loss

    Returns:
        Filtered and clamped loss tensor
    """
    if loss_tensor.numel() <= min_elements:
        # Too few elements, just return as-is
        return loss_tensor
    
    # Randomly sample if tensor is too large to avoid memory issues
    if loss_tensor.numel() > 100000000:
        # Flatten and randomly select 1M elements
        indices = torch.randperm(loss_tensor.numel(), device=loss_tensor.device)[:1_000_000]
        loss_tensor = loss_tensor.view(-1)[indices]

    # First clamp individual values to prevent extreme outliers
    loss_tensor = loss_tensor.clamp(max=hard_max)

    # Compute quantile threshold
    quantile_thresh = torch_quantile(loss_tensor.detach(), valid_range)
    quantile_thresh = min(quantile_thresh, hard_max)

    # Apply quantile filtering if enough elements remain
    quantile_mask = loss_tensor < quantile_thresh
    if quantile_mask.sum() > min_elements:
        return loss_tensor[quantile_mask]
    return loss_tensor


def torch_quantile(
    input,
    q,
    dim = None,
    keepdim: bool = False,
    *,
    interpolation: str = "nearest",
    out: torch.Tensor = None,
) -> torch.Tensor:
    """Better torch.quantile for one SCALAR  quantile.
    
    Using torch.kthvalue. Better than torch.quantile because:
        - No 2**24 input size limit (pytorch/issues/67592),
        - Much faster, at least on big input sizes.

    Arguments:
        input (torch.Tensor): See torch.quantile.
        q (float): See torch.quantile. Supports only scalar input
            currently.
        dim (int | None): See torch.quantile.
        keepdim (bool): See torch.quantile. Supports only False
            currently.
        interpolation: {"nearest", "lower", "higher"}
            See torch.quantile.
        out (torch.Tensor | None): See torch.quantile. Supports only
            None currently.
    """
    # https://github.com/pytorch/pytorch/issues/64947
    # Sanitization: q
    try:
        q = float(q)
        assert 0 <= q <= 1
    except Exception:
        raise ValueError(f"Only scalar input 0<=q<=1 is currently supported (got {q})!")
    
    # Handle dim=None case
    if dim_was_none := dim is None:
        dim = 0
        input = input.reshape((-1,) + (1,) * (input.ndim - 1))

    # Set interpolation method
    if interpolation == "nearest":
        inter = round
    elif interpolation == "lower":
        inter = floor
    elif interpolation == "higher":
        inter = ceil
    else:
        raise ValueError(
            "Supported interpolations currently are {'nearest', 'lower', 'higher'} "
            f"(got '{interpolation}')!"
        )
    
    # Validate out parameter
    if out is not None:
        raise ValueError(f"Only None value is currently supported for out (got {out})!")
    
    # Compute k-th value
    k = inter(q * (input.shape[dim] - 1)) + 1
    out = torch.kthvalue(input, k, dim, keepdim=True, out=out)[0]

    # Handle keepdim and dim=None cases
    if keepdim:
        return out
    if dim_was_none:
        return out.squeeze()
    else:
        return out.squeeze(dim)
    
    return out
