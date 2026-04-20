# This file includes code from SEA-RAFT (https://github.com/princeton-vl/SEA-RAFT)
# Copyright (c) 2024, Princeton Vision & Learning Lab
# Licensed under the BSD 3-Clause License

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# exclude extremely large displacements
MAX_FLOW = 200

def sequence_loss(output, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """
    n_predictions = len(output["flow"])
    flow_loss = 0.0
    # exclude invalid pixels and extremely large dispalcements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        loss_i = output['nf'][i]
        final_mask = (~torch.isnan(loss_i.detach())) & (~torch.isinf(loss_i.detach())) & valid[:, None]
        flow_loss += i_weight * ((final_mask * loss_i).sum() / final_mask.sum())

    flow_epe = torch.sum((output['final'] - flow_gt)**2, dim=1).sqrt()[valid].sum() / (valid.sum() + 1e-6)

    return flow_loss, flow_epe

def init_loss(output, flow_gt, valid, max_flow=MAX_FLOW):
    init = output["init"]
    # exclude invalid pixels and extremely large displacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)
    loss = torch.abs(init - flow_gt).sum(dim=1)

    init_epe = torch.sum((init - flow_gt)**2, dim=1).sqrt()[valid].sum() / (valid.sum() + 1e-6)

    return (loss * valid).sum() / (valid.sum() + 1e-6), init_epe