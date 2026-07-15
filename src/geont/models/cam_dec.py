# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn


class CameraDec(nn.Module):
    def __init__(self, dim_in=1536):
        super().__init__()
        output_dim = dim_in
        self.backbone = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
        )
        self.fc_t = nn.Linear(output_dim, 3)
        self.fc_qvec = nn.Linear(output_dim, 4)
        self.fc_s = nn.Linear(output_dim, 2)  # log-variance
        self.fc_log_scale = nn.Linear(output_dim, 1)
        self.fc_scale_conf = nn.Linear(output_dim, 1)

    def forward(self, feat):
        B, N = feat.shape[:2]
        feat = feat.reshape(B * N, -1)
        feat = self.backbone(feat.float())
        out_t = self.fc_t(feat).reshape(B, N, 3)
        out_qvec = self.fc_qvec(feat).reshape(B, N, 4)
        pos_enc = torch.cat([out_t, out_qvec], dim=-1)
        conf = torch.exp(self.fc_s(feat)).reshape(B, N, 2) + 1
        relative_log_scale = self.fc_log_scale(feat).reshape(B, N)
        scale_conf = torch.exp(self.fc_scale_conf(feat)).reshape(B, N) + 1
        return pos_enc, conf, relative_log_scale, scale_conf
