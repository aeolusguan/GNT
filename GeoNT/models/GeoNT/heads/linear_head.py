import torch
import torch.nn as nn


class LinearDepth(nn.Module):
    """
    Linear head for depth + confidence
    """
    def __init__(
        self, 
        patch_size, 
        dec_embed_dim, 
        activation: str = "inv_log",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.activation = activation

        self.proj = nn.Sequential(
            nn.Linear(dec_embed_dim, dec_embed_dim),
            nn.ReLU(),
            nn.Linear(dec_embed_dim, dec_embed_dim),
            nn.ReLU(),
            nn.Linear(dec_embed_dim, self.patch_size**2),
        )

        self.gate = nn.Sequential(
            nn.Linear(dec_embed_dim, dec_embed_dim),
            nn.GELU(),
            nn.Linear(dec_embed_dim, 1),
        )

    def forward(self, aggregated_tokens_list, img_shape):
        B, S, N, C = aggregated_tokens_list[0][0].shape
        H, W = img_shape
        if not self.training:  # inference stage, only use the last layer's output
            aggregated_tokens_list = aggregated_tokens_list[-1:]
        depths = []
        for aggregated_tokens in aggregated_tokens_list:
            gates = torch.softmax(self.gate(aggregated_tokens[0]), dim=1)
            feat = (aggregated_tokens[0] * gates).sum(dim=1)  # B,N,C
            feat = self.proj(feat)  # B,N,patch_size**2
            feat = feat.transpose(-1, -2).view(B, -1, H//self.patch_size, W//self.patch_size)
            feat = nn.functional.pixel_shuffle(feat, self.patch_size)  # B,1,H,W
            preds = self._apply_activation_single(feat, self.activation)
            depths.append(preds)
        return torch.cat(depths, dim=1)  # B,L,H,W
    
    def _apply_activation_single(
        self, x: torch.Tensor, activation: str = "linear"
    ) -> torch.Tensor:
        """
        Apply activation to single channel output, maintaining semantic consistency with value branch in multi-channel case.
        Supports: exp / relu / sigmoid / softplus / tanh / linear / expp1
        """
        act = activation.lower() if isinstance(activation, str) else activation
        if act == "exp":
            return torch.exp(x)
        if act == "expm1":
            return torch.expm1(x)
        if act == "expp1":
            return torch.exp(x) + 1
        if act == "relu":
            return torch.relu(x)
        if act == "sigmoid":
            return torch.sigmoid(x)
        if act == "softplus":
            return torch.nn.functional.softplus(x)
        if act == "tanh":
            return torch.tanh(x)
        # Default linear
        return x