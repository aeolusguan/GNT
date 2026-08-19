from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_projector(input_dim: int, descriptor_dim: int, hidden_dims: Sequence[int]) -> nn.Module:
    dims = [int(input_dim), *[int(dim) for dim in hidden_dims], int(descriptor_dim)]
    if len(dims) == 2:
        return nn.Linear(dims[0], dims[1])

    layers: list[nn.Module] = []
    for idx in range(len(dims) - 2):
        layers.append(nn.Linear(dims[idx], dims[idx + 1]))
        layers.append(nn.LayerNorm(dims[idx + 1]))
        layers.append(nn.GELU())
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


class GeNTRetrievalHead(nn.Module):
    """Project GeNT fmap patches into local retrieval descriptors.

    The patch score follows MASt3R retrieval's default `featweights=l2norm` path:
    score patches by the L2 norm of the projected descriptor before post-whitening,
    then L2-normalize the final descriptor for ASMK.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        descriptor_dim: int,
        hidden_dims: Sequence[int] = (),
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.descriptor_dim = int(descriptor_dim)
        self.hidden_dims = tuple(int(dim) for dim in hidden_dims)
        self.register_buffer("pre_mean", torch.zeros(self.input_dim, dtype=torch.float32))
        self.register_buffer("pre_transform", torch.eye(self.input_dim, dtype=torch.float32))
        self.projector = _build_projector(self.input_dim, self.descriptor_dim, self.hidden_dims)
        self.register_buffer("post_mean", torch.zeros(self.descriptor_dim, dtype=torch.float32))
        self.register_buffer("post_transform", torch.eye(self.descriptor_dim, dtype=torch.float32))

    def project_features(
        self,
        features: torch.Tensor,  # [...,input_dim]
    ) -> torch.Tensor:
        features = features.float()
        features = (features - self.pre_mean) @ self.pre_transform
        return self.projector(features)

    def descriptors_from_projected(
        self,
        projected: torch.Tensor,  # [...,descriptor_dim]
    ) -> torch.Tensor:
        descriptors = (projected - self.post_mean) @ self.post_transform
        return F.normalize(descriptors, p=2, dim=-1, eps=1e-6)

    def forward(
        self,
        fmap: torch.Tensor,  # [B,input_dim,H,W]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, height, width = fmap.shape
        features = fmap.float().permute(0, 2, 3, 1).reshape(batch, height * width, self.input_dim)
        projected = self.project_features(features)
        scores = projected.norm(dim=-1)
        descriptors = self.descriptors_from_projected(projected)
        return descriptors, scores

    def checkpoint_config(self) -> dict[str, object]:
        return {
            "input_dim": self.input_dim,
            "descriptor_dim": self.descriptor_dim,
            "hidden_dims": list(self.hidden_dims),
            "score": "projected_l2norm",
        }


def fit_pca_whitening(
    features: torch.Tensor,  # [N,D]
    *,
    eps: float = 1.0e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return row-vector PCA whitening buffers `(mean, transform)`.

    Callers apply it as `(x - mean) @ transform`.
    """

    if features.shape[0] < 2:
        raise ValueError(f"need at least two feature rows to fit whitening, got {features.shape[0]}")

    features = features.float()
    mean = features.mean(dim=0)
    centered = features - mean
    covariance = centered.T @ centered / float(features.shape[0] - 1)
    eigvals, eigvecs = torch.linalg.eigh(covariance)
    order = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[order].clamp_min(float(eps))
    eigvecs = eigvecs[:, order]
    transform = eigvecs @ torch.diag(torch.rsqrt(eigvals))
    return mean.contiguous(), transform.contiguous()


@torch.no_grad()
def set_pre_whitening(
    head: GeNTRetrievalHead,
    features: torch.Tensor,  # [N,input_dim]
    *,
    eps: float = 1.0e-5,
) -> None:
    mean, transform = fit_pca_whitening(features, eps=eps)
    if mean.numel() != head.input_dim:
        raise ValueError(f"pre-whitening dim mismatch: got {mean.numel()}, expected {head.input_dim}")
    head.pre_mean.copy_(mean.to(device=head.pre_mean.device, dtype=head.pre_mean.dtype))
    head.pre_transform.copy_(transform.to(device=head.pre_transform.device, dtype=head.pre_transform.dtype))


@torch.no_grad()
def set_post_whitening(
    head: GeNTRetrievalHead,
    projected_features: torch.Tensor,  # [N,descriptor_dim]
    *,
    eps: float = 1.0e-5,
) -> None:
    mean, transform = fit_pca_whitening(projected_features, eps=eps)
    if mean.numel() != head.descriptor_dim:
        raise ValueError(f"post-whitening dim mismatch: got {mean.numel()}, expected {head.descriptor_dim}")
    head.post_mean.copy_(mean.to(device=head.post_mean.device, dtype=head.post_mean.dtype))
    head.post_transform.copy_(transform.to(device=head.post_transform.device, dtype=head.post_transform.dtype))


def save_retrieval_head(
    head: GeNTRetrievalHead,
    path: str | Path,
    *,
    extra: Mapping[str, object] | None = None,
) -> None:
    payload: dict[str, object] = {
        "config": head.checkpoint_config(),
        "state_dict": head.state_dict(),
    }
    if extra is not None:
        payload["extra"] = dict(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_retrieval_head(
    path: str | Path,
    *,
    expected_input_dim: int,
    device: torch.device,
) -> GeNTRetrievalHead:
    head_path = Path(path)
    if not head_path.exists():
        raise FileNotFoundError(f"proximity_retrieval_head_path not found: {head_path}")
    checkpoint = torch.load(head_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"{head_path} must contain a GeNTRetrievalHead checkpoint dict")
    if "config" not in checkpoint or "state_dict" not in checkpoint:
        raise KeyError(f"{head_path} must contain 'config' and 'state_dict'")
    config = checkpoint["config"]
    if not isinstance(config, Mapping):
        raise TypeError(f"{head_path} config must be a mapping")

    input_dim = int(config["input_dim"])
    descriptor_dim = int(config["descriptor_dim"])
    hidden_dims = tuple(int(dim) for dim in config["hidden_dims"])
    score = str(config["score"])
    if score != "projected_l2norm":
        raise ValueError(f"{head_path} uses unsupported retrieval score mode: {score}")
    if input_dim != int(expected_input_dim):
        raise ValueError(
            "GeNT retrieval head input dimension mismatch: "
            f"checkpoint has {input_dim}, expected {expected_input_dim}"
        )

    head = GeNTRetrievalHead(
        input_dim=input_dim,
        descriptor_dim=descriptor_dim,
        hidden_dims=hidden_dims,
    )
    head.load_state_dict(checkpoint["state_dict"], strict=True)
    head.eval().to(device)
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    return head
