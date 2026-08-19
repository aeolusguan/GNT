from __future__ import annotations

import torch
import torch.nn.functional as F

from .head import GeNTRetrievalHead


def gather_correspondence_descriptors(
    descriptors0: torch.Tensor,  # [B,N,D]
    descriptors1: torch.Tensor,  # [B,N,D]
    src_indices: list[torch.Tensor],  # B x [K_b]
    tgt_indices: list[torch.Tensor],  # B x [K_b]
) -> tuple[torch.Tensor, torch.Tensor]:
    gathered0 = []
    gathered1 = []
    for batch_idx, (src_idx, tgt_idx) in enumerate(zip(src_indices, tgt_indices)):
        src_idx = src_idx.to(device=descriptors0.device, dtype=torch.long)
        tgt_idx = tgt_idx.to(device=descriptors1.device, dtype=torch.long)
        gathered0.append(descriptors0[batch_idx, src_idx])
        gathered1.append(descriptors1[batch_idx, tgt_idx])
    return torch.cat(gathered0, dim=0), torch.cat(gathered1, dim=0)


def symmetric_infonce_loss(
    descriptors0: torch.Tensor,  # [N,D]
    descriptors1: torch.Tensor,  # [N,D]
    *,
    temperature: float,
) -> torch.Tensor:
    logits = descriptors0 @ descriptors1.T / float(temperature)
    labels = torch.arange(logits.shape[0], device=logits.device, dtype=torch.long)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def retrieval_training_step(
    head: GeNTRetrievalHead,
    batch: dict,
    *,
    device: torch.device,
    temperature: float,
    max_correspondences: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    fmap0 = batch["fmap0"].to(device=device, dtype=torch.float32)
    fmap1 = batch["fmap1"].to(device=device, dtype=torch.float32)
    descriptors0, scores0 = head(fmap0)
    descriptors1, scores1 = head(fmap1)
    matched0, matched1 = gather_correspondence_descriptors(
        descriptors0,
        descriptors1,
        batch["src_idx"],
        batch["tgt_idx"],
    )
    if max_correspondences > 0 and matched0.shape[0] > int(max_correspondences):
        order = torch.randperm(matched0.shape[0], device=device)[: int(max_correspondences)]
        matched0 = matched0[order]
        matched1 = matched1[order]

    loss = symmetric_infonce_loss(matched0, matched1, temperature=temperature)
    with torch.no_grad():
        positive_sim = (matched0 * matched1).sum(dim=1)
        metrics = {
            "loss": float(loss.item()),
            "positive_sim": float(positive_sim.mean().item()),
            "score0": float(scores0.mean().item()),
            "score1": float(scores1.mean().item()),
            "n_correspondences": float(matched0.shape[0]),
        }
    return loss, metrics
