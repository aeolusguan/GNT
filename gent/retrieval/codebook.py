from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .head import GeNTRetrievalHead


def _pair_paths(root: str | Path) -> list[Path]:
    root = Path(root)
    paths = sorted(root.glob("pair_*.pt"))
    if not paths:
        raise FileNotFoundError(f"no retrieval pair cache files found under {root}")
    return paths


def _sample_rows(features: torch.Tensor, max_rows: int, generator: torch.Generator) -> torch.Tensor:
    features = features.reshape(-1, features.shape[-1]).float()
    if max_rows > 0 and features.shape[0] > int(max_rows):
        order = torch.randperm(features.shape[0], generator=generator)[: int(max_rows)]
        features = features[order]
    return features


def sample_raw_fmap_features(
    cache_root: str | Path,
    *,
    max_features: int,
    max_features_per_pair: int,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    paths = _pair_paths(cache_root)
    order = torch.randperm(len(paths), generator=generator).tolist()
    rows = []
    total = 0
    for path_idx in order:
        sample = torch.load(paths[path_idx], map_location="cpu", weights_only=False)
        fmap0 = sample["fmap0"].permute(1, 2, 0)
        fmap1 = sample["fmap1"].permute(1, 2, 0)
        features = torch.cat((fmap0.reshape(-1, fmap0.shape[-1]), fmap1.reshape(-1, fmap1.shape[-1])), dim=0)
        features = _sample_rows(features, max_features_per_pair, generator)
        rows.append(features)
        total += int(features.shape[0])
        if max_features > 0 and total >= int(max_features):
            break
    out = torch.cat(rows, dim=0)
    if max_features > 0 and out.shape[0] > int(max_features):
        out = out[: int(max_features)]
    return out.contiguous()


@torch.no_grad()
def sample_projected_features(
    head: GeNTRetrievalHead,
    cache_root: str | Path,
    *,
    max_features: int,
    max_features_per_image: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    paths = _pair_paths(cache_root)
    order = torch.randperm(len(paths), generator=generator).tolist()
    rows = []
    total = 0
    head.eval().to(device)
    for path_idx in order:
        sample = torch.load(paths[path_idx], map_location="cpu", weights_only=False)
        fmaps = torch.stack((sample["fmap0"], sample["fmap1"]), dim=0).to(device=device, dtype=torch.float32)
        fmap_features = fmaps.permute(0, 2, 3, 1).reshape(fmaps.shape[0], -1, head.input_dim)
        projected = head.project_features(fmap_features)
        scores = projected.norm(dim=-1)
        selected = []
        for image_idx in range(projected.shape[0]):
            feat = projected[image_idx]
            if max_features_per_image > 0 and feat.shape[0] > int(max_features_per_image):
                keep = torch.topk(scores[image_idx], int(max_features_per_image), largest=True).indices
                feat = feat[keep]
            selected.append(feat.cpu())
        features = torch.cat(selected, dim=0)
        rows.append(features)
        total += int(features.shape[0])
        if max_features > 0 and total >= int(max_features):
            break
    out = torch.cat(rows, dim=0)
    if max_features > 0 and out.shape[0] > int(max_features):
        out = out[: int(max_features)]
    return out.contiguous()


@torch.no_grad()
def sample_descriptors_for_codebook(
    head: GeNTRetrievalHead,
    cache_root: str | Path,
    *,
    max_descriptors: int,
    nfeat_per_image: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    paths = _pair_paths(cache_root)
    order = torch.randperm(len(paths), generator=generator).tolist()
    rows = []
    total = 0
    head.eval().to(device)
    for path_idx in order:
        sample = torch.load(paths[path_idx], map_location="cpu", weights_only=False)
        fmaps = torch.stack((sample["fmap0"], sample["fmap1"]), dim=0).to(device=device, dtype=torch.float32)
        descriptors, scores = head(fmaps)
        selected = []
        for image_idx in range(descriptors.shape[0]):
            desc = descriptors[image_idx]
            if nfeat_per_image > 0 and desc.shape[0] > int(nfeat_per_image):
                keep = torch.topk(scores[image_idx], int(nfeat_per_image), largest=True).indices
                desc = desc[keep]
            selected.append(desc.cpu())
        features = torch.cat(selected, dim=0)
        rows.append(features)
        total += int(features.shape[0])
        if max_descriptors > 0 and total >= int(max_descriptors):
            break
    out = torch.cat(rows, dim=0)
    if max_descriptors > 0 and out.shape[0] > int(max_descriptors):
        out = out[: int(max_descriptors)]
    return out.contiguous()


def train_kmeans_codebook(
    descriptors: torch.Tensor,  # [N,D]
    *,
    num_clusters: int,
    iterations: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    num_clusters = int(num_clusters)
    if descriptors.shape[0] < num_clusters:
        raise ValueError(f"need at least {num_clusters} descriptors for k-means, got {descriptors.shape[0]}")

    descriptors = descriptors.to(device=device, dtype=torch.float32).contiguous()
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    init = torch.randperm(descriptors.shape[0], device=device, generator=generator)[:num_clusters]
    centroids = descriptors[init].clone()

    for _ in range(int(iterations)):
        sums = torch.zeros(num_clusters, descriptors.shape[1], dtype=torch.float32, device=device)
        counts = torch.zeros(num_clusters, dtype=torch.float32, device=device)
        centroid_norm2 = (centroids * centroids).sum(dim=1)[None]
        for start in range(0, descriptors.shape[0], int(batch_size)):
            chunk = descriptors[start : start + int(batch_size)]
            distances = (chunk * chunk).sum(dim=1, keepdim=True) + centroid_norm2 - 2.0 * (chunk @ centroids.T)
            assignment = distances.argmin(dim=1)
            sums.index_add_(0, assignment, chunk)
            counts.index_add_(0, assignment, torch.ones_like(assignment, dtype=torch.float32))

        nonempty = counts > 0
        centroids[nonempty] = sums[nonempty] / counts[nonempty, None]
        if (~nonempty).any():
            replacement = torch.randperm(descriptors.shape[0], device=device, generator=generator)[: int((~nonempty).sum())]
            centroids[~nonempty] = descriptors[replacement]

    return centroids.cpu().contiguous()


def save_codebook_npz(centroids: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, centroids=centroids.float().cpu().numpy())
