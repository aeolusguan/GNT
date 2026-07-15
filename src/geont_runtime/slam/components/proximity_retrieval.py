from __future__ import annotations

import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from geont.retrieval import load_retrieval_head

from .buffer import GraphBuffer


@dataclass(frozen=True)
class RetrievalHit:
    target: int
    score: float


@dataclass(frozen=True)
class ProximityCandidate:
    source: int
    target: int
    score: float


def _load_codebook_centroids(path: str | Path) -> np.ndarray:
    codebook_path = Path(path)
    if not codebook_path.exists():
        raise FileNotFoundError(f"proximity_asmk_codebook_path not found: {codebook_path}")

    if codebook_path.suffix == ".npz":
        data = np.load(codebook_path)
        if "centroids" not in data:
            raise KeyError(f"{codebook_path} must contain a 'centroids' array")
        centroids = data["centroids"]
    else:
        with codebook_path.open("rb") as f:
            state = pickle.load(f)
        if isinstance(state, Mapping) and "state" in state and "centroids" in state["state"]:
            centroids = state["state"]["centroids"]
        elif isinstance(state, Mapping) and "centroids" in state:
            centroids = state["centroids"]
        else:
            raise KeyError(
                f"{codebook_path} must be an ASMK Codebook state dict or contain a 'centroids' entry"
            )

    centroids = np.asarray(centroids, dtype=np.float32)
    if centroids.ndim != 2:
        raise ValueError(f"ASMK codebook centroids must be 2D, got shape {centroids.shape}")
    if centroids.shape[0] == 0 or centroids.shape[1] == 0:
        raise ValueError(f"ASMK codebook centroids must be non-empty, got shape {centroids.shape}")
    return centroids


def filter_retrieval_hits(
    hits: Mapping[int, Sequence[RetrievalHit]],
    *,
    temporal_exclusion: int,
    max_pairs: int,
) -> list[ProximityCandidate]:
    selected: list[ProximityCandidate] = []
    seen: set[tuple[int, int]] = set()
    for source in sorted(int(k) for k in hits.keys()):
        for hit in hits[source]:
            target = int(hit.target)
            if target >= source:
                continue
            if source - target <= int(temporal_exclusion):
                continue
            pair = (min(source, target), max(source, target))
            if pair in seen:
                continue
            seen.add(pair)
            selected.append(ProximityCandidate(source=source, target=target, score=float(hit.score)))

    selected.sort(key=lambda candidate: (-candidate.score, candidate.source, candidate.target))
    if max_pairs > 0:
        selected = selected[: int(max_pairs)]
    return selected


class GeoNTFmapASMKRetrieval:
    """ASMK retrieval over GeoNT fmap descriptors projected by a retrieval head."""

    def __init__(
        self,
        buffer: GraphBuffer,
        device: torch.device,
        *,
        codebook_path: str | Path,
        retrieval_head_path: str | Path,
        nfeat: int,
        topk: int,
        score_thresh: float,
        max_pairs_per_update: int,
        temporal_exclusion: int,
        build_multiple_assignment: int = 1,
        query_multiple_assignment: int = 5,
        similarity_alpha: float = 3.0,
        similarity_threshold: float = 0.0,
    ) -> None:
        self.buffer = buffer
        self.device = device
        self.nfeat = int(nfeat)
        self.topk = int(topk)
        self.score_thresh = float(score_thresh)
        self.max_pairs_per_update = int(max_pairs_per_update)
        self.temporal_exclusion = int(temporal_exclusion)
        self.build_multiple_assignment = int(build_multiple_assignment)
        self.query_multiple_assignment = int(query_multiple_assignment)
        self.similarity_alpha = float(similarity_alpha)
        self.similarity_threshold = float(similarity_threshold)

        input_dim = int(buffer.fmaps.shape[2])
        self.retrieval_head = load_retrieval_head(
            retrieval_head_path,
            expected_input_dim=input_dim,
            device=device,
        )
        centroids_np = _load_codebook_centroids(codebook_path)
        descriptor_dim = int(self.retrieval_head.descriptor_dim)
        if int(centroids_np.shape[1]) != descriptor_dim:
            raise ValueError(
                "GeoNT ASMK codebook descriptor dimension mismatch: "
                f"codebook has {centroids_np.shape[1]}, retrieval head outputs {descriptor_dim}"
            )

        self.centroids = torch.as_tensor(centroids_np, dtype=torch.float32, device=device).contiguous()
        self.centroid_norm2 = (self.centroids * self.centroids).sum(dim=1)[None]
        self.buffer_size = int(buffer.fmaps.shape[0])
        self._descriptor_cache: dict[int, torch.Tensor] = {}
        self._indexed: set[int] = set()
        self._indexed_nonempty: set[int] = set()
        self._image_norm = torch.zeros(self.buffer_size, dtype=torch.float32, device=device)
        self._word_vectors: dict[int, torch.Tensor] = {}
        self._word_image_ids: dict[int, torch.Tensor] = {}

    def extract_descriptors(self, keyframe: int) -> torch.Tensor:
        keyframe = int(keyframe)
        if keyframe in self._descriptor_cache:
            return self._descriptor_cache[keyframe]

        fmap = self.buffer.fmaps[keyframe, 0].float()
        _, fmap_h, fmap_w = fmap.shape
        mask = self.buffer.non_sky_masks[keyframe : keyframe + 1, 0:1].float()
        mask = F.interpolate(mask, size=(fmap_h, fmap_w), mode="nearest")[0, 0] > 0.5

        with torch.no_grad():
            descriptors, scores = self.retrieval_head(fmap[None])
        descriptors = descriptors[0]
        scores = scores[0]
        valid = mask.reshape(-1)
        descriptors = descriptors[valid]
        scores = scores[valid]
        if descriptors.numel() == 0:
            descriptors = torch.empty(
                0,
                self.retrieval_head.descriptor_dim,
                dtype=torch.float32,
                device=self.device,
            )
            self._descriptor_cache[keyframe] = descriptors
            return descriptors

        finite = torch.isfinite(descriptors).all(dim=1) & torch.isfinite(scores)
        descriptors = descriptors[finite]
        scores = scores[finite]
        if descriptors.numel() == 0:
            descriptors = torch.empty(
                0,
                self.retrieval_head.descriptor_dim,
                dtype=torch.float32,
                device=self.device,
            )
            self._descriptor_cache[keyframe] = descriptors
            return descriptors

        if self.nfeat > 0 and descriptors.shape[0] > self.nfeat:
            keep = torch.topk(scores, self.nfeat, largest=True).indices
            descriptors = descriptors[keep]

        descriptors = F.normalize(descriptors, p=2, dim=1, eps=1e-6).contiguous()
        self._descriptor_cache[keyframe] = descriptors
        return descriptors

    def _quantize(self, descriptors: torch.Tensor, multiple_assignment: int) -> torch.Tensor:
        if descriptors.numel() == 0:
            return torch.empty(0, 0, dtype=torch.long, device=self.device)
        k = min(int(multiple_assignment), int(self.centroids.shape[0]))
        distances = self.centroid_norm2 - 2.0 * (descriptors @ self.centroids.T)
        return torch.topk(distances, k, dim=1, largest=False).indices

    def _aggregate(self, descriptors: torch.Tensor, word_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if descriptors.numel() == 0 or word_ids.numel() == 0:
            empty_words = torch.empty(0, dtype=torch.long, device=self.device)
            empty_vecs = torch.empty(0, self.centroids.shape[1], dtype=torch.float32, device=self.device)
            return empty_words, empty_vecs

        flat_words = word_ids.reshape(-1)
        unique_words, inverse = torch.unique(flat_words, sorted=True, return_inverse=True)
        expanded_descriptors = descriptors[:, None].expand(-1, word_ids.shape[1], -1).reshape(-1, descriptors.shape[1])
        residuals = expanded_descriptors - self.centroids[flat_words]
        vectors = torch.zeros(
            unique_words.shape[0],
            descriptors.shape[1],
            dtype=descriptors.dtype,
            device=self.device,
        )
        vectors.scatter_add_(0, inverse[:, None].expand(-1, descriptors.shape[1]), residuals)
        vectors = F.normalize(vectors, p=2, dim=1, eps=1e-6)
        return unique_words, vectors

    def _add_to_database(self, keyframe: int) -> None:
        keyframe = int(keyframe)
        if keyframe in self._indexed:
            return

        descriptors = self.extract_descriptors(keyframe)
        self._indexed.add(keyframe)
        if descriptors.numel() == 0:
            return

        words = self._quantize(descriptors, self.build_multiple_assignment)
        unique_words, vectors = self._aggregate(descriptors, words)
        if unique_words.numel() == 0:
            return

        self._indexed_nonempty.add(keyframe)
        self._image_norm[keyframe] = float(unique_words.numel())
        image_ids = torch.full((vectors.shape[0],), keyframe, dtype=torch.long, device=self.device)
        for offset, word in enumerate(unique_words.tolist()):
            vector = vectors[offset : offset + 1]
            image_id = image_ids[offset : offset + 1]
            if word in self._word_vectors:
                self._word_vectors[word] = torch.cat((self._word_vectors[word], vector), dim=0)
                self._word_image_ids[word] = torch.cat((self._word_image_ids[word], image_id), dim=0)
            else:
                self._word_vectors[word] = vector
                self._word_image_ids[word] = image_id

    def ensure_indexed_until(self, end: int) -> None:
        end = min(int(end), int(self.buffer.n_frames))
        for keyframe in range(end):
            self._add_to_database(keyframe)

    def query(self, keyframe: int, *, target_end: int | None = None) -> list[RetrievalHit]:
        keyframe = int(keyframe)
        descriptors = self.extract_descriptors(keyframe)
        if descriptors.numel() == 0 or not self._indexed_nonempty:
            return []

        words = self._quantize(descriptors, self.query_multiple_assignment)
        query_words, query_vectors = self._aggregate(descriptors, words)
        if query_words.numel() == 0:
            return []

        scores = torch.zeros(self.buffer_size, dtype=torch.float32, device=self.device)
        for word, query_vector in zip(query_words.tolist(), query_vectors):
            vectors = self._word_vectors.get(int(word))
            if vectors is None:
                continue
            image_ids = self._word_image_ids[int(word)]
            similarities = vectors @ query_vector
            keep = similarities >= self.similarity_threshold
            if not keep.any():
                continue
            kept_ids = image_ids[keep]
            kept_scores = similarities[keep].pow(self.similarity_alpha)
            kept_scores = kept_scores / torch.sqrt(self._image_norm[kept_ids].clamp_min(1.0))
            scores.scatter_add_(0, kept_ids, kept_scores)

        scores = scores / float(query_words.numel()) ** 0.5
        candidate_ids = torch.as_tensor(sorted(self._indexed_nonempty), dtype=torch.long, device=self.device)
        if target_end is not None:
            candidate_ids = candidate_ids[candidate_ids < int(target_end)]
        if candidate_ids.numel() == 0:
            return []
        candidate_scores = scores[candidate_ids]
        valid = candidate_scores > self.score_thresh
        if not valid.any():
            return []

        candidate_ids = candidate_ids[valid]
        candidate_scores = candidate_scores[valid]
        k = min(self.topk, int(candidate_ids.numel())) if self.topk > 0 else int(candidate_ids.numel())
        order = torch.topk(candidate_scores, k, largest=True).indices
        out = []
        for target, score in zip(candidate_ids[order].tolist(), candidate_scores[order].tolist()):
            out.append(RetrievalHit(target=int(target), score=float(score)))
        return out

    def select(
        self,
        source_start: int,
        source_end: int,
    ) -> tuple[list[ProximityCandidate], dict]:
        self.ensure_indexed_until(source_end)
        hits: dict[int, list[RetrievalHit]] = {}
        for source in range(int(source_start), int(source_end)):
            target_end = max(0, source - self.temporal_exclusion)
            hits[source] = self.query(source, target_end=target_end)

        retrieved_pairs = sum(len(values) for values in hits.values())
        candidates = filter_retrieval_hits(
            hits,
            temporal_exclusion=self.temporal_exclusion,
            max_pairs=self.max_pairs_per_update,
        )
        info = {
            "proximity_asmk_queries": int(max(0, int(source_end) - int(source_start))),
            "proximity_asmk_retrieved_pairs": int(retrieved_pairs),
            "proximity_asmk_filtered_candidates": int(len(candidates)),
            "proximity_asmk_indexed_keyframes": int(len(self._indexed_nonempty)),
        }
        return candidates, info
