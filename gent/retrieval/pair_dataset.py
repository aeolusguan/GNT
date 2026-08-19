from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class RetrievalPairDataset(Dataset):
    """Read cached GeNT fmap correspondence pairs written by the prep script."""

    def __init__(
        self,
        root: str | Path,
        *,
        correspondences_per_pair: int = 1024,
        seed: int = 0,
    ) -> None:
        self.root = Path(root)
        self.paths = sorted(self.root.glob("pair_*.pt"))
        if not self.paths:
            raise FileNotFoundError(f"no retrieval pair cache files found under {self.root}")
        self.correspondences_per_pair = int(correspondences_per_pair)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.paths[int(index)]
        sample = torch.load(path, map_location="cpu", weights_only=False)
        src_idx = sample["src_idx"].long()
        tgt_idx = sample["tgt_idx"].long()
        if self.correspondences_per_pair > 0 and src_idx.numel() > self.correspondences_per_pair:
            generator = torch.Generator()
            generator.manual_seed(self.seed + int(index))
            order = torch.randperm(src_idx.numel(), generator=generator)[: self.correspondences_per_pair]
            src_idx = src_idx[order]
            tgt_idx = tgt_idx[order]
        return {
            "fmap0": sample["fmap0"].float(),
            "fmap1": sample["fmap1"].float(),
            "src_idx": src_idx,
            "tgt_idx": tgt_idx,
            "path": str(path),
        }


def retrieval_pair_collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "fmap0": torch.stack([item["fmap0"] for item in items], dim=0),
        "fmap1": torch.stack([item["fmap1"] for item in items], dim=0),
        "src_idx": [item["src_idx"] for item in items],
        "tgt_idx": [item["tgt_idx"] for item in items],
        "path": [item["path"] for item in items],
    }
