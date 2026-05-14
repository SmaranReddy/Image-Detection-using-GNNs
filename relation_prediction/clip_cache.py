"""
Efficient CLIP embedding cache with contiguous tensor storage.

Problem:
    Old format: Dict[str, Tensor] pickled via pickle.dump
    451 MB on disk for only 13.9 MB of real tensor data = 32.4x bloat
    Each of ~4500 entries carries ~94 KB pickle overhead.
    At full VG scale (500k+ objects): projected 20-50 GB.

Root cause:
    pickle serialises each torch.Tensor independently with full
    type-info / wrapper overhead.  For 768-dim float32 vectors
    (3072 bytes each), the per-tensor pickle wrapper is ~30x larger
    than the payload.

Solution:
    Single contiguous FloatTensor[N, 768] + lightweight str→int index.
    torch.save uses zipfile compression for the tensor; the index
    is a plain Python dict serialised once, not once per entry.
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, Iterator, List, Optional, Tuple

import torch

CLIP_CACHE_VERSION = 2


class ClipCache:
    """Contiguous-tensor CLIP embedding cache.

    Storage layout (saved via torch.save):
        embeddings: FloatTensor (N, CLIP_DIM)
        index:      Dict[str, int]        key → row index
        metadata:   Dict                  version / dimension info

    Dict-compatible API: .get(), __getitem__, __len__, __contains__,
    keys(), items().
    """

    def __init__(self) -> None:
        self.embeddings: Optional[torch.Tensor] = None
        self.index: Dict[str, int] = {}
        self.metadata: Dict = {}

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        keys: List[str],
        embeddings: List[torch.Tensor],
    ) -> "ClipCache":
        cache = cls()
        if len(keys) == 0:
            cache.embeddings = torch.empty((0, 0))
            return cache
        cache.embeddings = torch.stack(embeddings)  # (N, CLIP_DIM)
        cache.index = {k: i for i, k in enumerate(keys)}
        cache.metadata = {"version": CLIP_CACHE_VERSION, "clip_dim": cache.embeddings.shape[-1]}
        return cache

    @classmethod
    def from_dict(cls, d: Dict[str, torch.Tensor]) -> "ClipCache":
        return cls.build(list(d.keys()), [d[k] for k in d])

    # ------------------------------------------------------------------
    # Dict-like interface
    # ------------------------------------------------------------------

    def get(self, key: str, default: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        idx = self.index.get(key)
        if idx is None:
            return default
        return self.embeddings[idx]

    def __getitem__(self, key: str) -> torch.Tensor:
        return self.embeddings[self.index[key]]

    def __len__(self) -> int:
        return len(self.index)

    def __contains__(self, key: str) -> bool:
        return key in self.index

    def keys(self) -> Iterator[str]:
        return iter(self.index.keys())

    def items(self) -> Iterator[Tuple[str, torch.Tensor]]:
        for k, i in self.index.items():
            yield k, self.embeddings[i]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "embeddings": self.embeddings,
                "index": self.index,
                "metadata": self.metadata,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "ClipCache":
        cache = cls()
        data = torch.load(path, weights_only=True, map_location="cpu")
        cache.embeddings = data["embeddings"]
        cache.index = data["index"]
        cache.metadata = data.get("metadata", {})
        return cache

    # ------------------------------------------------------------------
    # Backward compatibility with old pickle format
    # ------------------------------------------------------------------

    @classmethod
    def load_old_pickle(cls, path: str) -> "ClipCache":
        with open(path, "rb") as f:
            old = pickle.load(f)
        return cls.from_dict(old)
