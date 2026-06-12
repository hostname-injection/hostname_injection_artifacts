from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .config import ConeConfig
from .utils import l2_normalize, softmax, topk_indices


class RandomHyperplaneLSH:
    """Multi-probe random hyperplane LSH for cosine similarity."""
    def __init__(self, dim: int, tables: int = 8, bits: int = 12, seed: int = 13):
        self.dim = dim
        self.tables = tables
        self.bits = bits
        rng = np.random.default_rng(seed)
        # Hyperplanes: (T, bits, dim)
        self.hyperplanes = rng.standard_normal((tables, bits, dim)).astype(np.float32)
        self.tables_index: List[Dict[int, List[int]]] = []

    def _hash(self, v: np.ndarray, table_id: int) -> int:
        planes = self.hyperplanes[table_id]
        # sign of dot products
        signs = (planes @ v) >= 0
        h = 0
        for i, bit in enumerate(signs):
            if bit:
                h |= (1 << i)
        return h

    def fit(self, vectors: np.ndarray):
        self.tables_index = [dict() for _ in range(self.tables)]
        for idx, v in enumerate(vectors):
            for t in range(self.tables):
                h = self._hash(v, t)
                bucket = self.tables_index[t].setdefault(h, [])
                bucket.append(idx)

    def _neighbor_hashes(self, h: int, radius: int) -> Iterable[int]:
        if radius <= 0:
            yield h
            return
        yield h
        bits = self.bits
        for r in range(1, radius + 1):
            for flips in itertools.combinations(range(bits), r):
                h2 = h
                for b in flips:
                    h2 ^= (1 << b)
                yield h2

    def query_candidates(self, v: np.ndarray, probe_radius: int = 1) -> List[int]:
        candidates = set()
        for t in range(self.tables):
            h = self._hash(v, t)
            for h2 in self._neighbor_hashes(h, probe_radius):
                bucket = self.tables_index[t].get(h2)
                if bucket:
                    candidates.update(bucket)
        return list(candidates)


@dataclass
class ConePartition:
    axes: np.ndarray
    config: ConeConfig
    lsh: Optional[RandomHyperplaneLSH] = None

    @classmethod
    def build(cls, config: ConeConfig, axes: Optional[np.ndarray] = None) -> "ConePartition":
        if axes is None:
            axes = cls._init_axes(config)
        axes = l2_normalize(axes)
        lsh = None
        if config.use_lsh:
            lsh = RandomHyperplaneLSH(
                dim=axes.shape[1],
                tables=config.lsh_tables,
                bits=config.lsh_bits,
                seed=config.seed,
            )
            lsh.fit(axes)
        return cls(axes=axes, config=config, lsh=lsh)

    @staticmethod
    def _init_axes(config: ConeConfig) -> np.ndarray:
        rng = np.random.default_rng(config.seed)
        if config.axis_init == "random":
            axes = rng.standard_normal((config.num_cones, config.dim)).astype(np.float32)
            return axes
        if config.axis_init == "kmeans":
            # simple spherical kmeans initialization (requires embeddings later)
            raise ValueError("kmeans axis_init requires precomputed axes")
        raise ValueError(f"Unknown axis_init: {config.axis_init}")

    def nearest_axes(
        self,
        u: np.ndarray,
        R: Optional[int] = None,
        *,
        use_lsh: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        R = R or self.config.active_cones
        if use_lsh and self.lsh is not None:
            candidates = self.lsh.query_candidates(u, probe_radius=self.config.lsh_probe_radius)
            if len(candidates) >= R:
                cand_axes = self.axes[candidates]
                sims = cand_axes @ u
                idx_local = topk_indices(sims, R)
                idx = np.array([candidates[i] for i in idx_local], dtype=int)
                return idx, sims[idx_local]
        # fallback to brute force
        sims = self.axes @ u
        idx = topk_indices(sims, R)
        return idx, sims[idx]

    def cone_sketch(self, u: np.ndarray, *, use_lsh: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        if not np.isclose(np.linalg.norm(u), 1.0, atol=1e-3):
            u = l2_normalize(u)
        idx, sims = self.nearest_axes(u, self.config.active_cones, use_lsh=use_lsh)
        # temperature-scaled softmax over cosine similarities
        logits = self.config.temperature * sims
        weights = softmax(logits).astype(np.float32)
        return idx, weights
