from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np

from .config import PriorConfig
from .cone import ConePartition


def _accumulate_prior(
    embeddings: np.ndarray,
    cones: ConePartition,
    smoothing: float,
) -> np.ndarray:
    M = cones.config.num_cones
    counts = np.zeros(M, dtype=np.float64)
    for u in embeddings:
        idx, weights = cones.cone_sketch(u)
        counts[idx] += weights
    total = counts.sum()
    if total == 0:
        total = 1.0
    counts = (counts + smoothing) / (total + smoothing * M)
    return counts.astype(np.float32)


def build_benign_prior(
    benign_embeddings: np.ndarray,
    cones: ConePartition,
    config: PriorConfig = PriorConfig(),
) -> np.ndarray:
    return _accumulate_prior(benign_embeddings, cones, config.smoothing)


def build_malicious_priors(
    embeddings_by_family: Dict[str, np.ndarray],
    cones: ConePartition,
    config: PriorConfig = PriorConfig(),
) -> Dict[str, np.ndarray]:
    priors: Dict[str, np.ndarray] = {}
    for family, emb in embeddings_by_family.items():
        priors[family] = _accumulate_prior(emb, cones, config.smoothing)
    return priors
