from __future__ import annotations

import math
from typing import Dict

import numpy as np

from .config import PriorConfig
from .cone import ConePartition


def _accumulate_prior(
    embeddings: np.ndarray,
    cones: ConePartition,
    smoothing: float,
    name: str,
) -> np.ndarray:
    embeddings = _coerce_embeddings(embeddings, cones, name)
    smoothing = validate_prior_smoothing(smoothing)
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


def validate_prior_smoothing(smoothing: float) -> float:
    smoothing = float(smoothing)
    if not math.isfinite(smoothing) or smoothing <= 0.0:
        raise ValueError("prior smoothing must be finite and positive")
    return smoothing


def _coerce_embeddings(embeddings: np.ndarray, cones: ConePartition, name: str) -> np.ndarray:
    try:
        import torch

        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
    except Exception:
        pass
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array")
    if arr.size == 0 or arr.shape[0] == 0:
        raise ValueError(f"{name} cannot be empty")
    if arr.shape[1] != cones.axes.shape[1]:
        raise ValueError(
            f"{name} dimension does not match cone axes: "
            f"{arr.shape[1]} != {cones.axes.shape[1]}"
        )
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")
    norms = np.linalg.norm(arr, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 0.0):
        raise ValueError(f"{name} rows must have finite non-zero norms")
    return arr


def build_benign_prior(
    benign_embeddings: np.ndarray,
    cones: ConePartition,
    config: PriorConfig = PriorConfig(),
) -> np.ndarray:
    return _accumulate_prior(benign_embeddings, cones, config.smoothing, "benign_embeddings")


def build_malicious_priors(
    embeddings_by_family: Dict[str, np.ndarray],
    cones: ConePartition,
    config: PriorConfig = PriorConfig(),
) -> Dict[str, np.ndarray]:
    if not embeddings_by_family:
        raise ValueError("malicious_embeddings_by_family cannot be empty")
    priors: Dict[str, np.ndarray] = {}
    for family, emb in embeddings_by_family.items():
        if not isinstance(family, str) or not family.strip():
            raise ValueError("malicious family names must be non-empty strings")
        priors[family] = _accumulate_prior(
            emb,
            cones,
            config.smoothing,
            f"malicious_embeddings_by_family[{family!r}]",
        )
    return priors
