from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .cone import ConePartition
from .utils import stable_log, softmax


def _logsumexp(values: np.ndarray, axis=None) -> np.ndarray:
    max_values = np.max(values, axis=axis, keepdims=True)
    summed = np.sum(np.exp(values - max_values), axis=axis, keepdims=True)
    out = max_values + np.log(summed)
    if axis is None:
        return np.squeeze(out)
    return np.squeeze(out, axis=axis)


def mixture_log_weights(
    families: Sequence[str],
    mixture_weights: Optional[Mapping[str, float]] = None,
) -> np.ndarray:
    """Return normalized log mixture weights in family order.

    Empty or omitted weights mean a uniform executable-family mixture. If a
    weight map is supplied, it must cover every family so a model bundle cannot
    silently drop a positive prior.
    """
    if not families:
        raise ValueError("malicious_priors is empty; CCD requires at least one malicious prior.")
    if not mixture_weights:
        return np.full(len(families), -math.log(len(families)), dtype=np.float32)

    missing = [family for family in families if family not in mixture_weights]
    if missing:
        raise ValueError(f"mixture_weights missing families: {missing}")
    weights = np.array([float(mixture_weights[family]) for family in families], dtype=np.float64)
    if np.any(weights <= 0.0) or not np.isfinite(weights).all():
        raise ValueError("mixture_weights must be finite and positive")
    weights = weights / weights.sum()
    return np.log(weights).astype(np.float32)


def _validate_effective_count(effective_count: float) -> float:
    effective_count = float(effective_count)
    if not math.isfinite(effective_count) or effective_count <= 0.0:
        raise ValueError("effective_count must be finite and positive")
    return effective_count


def cross_entropy_sparse(idx: np.ndarray, weights: np.ndarray, prior: np.ndarray) -> float:
    # H(Q;P) = -sum Q(j) log P(j)
    return float(-np.sum(weights * stable_log(prior[idx])))


def cross_entropy_sparse_log(idx: np.ndarray, weights: np.ndarray, log_prior: np.ndarray) -> float:
    return float(-np.sum(weights * log_prior[idx]))


def ccd_score(
    u: np.ndarray,
    cones: ConePartition,
    benign_prior: np.ndarray,
    malicious_priors: Dict[str, np.ndarray],
    *,
    effective_count: float = 1.0,
    mixture_weights: Optional[Mapping[str, float]] = None,
) -> float:
    return ccd_score_logpriors(
        u,
        cones,
        stable_log(benign_prior),
        {name: stable_log(prior) for name, prior in malicious_priors.items()},
        effective_count=effective_count,
        mixture_weights=mixture_weights,
    )


def ccd_score_logpriors(
    u: np.ndarray,
    cones: ConePartition,
    benign_log_prior: np.ndarray,
    malicious_log_priors: Dict[str, np.ndarray],
    *,
    effective_count: float = 1.0,
    mixture_weights: Optional[Mapping[str, float]] = None,
) -> float:
    idx, weights = cones.cone_sketch(u)
    hb = cross_entropy_sparse_log(idx, weights, benign_log_prior)
    if not malicious_log_priors:
        raise ValueError("malicious_priors is empty; CCD requires at least one malicious prior.")
    effective_count = _validate_effective_count(effective_count)
    families = list(malicious_log_priors.keys())
    log_weights = mixture_log_weights(families, mixture_weights)
    h_m = np.array(
        [cross_entropy_sparse_log(idx, weights, malicious_log_priors[family]) for family in families],
        dtype=np.float64,
    )
    return float(_logsumexp(log_weights + effective_count * (hb - h_m)))


def ccd_scores(
    embeddings: np.ndarray,
    cones: ConePartition,
    benign_prior: np.ndarray,
    malicious_priors: Dict[str, np.ndarray],
    *,
    effective_count: float = 1.0,
    mixture_weights: Optional[Mapping[str, float]] = None,
) -> np.ndarray:
    if not malicious_priors:
        raise ValueError("malicious_priors is empty; CCD requires at least one malicious prior.")
    return ccd_scores_logpriors(
        embeddings,
        cones,
        stable_log(benign_prior),
        {name: stable_log(prior) for name, prior in malicious_priors.items()},
        effective_count=effective_count,
        mixture_weights=mixture_weights,
    )


def ccd_scores_logpriors(
    embeddings: np.ndarray,
    cones: ConePartition,
    benign_log_prior: np.ndarray,
    malicious_log_priors: Dict[str, np.ndarray],
    *,
    effective_count: float = 1.0,
    mixture_weights: Optional[Mapping[str, float]] = None,
) -> np.ndarray:
    if not malicious_log_priors:
        raise ValueError("malicious_priors is empty; CCD requires at least one malicious prior.")
    effective_count = _validate_effective_count(effective_count)
    families = list(malicious_log_priors.keys())
    log_weights = mixture_log_weights(families, mixture_weights)
    scores = np.zeros(len(embeddings), dtype=np.float32)
    for i, u in enumerate(embeddings):
        idx, weights = cones.cone_sketch(u)
        hb = cross_entropy_sparse_log(idx, weights, benign_log_prior)
        h_m = np.array(
            [cross_entropy_sparse_log(idx, weights, malicious_log_priors[family]) for family in families],
            dtype=np.float64,
        )
        scores[i] = _logsumexp(log_weights + effective_count * (hb - h_m))
    return scores


def ccd_scores_logpriors_topk(
    embeddings: np.ndarray,
    axes: np.ndarray,
    benign_log_prior: np.ndarray,
    malicious_log_priors: Dict[str, np.ndarray],
    temperature: float,
    k: int,
    *,
    effective_count: float = 1.0,
    mixture_weights: Optional[Mapping[str, float]] = None,
) -> np.ndarray:
    if not malicious_log_priors:
        raise ValueError("malicious_priors is empty; CCD requires at least one malicious prior.")
    effective_count = _validate_effective_count(effective_count)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    num_cones = axes.shape[0]
    k = max(1, min(int(k), num_cones))

    sims = embeddings @ axes.T
    idx = np.argpartition(sims, -k, axis=1)[:, -k:]
    top_sims = np.take_along_axis(sims, idx, axis=1)
    logits = temperature * top_sims
    weights = softmax(logits, axis=1).astype(np.float32)

    log_benign = benign_log_prior[idx]
    hb = -(weights * log_benign).sum(axis=1)

    families = list(malicious_log_priors.keys())
    log_weights = mixture_log_weights(families, mixture_weights)
    family_hm = []
    for family in families:
        prior = malicious_log_priors[family]
        log_mal = prior[idx]
        h = -(weights * log_mal).sum(axis=1)
        family_hm.append(h)
    h_m = np.stack(family_hm, axis=0)
    gaps = effective_count * (hb[None, :] - h_m)
    return _logsumexp(log_weights[:, None] + gaps, axis=0).astype(np.float32)


def ccd_scores_torch(
    embeddings,
    axes,
    benign_log_prior,
    malicious_log_priors,
    config,
    k_override=None,
    effective_count: float = 1.0,
    log_mixture_weights=None,
):
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("torch is required for ccd_scores_torch") from exc

    if embeddings.ndim == 1:
        embeddings = embeddings.unsqueeze(0)

    if embeddings.dtype in (torch.float16, torch.bfloat16):
        embeddings = embeddings.float()

    if axes.dtype != torch.float32:
        axes = axes.float()
    if benign_log_prior.dtype != torch.float32:
        benign_log_prior = benign_log_prior.float()
    if malicious_log_priors.dtype != torch.float32:
        malicious_log_priors = malicious_log_priors.float()

    num_cones = axes.shape[0]
    k = int(config.active_cones if k_override is None else k_override)
    if k > num_cones:
        k = num_cones

    # sims: (B, C)
    sims = embeddings @ axes.T
    topk = torch.topk(sims, k=k, dim=1)
    idx = topk.indices
    logits = config.temperature * topk.values
    weights = torch.softmax(logits, dim=1)

    # benign cross-entropy
    log_benign = benign_log_prior[idx]
    hb = -(weights * log_benign).sum(dim=1)

    # malicious cross-entropy (min over families)
    m = malicious_log_priors.shape[0]
    idx_exp = idx.unsqueeze(0).expand(m, -1, -1)
    log_mal_full = malicious_log_priors.unsqueeze(1).expand(-1, idx.shape[0], -1)
    log_mal = torch.gather(log_mal_full, 2, idx_exp)
    h_m = -(weights.unsqueeze(0) * log_mal).sum(dim=2)
    if log_mixture_weights is None:
        log_mixture_weights = torch.full(
            (m,),
            -math.log(m),
            device=h_m.device,
            dtype=torch.float32,
        )
    else:
        log_mixture_weights = log_mixture_weights.to(device=h_m.device, dtype=torch.float32)

    gaps = float(effective_count) * (hb.unsqueeze(0) - h_m)
    return torch.logsumexp(log_mixture_weights.view(m, 1) + gaps, dim=0)


def soft_mixture_score(
    u: np.ndarray,
    cones: ConePartition,
    benign_prior: np.ndarray,
    malicious_priors: Dict[str, np.ndarray],
    weights: Optional[Dict[str, float]] = None,
    *,
    effective_count: float = 1.0,
) -> float:
    return ccd_score(
        u,
        cones,
        benign_prior,
        malicious_priors,
        effective_count=effective_count,
        mixture_weights=weights,
    )
