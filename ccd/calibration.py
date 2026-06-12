from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


def calibrate_threshold(benign_scores: np.ndarray, alpha: float) -> float:
    """Split-conformal threshold for fixed FPR control.

    t_alpha is the ceil((1-alpha)(n+1))-th order statistic.
    """
    if len(benign_scores) == 0:
        raise ValueError("benign_scores cannot be empty")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    scores = np.sort(benign_scores)
    n = len(scores)
    k = math.ceil((1.0 - alpha) * (n + 1))
    k = max(1, min(k, n))
    return float(scores[k - 1])


def calibrate_thresholds_by_group(
    benign_scores: Sequence[float] | np.ndarray,
    groups: Sequence[str],
    alpha: float,
) -> dict[str, dict[str, Any]]:
    """Split-conformal fixed-FPR thresholds for tenant/window groups."""
    scores = np.asarray(benign_scores, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError("benign_scores must be a 1D array")
    if len(scores) != len(groups):
        raise ValueError("benign_scores and groups must have the same length")
    if len(scores) == 0:
        raise ValueError("benign_scores cannot be empty")

    grouped: dict[str, list[float]] = {}
    for score, group in zip(scores, groups):
        group_name = str(group).strip()
        if not group_name:
            raise ValueError("groups cannot contain empty values")
        grouped.setdefault(group_name, []).append(float(score))

    out: dict[str, dict[str, Any]] = {}
    for group_name, group_scores in sorted(grouped.items()):
        threshold = calibrate_threshold(np.asarray(group_scores, dtype=np.float64), alpha)
        out[group_name] = {
            "threshold": threshold,
            "num_samples": len(group_scores),
            "order_statistic_rank": _order_statistic_rank(len(group_scores), alpha),
        }
    return out


def threshold_for_group(
    group: str,
    default_threshold: float | None,
    grouped_thresholds: Mapping[str, Any] | None,
    *,
    missing: str = "default",
) -> float:
    """Resolve a row threshold from grouped calibration output."""
    if missing not in {"default", "error"}:
        raise ValueError("missing must be 'default' or 'error'")
    group_name = str(group).strip()
    if grouped_thresholds and group_name in grouped_thresholds:
        value = grouped_thresholds[group_name]
        if isinstance(value, Mapping):
            if "threshold" not in value:
                raise ValueError(f"threshold for group {group_name!r} is missing")
            value = value.get("threshold")
        threshold = float(value)
        if not math.isfinite(threshold):
            raise ValueError(f"threshold for group {group_name!r} must be finite")
        return threshold
    if missing == "error":
        raise KeyError(f"no threshold for calibration group {group_name!r}")
    if default_threshold is None:
        raise KeyError(f"no threshold for calibration group {group_name!r} and no default threshold")
    threshold = float(default_threshold)
    if not math.isfinite(threshold):
        raise ValueError("default_threshold must be finite")
    return threshold


def _order_statistic_rank(n: int, alpha: float) -> int:
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    k = math.ceil((1.0 - alpha) * (n + 1))
    return max(1, min(k, n))


def conformal_p_value(score: float, benign_scores: np.ndarray) -> float:
    """Upper-tail conformal p-value under the benign score distribution.

    CCD scores increase with executable-semantics evidence, so the relevant
    benign-tail probability counts calibration scores at least as large as the
    query score.
    """
    n = len(benign_scores)
    if n == 0:
        return 1.0
    count = int(np.sum(benign_scores >= score))
    return (1 + count) / (n + 1)
