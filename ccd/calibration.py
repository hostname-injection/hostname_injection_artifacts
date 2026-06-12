from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


SPLIT_CONFORMAL_ORDER_STATISTIC_FORMULA = "ceil((1-alpha)*(n+1)) clipped to [1,n]"
SPLIT_CONFORMAL_SCORE_ORDER = "ascending"
SPLIT_CONFORMAL_DECISION_RULE = "score > threshold"
SPLIT_CONFORMAL_CALIBRATION_SCORES = "benign_only"


def _coerce_score_vector(benign_scores: Sequence[float] | np.ndarray) -> np.ndarray:
    scores = np.asarray(benign_scores, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError("benign_scores must be a 1D array")
    if len(scores) == 0:
        raise ValueError("benign_scores cannot be empty")
    if not np.isfinite(scores).all():
        raise ValueError("benign_scores must be finite")
    return scores


def coerce_finite_threshold(value: float | int | str, *, name: str = "threshold") -> float:
    threshold = float(value)
    if not math.isfinite(threshold):
        raise ValueError(f"{name} must be finite")
    return threshold


def calibration_order_statistic_rank(n: int, alpha: float) -> int:
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    k = math.ceil((1.0 - alpha) * (n + 1))
    return max(1, min(k, n))


def calibrate_threshold(benign_scores: Sequence[float] | np.ndarray, alpha: float) -> float:
    """Split-conformal threshold for fixed FPR control.

    t_alpha is the ceil((1-alpha)(n+1))-th order statistic.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    scores = np.sort(_coerce_score_vector(benign_scores))
    n = len(scores)
    k = calibration_order_statistic_rank(n, alpha)
    return float(scores[k - 1])


def split_conformal_threshold_metadata(
    benign_scores: Sequence[float] | np.ndarray,
    alpha: float,
) -> dict[str, Any]:
    """Return the calibrated threshold plus auditable decision-rule metadata."""
    scores = _coerce_score_vector(benign_scores)
    threshold = calibrate_threshold(scores, alpha)
    return {
        "alpha": float(alpha),
        "threshold": float(threshold),
        "num_samples": int(len(scores)),
        "order_statistic_rank": calibration_order_statistic_rank(len(scores), alpha),
        "order_statistic_formula": SPLIT_CONFORMAL_ORDER_STATISTIC_FORMULA,
        "score_order": SPLIT_CONFORMAL_SCORE_ORDER,
        "decision_rule": SPLIT_CONFORMAL_DECISION_RULE,
        "calibration_scores": SPLIT_CONFORMAL_CALIBRATION_SCORES,
    }


def calibrate_thresholds_by_group(
    benign_scores: Sequence[float] | np.ndarray,
    groups: Sequence[str],
    alpha: float,
) -> dict[str, dict[str, Any]]:
    """Split-conformal fixed-FPR thresholds for tenant/window groups."""
    scores = _coerce_score_vector(benign_scores)
    if len(scores) != len(groups):
        raise ValueError("benign_scores and groups must have the same length")

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
            "order_statistic_rank": calibration_order_statistic_rank(len(group_scores), alpha),
            "order_statistic_formula": SPLIT_CONFORMAL_ORDER_STATISTIC_FORMULA,
            "score_order": SPLIT_CONFORMAL_SCORE_ORDER,
            "decision_rule": SPLIT_CONFORMAL_DECISION_RULE,
            "calibration_scores": SPLIT_CONFORMAL_CALIBRATION_SCORES,
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
    if not group_name:
        raise ValueError("groups cannot contain empty values")
    if grouped_thresholds and group_name in grouped_thresholds:
        value = grouped_thresholds[group_name]
        if isinstance(value, Mapping):
            if "threshold" not in value:
                raise ValueError(f"threshold for group {group_name!r} is missing")
            value = value.get("threshold")
        return coerce_finite_threshold(value, name=f"threshold for group {group_name!r}")
    if missing == "error":
        raise KeyError(f"no threshold for calibration group {group_name!r}")
    if default_threshold is None:
        raise KeyError(f"no threshold for calibration group {group_name!r} and no default threshold")
    return coerce_finite_threshold(default_threshold, name="default_threshold")


def _order_statistic_rank(n: int, alpha: float) -> int:
    return calibration_order_statistic_rank(n, alpha)


def conformal_p_value(score: float, benign_scores: np.ndarray) -> float:
    """Upper-tail conformal p-value under the benign score distribution.

    CCD scores increase with executable-semantics evidence, so the relevant
    benign-tail probability counts calibration scores at least as large as the
    query score.
    """
    score = coerce_finite_threshold(score, name="score")
    benign_scores = np.asarray(benign_scores, dtype=np.float64)
    if benign_scores.ndim != 1:
        raise ValueError("benign_scores must be a 1D array")
    n = len(benign_scores)
    if n == 0:
        return 1.0
    if not np.isfinite(benign_scores).all():
        raise ValueError("benign_scores must be finite")
    count = int(np.sum(benign_scores >= score))
    return (1 + count) / (n + 1)
