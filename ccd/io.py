from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import math
from typing import Any, Dict, Mapping, Optional

import numpy as np

from .config import CCDConfig
from .cone import ConePartition
from .encoder import CahoEncoder
from .calibration import threshold_for_group
from .model import CCDModel
from .scoring import mixture_log_weights


MODEL_FORMAT_VERSION = "1.2"


@dataclass
class ModelBundle:
    axes: np.ndarray
    benign_prior: np.ndarray
    malicious_priors: Dict[str, np.ndarray]
    config: CCDConfig
    threshold: Optional[float] = None
    grouped_thresholds: Optional[Dict[str, Any]] = None
    format_version: str = MODEL_FORMAT_VERSION


def save_model(path: Path, bundle: ModelBundle) -> None:
    """Save a CCD model bundle to an .npz file."""
    axes = _validate_axes(bundle.axes, bundle.config)
    benign_prior = _validate_prior_vector(bundle.benign_prior, "benign_prior", axes.shape[0])
    names, priors = _validate_malicious_prior_mapping(bundle.malicious_priors, axes.shape[0])
    _validate_scoring_config(bundle.config, names)
    payload = {
        "axes": axes,
        "benign_prior": benign_prior,
        "malicious_priors": priors,
        "malicious_names": np.array(names, dtype=str),
        "config": np.array([json.dumps(bundle.config.to_dict())]),
        "format_version": np.array([bundle.format_version]),
    }
    if bundle.threshold is not None:
        threshold = float(bundle.threshold)
        if not math.isfinite(threshold):
            raise ValueError("threshold must be finite when present")
        payload["threshold"] = np.array([threshold], dtype=np.float64)
    if bundle.grouped_thresholds:
        _validate_grouped_thresholds(bundle.grouped_thresholds, default_threshold=bundle.threshold)
        payload["grouped_thresholds"] = np.array(
            [json.dumps(bundle.grouped_thresholds, sort_keys=True, allow_nan=False)]
        )
    np.savez(path, **payload)


def load_model(path: Path) -> CCDModel:
    """Load a CCDModel from an .npz bundle."""
    data = np.load(path, allow_pickle=True)
    config_json = json.loads(data["config"][0])
    config = CCDConfig.from_dict(config_json)
    axes = _validate_axes(data["axes"], config)
    benign_prior = _validate_prior_vector(data["benign_prior"], "benign_prior", axes.shape[0])
    names, mal_priors = _validate_serialized_malicious_priors(
        data["malicious_names"],
        data["malicious_priors"],
        axes.shape[0],
    )
    _validate_scoring_config(config, names)
    threshold = None
    if "threshold" in data.files:
        threshold = float(data["threshold"][0])
        if not math.isfinite(threshold):
            raise ValueError("threshold must be finite when present")
    grouped_thresholds = None
    if "grouped_thresholds" in data.files:
        grouped_thresholds = json.loads(str(data["grouped_thresholds"][0]))
        if not isinstance(grouped_thresholds, dict):
            raise ValueError("grouped_thresholds must be an object")
        _validate_grouped_thresholds(grouped_thresholds, default_threshold=threshold)

    cones = ConePartition.build(config.cone, axes=axes)
    malicious = {name: mal_priors[i] for i, name in enumerate(names)}
    encoder = CahoEncoder(config.encoder)
    model = CCDModel(
        config,
        encoder,
        cones,
        benign_prior,
        malicious,
        threshold=threshold,
        grouped_thresholds=grouped_thresholds,
    )
    return model


def _validate_axes(axes: np.ndarray, config: CCDConfig) -> np.ndarray:
    axes = np.asarray(axes, dtype=np.float32)
    expected_shape = (int(config.cone.num_cones), int(config.cone.dim))
    if axes.ndim != 2 or axes.shape != expected_shape:
        raise ValueError(
            "axes shape must match config cone shape: "
            f"observed {axes.shape}, expected {expected_shape}"
        )
    if not np.isfinite(axes).all():
        raise ValueError("axes must contain only finite values")
    norms = np.linalg.norm(axes, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 0.0):
        raise ValueError("axes rows must have finite non-zero norms")
    return axes


def _validate_prior_vector(prior: np.ndarray, name: str, expected_len: int) -> np.ndarray:
    arr = np.asarray(prior, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] != expected_len:
        raise ValueError(f"{name} must be a 1D prior with {expected_len} entries")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must be non-negative")
    total = float(arr.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError(f"{name} must have positive finite mass")
    if not np.isclose(total, 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError(f"{name} must sum to 1.0")
    return arr.astype(np.float32)


def _validate_malicious_names(names: np.ndarray) -> list[str]:
    names_array = np.asarray(names)
    if names_array.ndim != 1 or names_array.shape[0] == 0:
        raise ValueError("malicious_names must be a non-empty 1D array")
    values = names_array.tolist()
    validated = []
    for name in values:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("malicious_names must contain non-empty strings")
        validated.append(name)
    if len(set(validated)) != len(validated):
        raise ValueError("malicious_names must be unique")
    return validated


def _validate_malicious_prior_mapping(
    malicious_priors: Mapping[str, np.ndarray],
    expected_len: int,
) -> tuple[list[str], np.ndarray]:
    if not malicious_priors:
        raise ValueError("malicious_priors is empty; CCD requires at least one malicious prior.")
    names = _validate_malicious_names(np.array(list(malicious_priors.keys()), dtype=object))
    priors = [
        _validate_prior_vector(malicious_priors[name], f"malicious_priors[{name!r}]", expected_len)
        for name in names
    ]
    return names, np.stack(priors, axis=0)


def _validate_serialized_malicious_priors(
    names: np.ndarray,
    priors: np.ndarray,
    expected_len: int,
) -> tuple[list[str], np.ndarray]:
    validated_names = _validate_malicious_names(names)
    prior_array = np.asarray(priors, dtype=np.float64)
    if prior_array.ndim != 2:
        raise ValueError("malicious_priors must be a 2D array")
    if prior_array.shape != (len(validated_names), expected_len):
        raise ValueError(
            "malicious_priors shape must match malicious_names and cone count: "
            f"observed {prior_array.shape}, expected {(len(validated_names), expected_len)}"
        )
    validated_priors = [
        _validate_prior_vector(prior_array[i], f"malicious_priors[{name!r}]", expected_len)
        for i, name in enumerate(validated_names)
    ]
    return validated_names, np.stack(validated_priors, axis=0)


def _validate_scoring_config(config: CCDConfig, malicious_names: list[str]) -> None:
    effective_count = float(config.scoring.effective_count)
    if not math.isfinite(effective_count) or effective_count <= 0.0:
        raise ValueError("scoring.effective_count must be finite and positive")
    weights = dict(config.scoring.mixture_weights or {})
    if not weights:
        mixture_log_weights(malicious_names, weights)
        return
    weight_names = set(weights)
    malicious_name_set = set(malicious_names)
    missing = sorted(malicious_name_set - weight_names)
    extra = sorted(weight_names - malicious_name_set)
    if missing or extra:
        raise ValueError(
            "scoring.mixture_weights must match malicious_names exactly: "
            f"missing={missing}, extra={extra}"
        )
    mixture_log_weights(malicious_names, weights)


def _validate_grouped_thresholds(
    grouped_thresholds: Mapping[str, Any],
    *,
    default_threshold: Optional[float],
) -> None:
    for group in grouped_thresholds:
        threshold_for_group(group, default_threshold, grouped_thresholds, missing="error")
