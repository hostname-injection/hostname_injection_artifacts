from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import math
from typing import Any, Dict, Optional

import numpy as np

from .config import CCDConfig
from .cone import ConePartition
from .encoder import CahoEncoder
from .calibration import threshold_for_group
from .model import CCDModel


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
    names = list(bundle.malicious_priors.keys())
    priors = np.stack([bundle.malicious_priors[n] for n in names], axis=0)
    payload = {
        "axes": bundle.axes,
        "benign_prior": bundle.benign_prior,
        "malicious_priors": priors,
        "malicious_names": np.array(names),
        "config": np.array([json.dumps(bundle.config.to_dict())]),
        "format_version": np.array([bundle.format_version]),
    }
    if bundle.threshold is not None:
        threshold = float(bundle.threshold)
        if not math.isfinite(threshold):
            raise ValueError("threshold must be finite when present")
        payload["threshold"] = np.array([threshold], dtype=np.float64)
    if bundle.grouped_thresholds:
        for group in bundle.grouped_thresholds:
            threshold_for_group(group, bundle.threshold, bundle.grouped_thresholds, missing="error")
        payload["grouped_thresholds"] = np.array([json.dumps(bundle.grouped_thresholds, sort_keys=True)])
    np.savez(path, **payload)


def load_model(path: Path) -> CCDModel:
    """Load a CCDModel from an .npz bundle."""
    data = np.load(path, allow_pickle=True)
    axes = data["axes"]
    benign_prior = data["benign_prior"]
    mal_priors = data["malicious_priors"]
    names = data["malicious_names"].tolist()
    config_json = json.loads(data["config"][0])
    config = CCDConfig.from_dict(config_json)
    threshold = None
    if "threshold" in data.files:
        threshold = float(data["threshold"][0])
        if not math.isfinite(threshold):
            threshold = None
    grouped_thresholds = None
    if "grouped_thresholds" in data.files:
        grouped_thresholds = json.loads(str(data["grouped_thresholds"][0]))

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
