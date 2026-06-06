from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Integral
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .calibration import (
    calibrate_threshold,
    calibrate_thresholds_by_group,
    coerce_finite_threshold,
    conformal_p_value,
    require_calibrated_threshold,
    split_conformal_threshold_metadata,
    threshold_for_group,
)
from .cone import ConePartition
from .config import CCDConfig
from .encoder import CahoEncoder
from .priors import build_benign_prior, build_malicious_priors
from .scoring import (
    ccd_scores,
    ccd_scores_logpriors,
    ccd_scores_logpriors_topk,
    ccd_scores_torch,
    mixture_log_weights,
)
from .preprocess import normalize_hostname
from .utils import l2_normalize, softmax, stable_log


@dataclass
class CCDModel:
    config: CCDConfig
    encoder: CahoEncoder
    cones: ConePartition
    benign_prior: np.ndarray
    malicious_priors: Dict[str, np.ndarray]
    threshold: Optional[float] = None
    grouped_thresholds: Optional[Dict[str, Any]] = None
    _log_benign_prior: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _log_malicious_priors: Optional[Dict[str, np.ndarray]] = field(default=None, init=False, repr=False)
    _torch_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict, init=False, repr=False)
    _fast_cone_scores: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._validate_model_state()

    @classmethod
    def from_embeddings(
        cls,
        benign_embeddings: np.ndarray,
        malicious_embeddings_by_family: Dict[str, np.ndarray],
        config: Optional[CCDConfig] = None,
        axes: Optional[np.ndarray] = None,
    ) -> "CCDModel":
        config = config or CCDConfig()
        cones = ConePartition.build(config.cone, axes=axes)
        benign_prior = build_benign_prior(benign_embeddings, cones, config.prior)
        malicious_priors = build_malicious_priors(malicious_embeddings_by_family, cones, config.prior)
        encoder = CahoEncoder(config.encoder)
        return cls(config, encoder, cones, benign_prior, malicious_priors)

    def score_embeddings(
        self,
        embeddings: np.ndarray,
        *,
        approximate: bool = False,
        approximate_k: Optional[int] = None,
    ) -> np.ndarray:
        if approximate_k is None and approximate:
            approximate_k = 1
        if approximate_k is not None:
            if approximate_k <= 0:
                raise ValueError("approximate_k must be positive")
        embeddings = self._coerce_score_embeddings(embeddings)
        try:
            import torch
        except Exception:
            torch = None

        if torch is not None and isinstance(embeddings, torch.Tensor):
            if approximate_k is not None:
                if approximate_k == 1:
                    return self._score_embeddings_fast(embeddings)
                return self._score_embeddings_topk(embeddings, approximate_k)
            return self._score_embeddings_torch(embeddings)
        if approximate_k is not None:
            if approximate_k == 1:
                return self._score_embeddings_fast(embeddings)
            return self._score_embeddings_topk(embeddings, approximate_k)
        log_benign, log_malicious = self._get_log_priors()
        return ccd_scores_logpriors(
            embeddings,
            self.cones,
            log_benign,
            log_malicious,
            effective_count=self.config.scoring.effective_count,
            mixture_weights=self.config.scoring.mixture_weights,
        )

    def explain_embeddings(
        self,
        embeddings: np.ndarray,
        *,
        hostnames: Optional[List[str]] = None,
        thresholds: Optional[Sequence[float]] = None,
        calibration_groups: Optional[Sequence[str]] = None,
        top_k: int = 3,
        approximate: bool = False,
        approximate_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        try:
            import torch

            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.detach().cpu().numpy()
        except Exception:
            pass

        embeddings = self._coerce_score_embeddings(embeddings)
        embeddings = l2_normalize(embeddings, axis=1)
        if thresholds is not None and len(thresholds) != len(embeddings):
            raise ValueError("thresholds must have one value per embedding")
        if calibration_groups is not None and len(calibration_groups) != len(embeddings):
            raise ValueError("calibration_groups must have one value per embedding")

        scores = self.score_embeddings(
            embeddings,
            approximate=approximate,
            approximate_k=approximate_k,
        )
        log_benign, log_malicious = self._get_log_priors()
        families = list(log_malicious.keys())
        calibrated_threshold = self._require_calibrated_threshold("CCDModel.explain_embeddings")
        if thresholds is None:
            thresholds = [calibrated_threshold] * len(embeddings)

        explanations: List[Dict[str, Any]] = []
        for i, u in enumerate(embeddings):
            threshold = float(thresholds[i])
            idx, weights = self.cones.cone_sketch(u)
            order = np.argsort(weights)[::-1][:top_k]
            cone_details = []
            for j in order:
                cone_idx = int(idx[j])
                weight = float(weights[j])
                sim = float(self.cones.axes[cone_idx] @ u)
                log_b = float(log_benign[cone_idx])
                mal_vals = {fam: float(log_malicious[fam][cone_idx]) for fam in families}
                log_ratio_vals = {fam: float(value - log_b) for fam, value in mal_vals.items()}
                best_family = max(mal_vals, key=mal_vals.get) if mal_vals else None
                min_family = min(mal_vals, key=mal_vals.get) if mal_vals else None
                cone_details.append(
                    {
                        "cone": cone_idx,
                        "similarity": sim,
                        "weight": weight,
                        "log_benign": log_b,
                        "log_malicious": mal_vals,
                        "log_malicious_over_benign": log_ratio_vals,
                        "best_malicious_family": best_family,
                        "min_malicious_family": min_family,
                        "best_log_malicious_over_benign": None
                        if best_family is None
                        else log_ratio_vals[best_family],
                        "min_log_malicious_over_benign": None
                        if min_family is None
                        else log_ratio_vals[min_family],
                    }
                )
            explanation = {
                "index": i,
                "hostname": hostnames[i] if hostnames else None,
                "score": float(scores[i]),
                "prediction": int(scores[i] > threshold),
                "threshold": float(threshold),
                "top_cones": cone_details,
            }
            if calibration_groups is not None:
                explanation["calibration_group"] = calibration_groups[i]
            explanations.append(explanation)
        return explanations

    def explain(
        self,
        hostnames: List[str],
        batch_size: int = 32,
        normalize: bool = True,
        *,
        top_k: int = 3,
        calibration_groups: Optional[Sequence[str]] = None,
        missing_group_threshold: str = "default",
        approximate: bool = False,
        approximate_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not hostnames:
            return []
        self._require_calibrated_threshold("CCDModel.explain")
        if normalize:
            hostnames = [normalize_hostname(h) for h in hostnames]
        thresholds = None
        if calibration_groups is not None:
            if len(calibration_groups) != len(hostnames):
                raise ValueError("calibration_groups must have one value per hostname")
            thresholds = [
                self.threshold_for_group(group, missing=missing_group_threshold)
                for group in calibration_groups
            ]
        embeddings = None
        try:
            import torch

            device_type = self.encoder.device_type()
            if device_type in {"cuda", "mps"}:
                embeddings = self.encoder.encode_torch(
                    hostnames,
                    batch_size=batch_size,
                    normalize=True,
                )
        except Exception:
            embeddings = None

        if embeddings is None:
            embeddings = self.encoder.encode(hostnames, batch_size=batch_size, normalize=True)

        return self.explain_embeddings(
            embeddings,
            hostnames=hostnames,
            thresholds=thresholds,
            calibration_groups=calibration_groups,
            top_k=top_k,
            approximate=approximate,
            approximate_k=approximate_k,
        )

    def score(
        self,
        hostnames: List[str],
        batch_size: int = 32,
        normalize: bool = True,
        *,
        approximate: bool = False,
        approximate_k: Optional[int] = None,
    ) -> np.ndarray:
        if not hostnames:
            return np.zeros(0, dtype=np.float32)
        if normalize:
            hostnames = [normalize_hostname(h) for h in hostnames]
        embeddings = None
        try:
            import torch

            device_type = self.encoder.device_type()
            if device_type in {"cuda", "mps"}:
                embeddings = self.encoder.encode_torch(
                    hostnames,
                    batch_size=batch_size,
                    normalize=True,
                )
        except Exception:
            embeddings = None

        if embeddings is None:
            embeddings = self.encoder.encode(hostnames, batch_size=batch_size, normalize=True)

        return self.score_embeddings(embeddings, approximate=approximate, approximate_k=approximate_k)

    def _validate_model_state(self) -> None:
        """Reject malformed in-memory model state before it can score."""
        if self.config.cone.to_dict() != self.cones.config.to_dict():
            raise ValueError("model config cone must match cone partition config")

        expected_shape = (int(self.config.cone.num_cones), int(self.config.cone.dim))
        axes = np.asarray(self.cones.axes, dtype=np.float32)
        if axes.ndim != 2 or axes.shape != expected_shape:
            raise ValueError(
                "cone axes shape must match model config: "
                f"observed {axes.shape}, expected {expected_shape}"
            )
        if not np.isfinite(axes).all():
            raise ValueError("cone axes must contain only finite values")
        axis_norms = np.linalg.norm(axes, axis=1)
        if not np.isfinite(axis_norms).all() or np.any(axis_norms <= 0.0):
            raise ValueError("cone axes rows must have finite non-zero norms")
        if not np.allclose(axis_norms, 1.0, rtol=1e-5, atol=1e-5):
            raise ValueError("cone axes must be unit-normalized")
        self.cones.axes = axes

        self.benign_prior = self._validate_prior_vector(
            self.benign_prior,
            "benign_prior",
            expected_shape[0],
        )
        if not self.malicious_priors:
            raise ValueError("malicious_priors is empty; CCD requires at least one malicious prior.")
        validated_malicious: Dict[str, np.ndarray] = {}
        for name, prior in self.malicious_priors.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("malicious_priors keys must be non-empty strings")
            if name in validated_malicious:
                raise ValueError("malicious_priors keys must be unique")
            validated_malicious[name] = self._validate_prior_vector(
                prior,
                f"malicious_priors[{name!r}]",
                expected_shape[0],
            )
        self.malicious_priors = validated_malicious

        self._validate_scoring_config(list(validated_malicious))
        if self.threshold is not None:
            self.threshold = coerce_finite_threshold(self.threshold)
        if self.grouped_thresholds is not None:
            if not isinstance(self.grouped_thresholds, dict):
                raise ValueError("grouped_thresholds must be an object")
            for group in self.grouped_thresholds:
                threshold_for_group(group, self.threshold, self.grouped_thresholds, missing="error")

    @staticmethod
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

    def _validate_scoring_config(self, malicious_names: Sequence[str]) -> None:
        effective_count = float(self.config.scoring.effective_count)
        if not math.isfinite(effective_count) or effective_count <= 0.0:
            raise ValueError("scoring.effective_count must be finite and positive")
        weights = dict(self.config.scoring.mixture_weights or {})
        if weights:
            weight_names = set(weights)
            malicious_name_set = set(malicious_names)
            missing = sorted(malicious_name_set - weight_names)
            extra = sorted(weight_names - malicious_name_set)
            if missing or extra:
                raise ValueError(
                    "scoring.mixture_weights must match malicious_priors exactly: "
                    f"missing={missing}, extra={extra}"
                )
        mixture_log_weights(malicious_names, weights)

    def predict(
        self,
        hostnames: List[str],
        batch_size: int = 32,
        normalize: bool = True,
        *,
        calibration_groups: Optional[Sequence[str]] = None,
        missing_group_threshold: str = "default",
        approximate: bool = False,
        approximate_k: Optional[int] = None,
    ) -> np.ndarray:
        if calibration_groups is not None and len(calibration_groups) != len(hostnames):
            raise ValueError("calibration_groups must have one value per hostname")
        calibrated_threshold = self._require_calibrated_threshold("CCDModel.predict")
        scores = self.score(
            hostnames,
            batch_size=batch_size,
            normalize=normalize,
            approximate=approximate,
            approximate_k=approximate_k,
        )
        if calibration_groups is None:
            return scores > calibrated_threshold
        thresholds = np.asarray(
            [
                self.threshold_for_group(group, missing=missing_group_threshold)
                for group in calibration_groups
            ],
            dtype=np.float64,
        )
        return scores > thresholds

    def threshold_for_group(self, group: str, *, missing: str = "default") -> float:
        return threshold_for_group(group, self.threshold, self.grouped_thresholds, missing=missing)

    def _require_calibrated_threshold(self, purpose: str) -> float:
        return require_calibrated_threshold(self, purpose=purpose)

    def certify(
        self,
        hostname: str,
        *,
        radius: int,
        threshold: Optional[float] = None,
        edit_model=None,
        normalize: bool = True,
        batch_size: int = 32,
        max_nodes: int = 10000,
        method: str = "combined",
        sketch_lipschitz: Optional[float] = None,
        embedding_rotation_bound: Optional[float] = None,
        eps: float = 1e-12,
    ):
        """Certify finite-edit decision stability for one raw hostname.

        The certificate is scoped to this model's normalizer, encoder, cone
        partition, exact score path, threshold, and edit manifest. Approximate
        cone retrieval is intentionally not used while issuing the certificate.
        """
        from .certify import certify_by_calibrated_margin, certify_by_enumeration

        if method not in {"enumeration", "calibrated-margin", "combined"}:
            raise ValueError("method must be one of: enumeration, calibrated-margin, combined")
        if not isinstance(radius, Integral) or radius < 0:
            raise ValueError("radius must be a non-negative integer")
        if method in {"enumeration", "combined"} and (
            not isinstance(max_nodes, Integral) or max_nodes <= 0
        ):
            raise ValueError("max_nodes must be a positive integer")

        calibrated_threshold = self._require_calibrated_threshold("CCDModel.certify")
        threshold = calibrated_threshold if threshold is None else coerce_finite_threshold(threshold)
        has_margin_bounds = sketch_lipschitz is not None or embedding_rotation_bound is not None
        should_try_margin = method == "calibrated-margin" or (method == "combined" and has_margin_bounds)
        if should_try_margin:
            if sketch_lipschitz is None or embedding_rotation_bound is None:
                raise ValueError(
                    "sketch_lipschitz and embedding_rotation_bound are required "
                    "for calibrated-margin certification"
                )
            sketch_lipschitz = float(sketch_lipschitz)
            embedding_rotation_bound = float(embedding_rotation_bound)
            eps = float(eps)
            if not math.isfinite(sketch_lipschitz) or sketch_lipschitz < 0.0:
                raise ValueError("sketch_lipschitz must be finite and non-negative")
            if not math.isfinite(embedding_rotation_bound) or embedding_rotation_bound < 0.0:
                raise ValueError("embedding_rotation_bound must be finite and non-negative")
            if not math.isfinite(eps) or eps <= 0.0:
                raise ValueError("eps must be finite and positive")
        normalizer = normalize_hostname if normalize else (lambda s: s)

        def score_one(text: str) -> float:
            scores = self.score(
                [text],
                batch_size=batch_size,
                normalize=False,
                approximate=False,
            )
            return float(scores[0])

        if should_try_margin:
            margin_cert = certify_by_calibrated_margin(
                score_one(normalizer(hostname)),
                threshold,
                radius=radius,
                effective_count=self.config.scoring.effective_count,
                benign_prior=self.benign_prior,
                malicious_priors=self.malicious_priors,
                sketch_lipschitz=sketch_lipschitz,
                embedding_rotation_bound=embedding_rotation_bound,
                eps=eps,
            )
            if method == "calibrated-margin" or margin_cert.certified:
                return margin_cert

        return certify_by_enumeration(
            hostname,
            score_one,
            threshold,
            radius=radius,
            edit_model=edit_model,
            normalizer=normalizer,
            max_nodes=max_nodes,
        )

    def calibrate(self, benign_scores: np.ndarray, alpha: Optional[float] = None) -> float:
        alpha = alpha if alpha is not None else self.config.calibration.alpha
        self.threshold = calibrate_threshold(benign_scores, alpha)
        self.grouped_thresholds = None
        return self.threshold

    def update_benign_prior(self, benign_embeddings: np.ndarray) -> np.ndarray:
        """Refresh only the benign reference distribution P_B.

        The encoder, cone axes, malicious priors, mixture weights, and scoring
        configuration remain fixed. This is the drift-refresh operation
        described in the paper; it is intentionally narrower than retraining.
        """
        benign_embeddings = self._coerce_benign_embeddings(benign_embeddings)
        self.benign_prior = build_benign_prior(benign_embeddings, self.cones, self.config.prior)
        self._invalidate_prior_caches()
        return self.benign_prior

    def _coerce_score_embeddings(self, embeddings):
        try:
            import torch

            if isinstance(embeddings, torch.Tensor):
                if embeddings.ndim == 1:
                    embeddings = embeddings.unsqueeze(0)
                if embeddings.ndim != 2:
                    raise ValueError("embeddings must be a 1D or 2D array")
                if embeddings.shape[1] != self.cones.axes.shape[1]:
                    raise ValueError(
                        "embeddings dimension does not match cone axes: "
                        f"{embeddings.shape[1]} != {self.cones.axes.shape[1]}"
                    )
                if not torch.isfinite(embeddings).all().item():
                    raise ValueError("embeddings must contain only finite values")
                if embeddings.shape[0] > 0:
                    norms = torch.linalg.vector_norm(embeddings.float(), dim=1)
                    if not torch.isfinite(norms).all().item() or torch.any(norms <= 0.0).item():
                        raise ValueError("embeddings rows must have finite non-zero norms")
                return embeddings
        except ImportError:
            pass

        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a 1D or 2D array")
        if embeddings.shape[1] != self.cones.axes.shape[1]:
            raise ValueError(
                "embeddings dimension does not match cone axes: "
                f"{embeddings.shape[1]} != {self.cones.axes.shape[1]}"
            )
        if not np.isfinite(embeddings).all():
            raise ValueError("embeddings must contain only finite values")
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.isfinite(norms).all() or np.any(norms <= 0.0):
            raise ValueError("embeddings rows must have finite non-zero norms")
        return embeddings

    def _coerce_benign_embeddings(self, benign_embeddings: np.ndarray) -> np.ndarray:
        try:
            import torch

            if isinstance(benign_embeddings, torch.Tensor):
                benign_embeddings = benign_embeddings.detach().cpu().numpy()
        except Exception:
            pass
        benign_embeddings = np.asarray(benign_embeddings, dtype=np.float32)
        if benign_embeddings.ndim == 1:
            benign_embeddings = benign_embeddings.reshape(1, -1)
        if benign_embeddings.ndim != 2:
            raise ValueError("benign_embeddings must be a 1D or 2D array")
        if benign_embeddings.size == 0 or benign_embeddings.shape[0] == 0:
            raise ValueError("benign_embeddings cannot be empty")
        if benign_embeddings.shape[1] != self.cones.axes.shape[1]:
            raise ValueError(
                "benign_embeddings dimension does not match cone axes: "
                f"{benign_embeddings.shape[1]} != {self.cones.axes.shape[1]}"
            )
        if not np.isfinite(benign_embeddings).all():
            raise ValueError("benign_embeddings must contain only finite values")
        return benign_embeddings

    def refresh_benign_reference(
        self,
        benign_embeddings: np.ndarray,
        *,
        alpha: Optional[float] = None,
        calibration_groups: Optional[Sequence[str]] = None,
        drop_grouped_thresholds: bool = False,
        approximate: bool = False,
        approximate_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Refresh P_B and recalibrate tau_alpha from a benign window.

        This updates ``(P_B, tau_alpha)`` only. Positive references, encoder
        config, cone axes, and the score path are left fixed so the operation
        matches the paper's benign-only drift refresh contract. A model with
        tenant/window grouped thresholds must be refreshed with replacement
        groups unless the caller explicitly drops the grouped thresholds.
        """
        alpha_value = alpha if alpha is not None else self.config.calibration.alpha
        old_threshold = self.threshold
        old_grouped_thresholds = self.grouped_thresholds
        if old_grouped_thresholds is not None and calibration_groups is None and not drop_grouped_thresholds:
            raise ValueError(
                "calibration_groups are required when refreshing a model with "
                "grouped thresholds; pass drop_grouped_thresholds=True to "
                "intentionally discard grouped thresholds and keep only the "
                "refreshed global threshold."
            )
        old_benign_prior = self.benign_prior.copy()
        benign_embeddings = self._coerce_benign_embeddings(benign_embeddings)
        new_benign_prior = build_benign_prior(benign_embeddings, self.cones, self.config.prior)

        self.benign_prior = new_benign_prior
        self._invalidate_prior_caches()
        try:
            scores = self.score_embeddings(
                benign_embeddings,
                approximate=approximate,
                approximate_k=approximate_k,
            )
            calibration_metadata = split_conformal_threshold_metadata(scores, alpha_value)
            threshold = calibration_metadata["threshold"]
            grouped_thresholds = (
                calibrate_thresholds_by_group(scores, calibration_groups, alpha_value)
                if calibration_groups is not None
                else None
            )
        except Exception:
            self.benign_prior = old_benign_prior
            self.threshold = old_threshold
            self.grouped_thresholds = old_grouped_thresholds
            self._invalidate_prior_caches()
            raise

        self.threshold = threshold
        grouped_thresholds_dropped = old_grouped_thresholds is not None and calibration_groups is None
        self.grouped_thresholds = grouped_thresholds
        return {
            **calibration_metadata,
            "old_threshold": None if old_threshold is None else float(old_threshold),
            "old_n_calibration_groups": 0 if old_grouped_thresholds is None else len(old_grouped_thresholds),
            "threshold_source": (
                "grouped_benign_refresh_scores" if grouped_thresholds is not None else "benign_refresh_scores"
            ),
            "grouped_thresholds": grouped_thresholds or {},
            "n_calibration_groups": 0 if grouped_thresholds is None else len(grouped_thresholds),
            "grouped_thresholds_dropped": grouped_thresholds_dropped,
            "benign_prior_l1_delta": float(np.sum(np.abs(self.benign_prior - old_benign_prior))),
            "score_path": {
                "approximate": bool(approximate),
                "approximate_k": approximate_k,
            },
            "refresh_scope": {
                "benign_prior_updated": True,
                "threshold_updated": True,
                "grouped_thresholds_updated": grouped_thresholds is not None,
                "grouped_thresholds_dropped": grouped_thresholds_dropped,
                "malicious_priors_fixed": True,
                "encoder_config_fixed": True,
                "cone_axes_fixed": True,
                "scoring_config_fixed": True,
            },
        }

    def p_value(self, score: float, benign_scores: np.ndarray) -> float:
        return conformal_p_value(score, benign_scores)

    def _invalidate_prior_caches(self) -> None:
        self._log_benign_prior = None
        self._log_malicious_priors = None
        self._torch_cache.clear()
        self._fast_cone_scores = None

    def _get_log_priors(self) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
        if self._log_benign_prior is None or self._log_malicious_priors is None:
            self._log_benign_prior = stable_log(self.benign_prior)
            self._log_malicious_priors = {
                name: stable_log(prior) for name, prior in self.malicious_priors.items()
            }
        return self._log_benign_prior, self._log_malicious_priors

    def _get_fast_cone_scores(self) -> np.ndarray:
        if not hasattr(self, "_fast_cone_scores") or self._fast_cone_scores is None:
            log_benign, log_malicious = self._get_log_priors()
            mal_stack = np.stack(list(log_malicious.values()), axis=0)
            families = list(log_malicious.keys())
            log_weights = mixture_log_weights(families, self.config.scoring.mixture_weights)
            gaps = float(self.config.scoring.effective_count) * (mal_stack - log_benign[None, :])
            vals = log_weights[:, None] + gaps
            max_vals = vals.max(axis=0, keepdims=True)
            self._fast_cone_scores = (
                max_vals + np.log(np.exp(vals - max_vals).sum(axis=0, keepdims=True))
            ).reshape(-1).astype(np.float32)
        return self._fast_cone_scores

    def _get_torch_cache(self, device) -> Dict[str, Any]:
        key = str(device)
        cached = self._torch_cache.get(key)
        if cached is not None:
            return cached

        try:
            import torch
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("torch is required for torch-based CCD scoring") from exc

        log_benign, log_malicious = self._get_log_priors()
        families = list(log_malicious.keys())
        axes = torch.as_tensor(self.cones.axes, device=device, dtype=torch.float32)
        log_benign_t = torch.as_tensor(log_benign, device=device, dtype=torch.float32)
        log_mal_list = [torch.as_tensor(log_malicious[name], device=device, dtype=torch.float32) for name in families]
        if not log_mal_list:
            raise ValueError("malicious_priors is empty; CCD requires at least one malicious prior.")
        log_mal_t = torch.stack(log_mal_list, dim=0)
        log_weights = torch.as_tensor(
            mixture_log_weights(families, self.config.scoring.mixture_weights),
            device=device,
            dtype=torch.float32,
        )
        gaps = float(self.config.scoring.effective_count) * (log_mal_t - log_benign_t.unsqueeze(0))
        fast_scores = torch.logsumexp(log_weights.view(-1, 1) + gaps, dim=0)

        cached = {
            "axes": axes,
            "log_benign": log_benign_t,
            "log_malicious": log_mal_t,
            "log_mixture_weights": log_weights,
            "fast_scores": fast_scores,
        }
        self._torch_cache[key] = cached
        return cached

    def _score_embeddings_torch(self, embeddings) -> np.ndarray:
        try:
            import torch
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("torch is required for torch-based CCD scoring") from exc

        if embeddings.ndim == 1:
            embeddings = embeddings.unsqueeze(0)
        cache = self._get_torch_cache(embeddings.device)
        scores = ccd_scores_torch(
            embeddings,
            cache["axes"],
            cache["log_benign"],
            cache["log_malicious"],
            self.cones.config,
            effective_count=self.config.scoring.effective_count,
            log_mixture_weights=cache["log_mixture_weights"],
        )
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
        return scores

    def _score_embeddings_topk(self, embeddings, k: int) -> np.ndarray:
        try:
            import torch
        except Exception:
            torch = None

        if self.cones.lsh is not None:
            if torch is not None and isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.detach().cpu().numpy()
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            log_benign, log_malicious = self._get_log_priors()
            scores = np.zeros(len(embeddings), dtype=np.float32)
            ref_k = self.cones.config.active_cones
            for i, u in enumerate(embeddings):
                if not np.isclose(np.linalg.norm(u), 1.0, atol=1e-3):
                    u = l2_normalize(u)
                idx_ref, sims_ref = self.cones.nearest_axes(u, R=ref_k, use_lsh=True)
                use_k = min(k, len(idx_ref))
                idx = idx_ref[:use_k]
                sims = sims_ref[:use_k]
                logits = self.cones.config.temperature * sims
                weights = softmax(logits)
                hb = -np.sum(weights * log_benign[idx])
                families = list(log_malicious.keys())
                log_weights = mixture_log_weights(families, self.config.scoring.mixture_weights)
                h_m = []
                for family in families:
                    prior = log_malicious[family]
                    h = -np.sum(weights * prior[idx])
                    h_m.append(h)
                vals = log_weights + float(self.config.scoring.effective_count) * (hb - np.array(h_m))
                max_val = vals.max()
                scores[i] = max_val + np.log(np.exp(vals - max_val).sum())
            return scores

        if torch is not None and isinstance(embeddings, torch.Tensor):
            if embeddings.ndim == 1:
                embeddings = embeddings.unsqueeze(0)
            cache = self._get_torch_cache(embeddings.device)
            scores = ccd_scores_torch(
                embeddings,
                cache["axes"],
                cache["log_benign"],
                cache["log_malicious"],
                self.cones.config,
                k_override=k,
                effective_count=self.config.scoring.effective_count,
                log_mixture_weights=cache["log_mixture_weights"],
            )
            if isinstance(scores, torch.Tensor):
                scores = scores.detach().cpu().numpy()
            return scores

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        log_benign, log_malicious = self._get_log_priors()
        return ccd_scores_logpriors_topk(
            embeddings,
            self.cones.axes,
            log_benign,
            log_malicious,
            self.cones.config.temperature,
            k,
            effective_count=self.config.scoring.effective_count,
            mixture_weights=self.config.scoring.mixture_weights,
        )

    def _score_embeddings_fast(self, embeddings) -> np.ndarray:
        try:
            import torch
        except Exception:
            torch = None

        if torch is not None and isinstance(embeddings, torch.Tensor):
            if embeddings.ndim == 1:
                embeddings = embeddings.unsqueeze(0)
            embeddings = torch.nn.functional.normalize(embeddings.float(), p=2, dim=1)
            cache = self._get_torch_cache(embeddings.device)
            sims = embeddings @ cache["axes"].T
            idx = torch.argmax(sims, dim=1)
            scores = cache["fast_scores"][idx]
            return scores.detach().cpu().numpy()

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        embeddings = l2_normalize(np.asarray(embeddings, dtype=np.float32), axis=1)
        scores = self._get_fast_cone_scores()
        sims = embeddings @ self.cones.axes.T
        idx = np.argmax(sims, axis=1)
        return scores[idx]
