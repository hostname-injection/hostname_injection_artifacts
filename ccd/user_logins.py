from __future__ import annotations

import csv
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .config import CCDConfig
from .cone import ConePartition
from .encoder import CahoEncoder
from .io import ModelBundle
from .priors import validate_prior_smoothing
from .preprocess import normalize_hostname


DEFAULT_HOSTNAME_COLUMN = "HOSTNAME"
DEFAULT_USER_LOGINS_COLUMN = "USERNAME"
DEFAULT_SONNET_COLUMN = "GPT_5_5_IS_DNS_CMD_INJECTION"
DEFAULT_OPUS_COLUMN = "CLAUDE_OPUS_4_8_IS_DNS_CMD_INJECTION"
DEFAULT_SONNET_CONF_COLUMN = "GPT_5_5_DNS_CMD_INJECTION_CONFIDENCE"
DEFAULT_OPUS_CONF_COLUMN = "CLAUDE_OPUS_4_8_DNS_CMD_INJECTION_CONFIDENCE"


class LabelPolicy(str, Enum):
    """How to combine GPT 5.5 / Opus 4.8 labels into a training label."""

    BOTH_M = "both-m"
    EITHER_M = "either-m"
    AGREEMENT = "agreement"
    GPT_5_5_ONLY = "gpt-5.5-only"
    OPUS_4_8_ONLY = "opus-4.8-only"
    SONNET_ONLY = "sonnet-only"
    OPUS_ONLY = "opus-only"
    NON_U = "non-u"
    PREFER_M = "prefer-m"
    PREFER_B = "prefer-b"


LABEL_POLICY_DESCRIPTIONS: Dict[str, str] = {
    LabelPolicy.BOTH_M.value: "M if both M, B if both B, otherwise drop (default).",
    LabelPolicy.EITHER_M.value: "M if any M, B if both B, otherwise drop.",
    LabelPolicy.AGREEMENT.value: "Keep only rows where labels agree on B/M.",
    LabelPolicy.GPT_5_5_ONLY.value: "Use GPT 5.5 label only (drop U).",
    LabelPolicy.OPUS_4_8_ONLY.value: "Use Opus 4.8 label only (drop U).",
    LabelPolicy.SONNET_ONLY.value: "Deprecated alias for gpt-5.5-only.",
    LabelPolicy.OPUS_ONLY.value: "Deprecated alias for opus-4.8-only.",
    LabelPolicy.NON_U.value: "Use the non-U label when the other is U; drop B/M conflicts.",
    LabelPolicy.PREFER_M.value: "Prefer M on conflict; otherwise use any B/M.",
    LabelPolicy.PREFER_B.value: "Prefer B on conflict; otherwise use any B/M.",
}


@dataclass
class LabelStats:
    total_rows: int = 0
    used_benign: int = 0
    used_malicious: int = 0
    dropped_rows: int = 0
    combo_counts: Dict[str, int] = field(default_factory=dict)

    def record_combo(self, sonnet: str, opus: str) -> None:
        key = f"{sonnet}/{opus}"
        self.combo_counts[key] = self.combo_counts.get(key, 0) + 1


def normalize_label(label: Optional[str]) -> str:
    if label is None:
        return "U"
    text = str(label).strip().upper()
    if not text:
        return "U"
    if text.startswith("B"):
        return "B"
    if text.startswith("M"):
        return "M"
    if text.startswith("U"):
        return "U"
    return "U"


def parse_confidence(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except Exception:
        return None


def apply_confidence_filter(label: str, confidence: Optional[float], minimum: Optional[float]) -> str:
    if minimum is None:
        return label
    if confidence is None:
        return "U"
    return label if confidence >= minimum else "U"


def resolve_label(sonnet_label: Optional[str], opus_label: Optional[str], policy: LabelPolicy) -> Optional[str]:
    s = normalize_label(sonnet_label)
    o = normalize_label(opus_label)

    if policy in {LabelPolicy.GPT_5_5_ONLY, LabelPolicy.SONNET_ONLY}:
        return s if s in {"B", "M"} else None
    if policy in {LabelPolicy.OPUS_4_8_ONLY, LabelPolicy.OPUS_ONLY}:
        return o if o in {"B", "M"} else None
    if policy == LabelPolicy.AGREEMENT:
        if s == o and s in {"B", "M"}:
            return s
        return None
    if policy == LabelPolicy.BOTH_M:
        if s == "M" and o == "M":
            return "M"
        if s == "B" and o == "B":
            return "B"
        return None
    if policy == LabelPolicy.EITHER_M:
        if s == "M" or o == "M":
            return "M"
        if s == "B" and o == "B":
            return "B"
        return None
    if policy == LabelPolicy.NON_U:
        if s == o and s in {"B", "M"}:
            return s
        if s in {"B", "M"} and o == "U":
            return s
        if o in {"B", "M"} and s == "U":
            return o
        return None
    if policy == LabelPolicy.PREFER_M:
        if s == "M" or o == "M":
            return "M"
        if s == "B" or o == "B":
            return "B"
        return None
    if policy == LabelPolicy.PREFER_B:
        if s == "B" or o == "B":
            return "B"
        if s == "M" or o == "M":
            return "M"
        return None
    raise ValueError(f"Unknown label policy: {policy}")


def _resolve_column(fieldnames: Sequence[str], desired: str) -> str:
    if desired in fieldnames:
        return desired
    lower_map = {name.lower(): name for name in fieldnames}
    match = lower_map.get(desired.lower())
    if match:
        return match
    raise ValueError(f"Column '{desired}' not found in CSV header: {fieldnames}")


def iter_user_login_rows(
    user_logins_dir: Path,
    *,
    hostname_col: str = DEFAULT_USER_LOGINS_COLUMN,
    sonnet_col: str = DEFAULT_SONNET_COLUMN,
    opus_col: str = DEFAULT_OPUS_COLUMN,
    sonnet_conf_col: str = DEFAULT_SONNET_CONF_COLUMN,
    opus_conf_col: str = DEFAULT_OPUS_CONF_COLUMN,
) -> Iterator[Tuple[str, str, str, str, str]]:
    paths = sorted(user_logins_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {user_logins_dir}")
    for path in paths:
        with path.open("r", newline="", errors="ignore") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                continue
            hostname_key = _resolve_column(reader.fieldnames, hostname_col)
            sonnet_key = _resolve_column(reader.fieldnames, sonnet_col)
            opus_key = _resolve_column(reader.fieldnames, opus_col)
            sonnet_conf_key = _resolve_column(reader.fieldnames, sonnet_conf_col)
            opus_conf_key = _resolve_column(reader.fieldnames, opus_conf_col)
            for row in reader:
                yield (
                    row.get(hostname_key, ""),
                    row.get(sonnet_key, ""),
                    row.get(opus_key, ""),
                    row.get(sonnet_conf_key, ""),
                    row.get(opus_conf_key, ""),
                )


def _accumulate_counts(counts: np.ndarray, embeddings: np.ndarray, cones: ConePartition) -> None:
    for u in embeddings:
        idx, weights = cones.cone_sketch(u)
        counts[idx] += weights


def _finalize_prior(counts: np.ndarray, smoothing: float) -> np.ndarray:
    smoothing = validate_prior_smoothing(smoothing)
    total = float(counts.sum())
    if total == 0.0:
        total = 1.0
    m = counts.shape[0]
    return ((counts + smoothing) / (total + smoothing * m)).astype(np.float32)


def build_priors_from_user_logins(
    user_logins_dir: Path,
    config: CCDConfig,
    *,
    encoder: Optional[CahoEncoder] = None,
    label_policy: LabelPolicy = LabelPolicy.BOTH_M,
    min_sonnet_confidence: Optional[float] = None,
    min_opus_confidence: Optional[float] = None,
    batch_size: int = 64,
    buffer_size: int = 2048,
    normalize: bool = True,
    malicious_family: str = "dns_cmd_injection",
    hostname_col: str = DEFAULT_USER_LOGINS_COLUMN,
    sonnet_col: str = DEFAULT_SONNET_COLUMN,
    opus_col: str = DEFAULT_OPUS_COLUMN,
    sonnet_conf_col: str = DEFAULT_SONNET_CONF_COLUMN,
    opus_conf_col: str = DEFAULT_OPUS_CONF_COLUMN,
) -> Tuple[ModelBundle, LabelStats]:
    encoder = encoder or CahoEncoder(config.encoder)
    cones = ConePartition.build(config.cone)
    counts_benign = np.zeros(cones.config.num_cones, dtype=np.float64)
    counts_malicious = np.zeros(cones.config.num_cones, dtype=np.float64)
    stats = LabelStats()

    benign_batch: List[str] = []
    malicious_batch: List[str] = []

    def flush(batch: List[str], counts: np.ndarray) -> int:
        if not batch:
            return 0
        embeddings = encoder.encode(batch, batch_size=batch_size, normalize=True)
        _accumulate_counts(counts, embeddings, cones)
        used = len(batch)
        batch.clear()
        return used

    for hostname, sonnet_label, opus_label, sonnet_conf, opus_conf in iter_user_login_rows(
        user_logins_dir,
        hostname_col=hostname_col,
        sonnet_col=sonnet_col,
        opus_col=opus_col,
        sonnet_conf_col=sonnet_conf_col,
        opus_conf_col=opus_conf_col,
    ):
        stats.total_rows += 1
        s = normalize_label(sonnet_label)
        o = normalize_label(opus_label)
        s = apply_confidence_filter(s, parse_confidence(sonnet_conf), min_sonnet_confidence)
        o = apply_confidence_filter(o, parse_confidence(opus_conf), min_opus_confidence)
        stats.record_combo(s, o)

        label = resolve_label(s, o, label_policy)
        if label is None:
            stats.dropped_rows += 1
            continue

        if not hostname:
            stats.dropped_rows += 1
            continue

        if normalize:
            hostname = normalize_hostname(hostname)
            if not hostname:
                stats.dropped_rows += 1
                continue

        if label == "B":
            benign_batch.append(hostname)
            if len(benign_batch) >= buffer_size:
                stats.used_benign += flush(benign_batch, counts_benign)
        else:
            malicious_batch.append(hostname)
            if len(malicious_batch) >= buffer_size:
                stats.used_malicious += flush(malicious_batch, counts_malicious)

    stats.used_benign += flush(benign_batch, counts_benign)
    stats.used_malicious += flush(malicious_batch, counts_malicious)
    if stats.used_benign == 0:
        raise ValueError("user-login prior training requires at least one benign row")
    if stats.used_malicious == 0:
        raise ValueError("user-login prior training requires at least one malicious row")

    benign_prior = _finalize_prior(counts_benign, config.prior.smoothing)
    malicious_prior = _finalize_prior(counts_malicious, config.prior.smoothing)

    bundle = ModelBundle(
        axes=cones.axes,
        benign_prior=benign_prior,
        malicious_priors={malicious_family: malicious_prior},
        config=config,
    )
    return bundle, stats


def collect_label_stats_from_user_logins(
    user_logins_dir: Path,
    *,
    label_policy: LabelPolicy = LabelPolicy.BOTH_M,
    min_sonnet_confidence: Optional[float] = None,
    min_opus_confidence: Optional[float] = None,
    normalize: bool = True,
    hostname_col: str = DEFAULT_USER_LOGINS_COLUMN,
    sonnet_col: str = DEFAULT_SONNET_COLUMN,
    opus_col: str = DEFAULT_OPUS_COLUMN,
    sonnet_conf_col: str = DEFAULT_SONNET_CONF_COLUMN,
    opus_conf_col: str = DEFAULT_OPUS_CONF_COLUMN,
) -> LabelStats:
    stats = LabelStats()
    for hostname, sonnet_label, opus_label, sonnet_conf, opus_conf in iter_user_login_rows(
        user_logins_dir,
        hostname_col=hostname_col,
        sonnet_col=sonnet_col,
        opus_col=opus_col,
        sonnet_conf_col=sonnet_conf_col,
        opus_conf_col=opus_conf_col,
    ):
        stats.total_rows += 1
        s = normalize_label(sonnet_label)
        o = normalize_label(opus_label)
        s = apply_confidence_filter(s, parse_confidence(sonnet_conf), min_sonnet_confidence)
        o = apply_confidence_filter(o, parse_confidence(opus_conf), min_opus_confidence)
        stats.record_combo(s, o)

        label = resolve_label(s, o, label_policy)
        if label is None:
            stats.dropped_rows += 1
            continue
        if not hostname:
            stats.dropped_rows += 1
            continue
        if normalize:
            hostname = normalize_hostname(hostname)
            if not hostname:
                stats.dropped_rows += 1
                continue
        if label == "B":
            stats.used_benign += 1
        else:
            stats.used_malicious += 1
    return stats


def collect_caho_samples_from_user_logins(
    user_logins_dir: Path,
    *,
    label_policy: LabelPolicy = LabelPolicy.BOTH_M,
    min_sonnet_confidence: Optional[float] = None,
    min_opus_confidence: Optional[float] = None,
    normalize: bool = True,
    malicious_family: str = "dns_cmd_injection",
    hostname_col: str = DEFAULT_USER_LOGINS_COLUMN,
    sonnet_col: str = DEFAULT_SONNET_COLUMN,
    opus_col: str = DEFAULT_OPUS_COLUMN,
    sonnet_conf_col: str = DEFAULT_SONNET_CONF_COLUMN,
    opus_conf_col: str = DEFAULT_OPUS_CONF_COLUMN,
    sample_per_class: Optional[int] = None,
    seed: int = 13,
) -> Tuple[List["Sample"], LabelStats]:
    from .train import Sample
    import random

    rng = random.Random(seed)
    benign_samples: List[Sample] = []
    malicious_samples: List[Sample] = []
    stats = LabelStats()
    seen_benign = 0
    seen_malicious = 0

    def add_sample(target: List[Sample], sample: Sample, seen: int) -> int:
        if sample_per_class is None:
            target.append(sample)
            return seen + 1
        seen += 1
        if len(target) < sample_per_class:
            target.append(sample)
        else:
            j = rng.randrange(seen)
            if j < sample_per_class:
                target[j] = sample
        return seen

    for hostname, sonnet_label, opus_label, sonnet_conf, opus_conf in iter_user_login_rows(
        user_logins_dir,
        hostname_col=hostname_col,
        sonnet_col=sonnet_col,
        opus_col=opus_col,
        sonnet_conf_col=sonnet_conf_col,
        opus_conf_col=opus_conf_col,
    ):
        stats.total_rows += 1
        s = normalize_label(sonnet_label)
        o = normalize_label(opus_label)
        s = apply_confidence_filter(s, parse_confidence(sonnet_conf), min_sonnet_confidence)
        o = apply_confidence_filter(o, parse_confidence(opus_conf), min_opus_confidence)
        stats.record_combo(s, o)

        label = resolve_label(s, o, label_policy)
        if label is None:
            stats.dropped_rows += 1
            continue
        if not hostname:
            stats.dropped_rows += 1
            continue
        if normalize:
            hostname = normalize_hostname(hostname)
            if not hostname:
                stats.dropped_rows += 1
                continue

        if label == "B":
            seen_benign = add_sample(benign_samples, Sample(hostname, is_malicious=False), seen_benign)
        else:
            seen_malicious = add_sample(
                malicious_samples,
                Sample(hostname, is_malicious=True, family=malicious_family),
                seen_malicious,
            )

    stats.used_benign = len(benign_samples)
    stats.used_malicious = len(malicious_samples)
    return benign_samples + malicious_samples, stats
