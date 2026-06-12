from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ccd.preprocess import normalize_hostname
from ccd.user_logins import (
    DEFAULT_OPUS_COLUMN,
    DEFAULT_OPUS_CONF_COLUMN,
    DEFAULT_SONNET_COLUMN,
    DEFAULT_SONNET_CONF_COLUMN,
    DEFAULT_USER_LOGINS_COLUMN,
    LabelPolicy,
    apply_confidence_filter,
    iter_user_login_rows,
    normalize_label,
    parse_confidence,
    resolve_label,
)


@dataclass
class DatasetStats:
    total_rows: int
    used_rows: int
    dropped_rows: int
    benign: int
    malicious: int


def iter_labelled_user_logins(
    user_logins_dir: Path,
    *,
    hostname_col: str = DEFAULT_USER_LOGINS_COLUMN,
    label_policy: LabelPolicy = LabelPolicy.BOTH_M,
    sonnet_col: str = DEFAULT_SONNET_COLUMN,
    opus_col: str = DEFAULT_OPUS_COLUMN,
    sonnet_conf_col: str = DEFAULT_SONNET_CONF_COLUMN,
    opus_conf_col: str = DEFAULT_OPUS_CONF_COLUMN,
    min_confidence: Optional[float] = None,
    min_sonnet_confidence: Optional[float] = None,
    min_opus_confidence: Optional[float] = None,
    normalize: bool = True,
) -> Iterable[Tuple[str, int]]:
    min_sonnet = min_sonnet_confidence if min_sonnet_confidence is not None else min_confidence
    min_opus = min_opus_confidence if min_opus_confidence is not None else min_confidence

    for hostname, sonnet, opus, sonnet_conf, opus_conf in iter_user_login_rows(
        user_logins_dir,
        hostname_col=hostname_col,
        sonnet_col=sonnet_col,
        opus_col=opus_col,
        sonnet_conf_col=sonnet_conf_col,
        opus_conf_col=opus_conf_col,
    ):
        s_label = normalize_label(sonnet)
        o_label = normalize_label(opus)
        s_conf = parse_confidence(sonnet_conf)
        o_conf = parse_confidence(opus_conf)
        s_label = apply_confidence_filter(s_label, s_conf, min_sonnet)
        o_label = apply_confidence_filter(o_label, o_conf, min_opus)

        resolved = resolve_label(s_label, o_label, label_policy)
        if resolved is None:
            continue

        text = str(hostname).strip()
        if not text:
            continue
        if normalize:
            text = normalize_hostname(text)
        yield text, 1 if resolved == "M" else 0


def _reservoir_update(samples: List[str], seen: int, candidate: str, k: int, rng: np.random.Generator) -> None:
    if len(samples) < k:
        samples.append(candidate)
        return
    idx = rng.integers(0, seen)
    if idx < k:
        samples[idx] = candidate


def load_user_logins_dataset(
    user_logins_dir: Path,
    *,
    hostname_col: str = DEFAULT_USER_LOGINS_COLUMN,
    label_policy: LabelPolicy = LabelPolicy.BOTH_M,
    min_confidence: Optional[float] = None,
    min_sonnet_confidence: Optional[float] = None,
    min_opus_confidence: Optional[float] = None,
    sample_per_class: Optional[int] = None,
    deduplicate: bool = False,
    seed: int = 13,
    normalize: bool = True,
) -> Tuple[List[str], np.ndarray, DatasetStats]:
    rng = np.random.default_rng(seed)
    seen_b = 0
    seen_m = 0
    samples_b: List[str] = []
    samples_m: List[str] = []
    texts: List[str] = []
    labels: List[int] = []
    seen: set = set()

    total_rows = 0
    used_rows = 0
    dropped_rows = 0

    for text, label in iter_labelled_user_logins(
        user_logins_dir,
        hostname_col=hostname_col,
        label_policy=label_policy,
        min_confidence=min_confidence,
        min_sonnet_confidence=min_sonnet_confidence,
        min_opus_confidence=min_opus_confidence,
        normalize=normalize,
    ):
        total_rows += 1
        if deduplicate:
            if text in seen:
                continue
            seen.add(text)
        used_rows += 1
        if sample_per_class is None:
            texts.append(text)
            labels.append(label)
        else:
            if label == 0:
                seen_b += 1
                _reservoir_update(samples_b, seen_b, text, sample_per_class, rng)
            else:
                seen_m += 1
                _reservoir_update(samples_m, seen_m, text, sample_per_class, rng)

    if sample_per_class is not None:
        texts = samples_b + samples_m
        labels = [0] * len(samples_b) + [1] * len(samples_m)
    dropped_rows = max(total_rows - used_rows, 0)

    stats = DatasetStats(
        total_rows=total_rows,
        used_rows=used_rows,
        dropped_rows=dropped_rows,
        benign=labels.count(0),
        malicious=labels.count(1),
    )
    return texts, np.asarray(labels, dtype=np.int64), stats


def train_test_split(
    texts: Sequence[str],
    labels: Sequence[int],
    *,
    test_size: float = 0.2,
    seed: int = 13,
) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have same length")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")

    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=np.int64)
    idx_b = np.where(labels == 0)[0]
    idx_m = np.where(labels == 1)[0]

    rng.shuffle(idx_b)
    rng.shuffle(idx_m)

    n_test_b = max(1, int(len(idx_b) * test_size)) if len(idx_b) > 0 else 0
    n_test_m = max(1, int(len(idx_m) * test_size)) if len(idx_m) > 0 else 0

    test_idx = np.concatenate([idx_b[:n_test_b], idx_m[:n_test_m]])
    train_idx = np.concatenate([idx_b[n_test_b:], idx_m[n_test_m:]])

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    x_train = [texts[i] for i in train_idx]
    x_test = [texts[i] for i in test_idx]
    y_train = labels[train_idx]
    y_test = labels[test_idx]

    return x_train, x_test, y_train, y_test
