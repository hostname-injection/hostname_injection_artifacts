from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ccd.benchmark_dataset import (
    BenchmarkFamily,
    BenchmarkLabelMethod,
    BenchmarkTextField,
    HostnameCommandInjectionBenchmarkDataset,
)


@dataclass
class DatasetStats:
    total_rows: int
    used_rows: int
    dropped_rows: int
    benign: int
    malicious: int
    family_rows: Mapping[str, int]


def _reservoir_update(samples: List[str], seen: int, candidate: str, k: int, rng: np.random.Generator) -> None:
    if len(samples) < k:
        samples.append(candidate)
        return
    idx = rng.integers(0, seen)
    if idx < k:
        samples[idx] = candidate


def load_benchmark_dataset(
    root: Path,
    *,
    label_method: BenchmarkLabelMethod | str = BenchmarkLabelMethod.ANY_MALICIOUS_ELSE_BENIGN,
    sample_per_class: Optional[int] = None,
    deduplicate: bool = False,
    seed: int = 13,
    normalize: bool = True,
    max_rows: Optional[int] = None,
) -> Tuple[List[str], np.ndarray, DatasetStats]:
    """Load baseline data from all benchmark families.

    Baselines intentionally mirror the CAHO/CCD contract and use the complete
    benchmark family set. This helper does not expose a user-logins-only path.
    """

    dataset = HostnameCommandInjectionBenchmarkDataset(
        root,
        family=BenchmarkFamily.BOTH,
        label_method=label_method,
        drop_unknown=True,
        include_explanations=False,
        include_metadata=False,
        return_dict=True,
        text_field=BenchmarkTextField.AUTO,
        normalize_text=normalize,
        max_rows=max_rows,
        cache_chunks=1,
    )

    rng = np.random.default_rng(seed)
    seen_b = 0
    seen_m = 0
    samples_b: List[str] = []
    samples_m: List[str] = []
    texts: List[str] = []
    labels: List[int] = []
    seen: set[str] = set()

    for idx in range(len(dataset)):
        item = dataset[idx]
        text = str(item["text"]).strip()
        label = int(item["label"])
        if not text or label not in {0, 1}:
            continue
        if deduplicate:
            if text in seen:
                continue
            seen.add(text)
        if sample_per_class is None:
            texts.append(text)
            labels.append(label)
            continue
        if label == 0:
            seen_b += 1
            _reservoir_update(samples_b, seen_b, text, sample_per_class, rng)
        else:
            seen_m += 1
            _reservoir_update(samples_m, seen_m, text, sample_per_class, rng)

    if sample_per_class is not None:
        texts = samples_b + samples_m
        labels = [0] * len(samples_b) + [1] * len(samples_m)

    stats = DatasetStats(
        total_rows=dataset.stats.total_rows,
        used_rows=len(texts),
        dropped_rows=max(dataset.stats.total_rows - len(texts), 0),
        benign=labels.count(0),
        malicious=labels.count(1),
        family_rows=dataset.stats.family_rows,
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
