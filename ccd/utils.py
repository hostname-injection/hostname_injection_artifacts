from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def batched(iterable: Sequence, batch_size: int) -> Iterable[Sequence]:
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]


def topk_indices(values: np.ndarray, k: int) -> np.ndarray:
    if k >= values.shape[0]:
        return np.argsort(values)[::-1]
    idx = np.argpartition(values, -k)[-k:]
    return idx[np.argsort(values[idx])[::-1]]


def stable_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log(np.maximum(x, eps))
