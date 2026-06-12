from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Sequence

import numpy as np

from .base import BaselineModel


class MarkovBaseline(BaselineModel):
    def __init__(self, n: int = 3, smoothing: float = 1.0) -> None:
        self.n = n
        self.smoothing = smoothing
        self._log_prob_b: Dict[str, float] = {}
        self._log_prob_m: Dict[str, float] = {}
        self._unk_b = 0.0
        self._unk_m = 0.0

    def _ngrams(self, text: str) -> Iterable[str]:
        if len(text) < self.n:
            return []
        return [text[i : i + self.n] for i in range(len(text) - self.n + 1)]

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> None:
        counts_b = Counter()
        counts_m = Counter()
        for text, label in zip(texts, labels):
            grams = self._ngrams(text)
            if int(label) == 1:
                counts_m.update(grams)
            else:
                counts_b.update(grams)

        vocab = set(counts_b.keys()) | set(counts_m.keys())
        vocab_size = max(len(vocab), 1)

        total_b = sum(counts_b.values())
        total_m = sum(counts_m.values())

        denom_b = total_b + self.smoothing * vocab_size
        denom_m = total_m + self.smoothing * vocab_size

        self._unk_b = float(np.log(self.smoothing / denom_b))
        self._unk_m = float(np.log(self.smoothing / denom_m))

        self._log_prob_b = {gram: float(np.log((count + self.smoothing) / denom_b)) for gram, count in counts_b.items()}
        self._log_prob_m = {gram: float(np.log((count + self.smoothing) / denom_m)) for gram, count in counts_m.items()}

    def _score(self, text: str) -> float:
        grams = self._ngrams(text)
        if not grams:
            return 0.0
        log_b = 0.0
        log_m = 0.0
        for gram in grams:
            log_b += self._log_prob_b.get(gram, self._unk_b)
            log_m += self._log_prob_m.get(gram, self._unk_m)
        return log_m - log_b

    def predict(self, texts: Sequence[str], batch_size: int | None = None) -> List[int]:
        scores = [self._score(text) for text in texts]
        return [1 if score > 0 else 0 for score in scores]

    def predict_scores(self, texts: Sequence[str], batch_size: int | None = None) -> List[float]:
        return [self._score(text) for text in texts]
