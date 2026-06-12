from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from ccd.config import EncoderConfig
from ccd.encoder import CahoEncoder

from .base import BaselineModel


def _resolve_encoder(model_name: str, device: str) -> CahoEncoder:
    config = EncoderConfig(model_name=model_name, device=device)
    return CahoEncoder(config)


@dataclass
class _Threshold:
    value: float

    @classmethod
    def from_scores(cls, scores: np.ndarray, quantile: float = 0.95) -> "_Threshold":
        return cls(float(np.quantile(scores, quantile)))


class KNNAnomalyBaseline(BaselineModel):
    def __init__(
        self,
        *,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        k: int = 5,
        batch_size: int = 256,
        threshold_quantile: float = 0.95,
    ) -> None:
        self.encoder = _resolve_encoder(model_name, device)
        self.k = k
        self.batch_size = batch_size
        self.threshold_quantile = threshold_quantile
        self._nn = None
        self._threshold = None
        self._benign_embeddings = None

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> None:
        try:
            from sklearn.neighbors import NearestNeighbors
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "scikit-learn is required for kNN baselines. Install with conda: "
                "conda install -c conda-forge scikit-learn"
            ) from exc

        embeddings = self.encoder.encode(list(texts), batch_size=self.batch_size)
        labels = np.asarray(labels)
        benign = embeddings[labels == 0]
        if benign.size == 0:
            benign = embeddings
        self._benign_embeddings = benign
        self._nn = NearestNeighbors(n_neighbors=min(self.k, len(benign)), metric="euclidean")
        self._nn.fit(benign)
        scores = self._score_embeddings(benign)
        self._threshold = _Threshold.from_scores(scores, quantile=self.threshold_quantile)

    def _score_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        distances, _ = self._nn.kneighbors(embeddings, return_distance=True)
        return distances.mean(axis=1)

    def predict(self, texts: Sequence[str], batch_size: int | None = None) -> List[int]:
        embeddings = self.encoder.encode(list(texts), batch_size=batch_size or self.batch_size)
        scores = self._score_embeddings(embeddings)
        threshold = self._threshold.value if self._threshold else float(np.quantile(scores, self.threshold_quantile))
        return (scores > threshold).astype(int).tolist()

    def predict_scores(self, texts: Sequence[str], batch_size: int | None = None):
        embeddings = self.encoder.encode(list(texts), batch_size=batch_size or self.batch_size)
        scores = self._score_embeddings(embeddings)
        return scores.tolist()


class MahalanobisBaseline(BaselineModel):
    def __init__(
        self,
        *,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        batch_size: int = 256,
        threshold_quantile: float = 0.95,
        eps: float = 1e-6,
    ) -> None:
        self.encoder = _resolve_encoder(model_name, device)
        self.batch_size = batch_size
        self.threshold_quantile = threshold_quantile
        self.eps = eps
        self._mean = None
        self._inv_cov = None
        self._threshold = None

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> None:
        embeddings = self.encoder.encode(list(texts), batch_size=self.batch_size)
        labels = np.asarray(labels)
        benign = embeddings[labels == 0]
        if benign.size == 0:
            benign = embeddings
        mean = benign.mean(axis=0)
        cov = np.cov(benign.T)
        cov = cov + np.eye(cov.shape[0]) * self.eps
        inv_cov = np.linalg.pinv(cov)
        self._mean = mean
        self._inv_cov = inv_cov
        scores = self._score_embeddings(benign)
        self._threshold = _Threshold.from_scores(scores, quantile=self.threshold_quantile)

    def _score_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        diff = embeddings - self._mean
        return np.sqrt(np.einsum("bi,ij,bj->b", diff, self._inv_cov, diff))

    def predict(self, texts: Sequence[str], batch_size: int | None = None) -> List[int]:
        embeddings = self.encoder.encode(list(texts), batch_size=batch_size or self.batch_size)
        scores = self._score_embeddings(embeddings)
        threshold = self._threshold.value if self._threshold else float(np.quantile(scores, self.threshold_quantile))
        return (scores > threshold).astype(int).tolist()

    def predict_scores(self, texts: Sequence[str], batch_size: int | None = None):
        embeddings = self.encoder.encode(list(texts), batch_size=batch_size or self.batch_size)
        scores = self._score_embeddings(embeddings)
        return scores.tolist()
