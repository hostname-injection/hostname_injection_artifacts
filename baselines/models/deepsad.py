from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from ccd.config import EncoderConfig
from ccd.encoder import CahoEncoder

from .base import BaselineModel
from .char_cnn import _resolve_device


class DeepSADBaseline(BaselineModel):
    def __init__(
        self,
        *,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        batch_size: int = 256,
        epochs: int = 3,
        lr: float = 1e-3,
        hidden_dim: int = 128,
        margin: float = 5.0,
        repo_root: Optional[str] = None,
        use_official_repo: bool = False,
        allow_downloads: bool = False,
    ) -> None:
        self.encoder = CahoEncoder(EncoderConfig(model_name=model_name, device=device))
        self.device = _resolve_device(device)
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.margin = margin
        self.repo_root = repo_root
        self.use_official_repo = use_official_repo
        self.allow_downloads = allow_downloads
        self._net = None
        self._center = None
        self._threshold = None

        if self.use_official_repo and self.repo_root:
            from pathlib import Path

            from baselines.downloads import ensure_repo

            ensure_repo("deep-sad", Path(self.repo_root), allow_downloads=self.allow_downloads)

    def _build_net(self, input_dim: int):
        import torch
        import torch.nn as nn

        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        embeddings = self.encoder.encode(list(texts), batch_size=self.batch_size)
        labels = np.asarray(labels)
        input_dim = embeddings.shape[1]
        self._net = self._build_net(input_dim).to(self.device)

        benign = embeddings[labels == 0]
        if benign.size == 0:
            benign = embeddings
        self._center = torch.tensor(benign.mean(axis=0), dtype=torch.float32, device=self.device)

        xs = torch.tensor(embeddings, dtype=torch.float32)
        ys = torch.tensor(labels, dtype=torch.long)
        dataset = TensorDataset(xs, ys)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)

        self._net.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                z = self._net(batch_x)
                dist = torch.sum((z - self._center) ** 2, dim=1)
                loss = torch.where(
                    batch_y == 0,
                    dist,
                    torch.clamp(self.margin - dist, min=0.0),
                ).mean()
                loss.backward()
                optimizer.step()

        # Calibrate threshold on benign distances
        self._net.eval()
        with torch.no_grad():
            benign_t = torch.tensor(benign, dtype=torch.float32, device=self.device)
            z = self._net(benign_t)
            dist = torch.sum((z - self._center) ** 2, dim=1).cpu().numpy()
            self._threshold = float(np.quantile(dist, 0.95))

    def _score(self, embeddings: np.ndarray) -> np.ndarray:
        import torch

        self._net.eval()
        with torch.no_grad():
            x = torch.tensor(embeddings, dtype=torch.float32, device=self.device)
            z = self._net(x)
            dist = torch.sum((z - self._center) ** 2, dim=1).cpu().numpy()
        return dist

    def predict(self, texts: Sequence[str], batch_size: int | None = None) -> List[int]:
        embeddings = self.encoder.encode(list(texts), batch_size=batch_size or self.batch_size)
        scores = self._score(embeddings)
        threshold = self._threshold if self._threshold is not None else float(np.quantile(scores, 0.95))
        return (scores > threshold).astype(int).tolist()

    def predict_scores(self, texts: Sequence[str], batch_size: int | None = None):
        embeddings = self.encoder.encode(list(texts), batch_size=batch_size or self.batch_size)
        scores = self._score(embeddings)
        return scores.tolist()
