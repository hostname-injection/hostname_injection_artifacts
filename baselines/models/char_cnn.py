from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .base import BaselineModel


DEFAULT_ALPHABET = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "-._~/\\:?=&%+$@!#;:,[](){}<>\"'|*^`"
)


def _resolve_device(device: str) -> str:
    if device == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            return "cpu"
        return "cpu"
    return device


@dataclass
class CharTokenizer:
    max_len: int = 128
    alphabet: str = DEFAULT_ALPHABET

    def __post_init__(self) -> None:
        self.char_to_idx = {ch: i + 1 for i, ch in enumerate(self.alphabet)}

    def encode(self, text: str) -> np.ndarray:
        arr = np.zeros(self.max_len, dtype=np.int64)
        for i, ch in enumerate(text[: self.max_len]):
            arr[i] = self.char_to_idx.get(ch, 0)
        return arr


class CharCNNBaseline(BaselineModel):
    def __init__(
        self,
        *,
        max_len: int = 128,
        embed_dim: int = 32,
        num_filters: int = 128,
        kernel_sizes: Sequence[int] = (3, 4, 5),
        hidden_dim: int = 128,
        dropout: float = 0.2,
        epochs: int = 2,
        batch_size: int = 128,
        lr: float = 1e-3,
        device: str = "auto",
    ) -> None:
        self.tokenizer = CharTokenizer(max_len=max_len)
        self.embed_dim = embed_dim
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = _resolve_device(device)
        self._model = None

    def _build_model(self):
        import torch
        import torch.nn as nn

        vocab_size = len(self.tokenizer.alphabet) + 1
        convs = nn.ModuleList(
            [
                nn.Conv1d(self.embed_dim, self.num_filters, kernel_size=k, padding=k // 2)
                for k in self.kernel_sizes
            ]
        )

        model = nn.Sequential(
            nn.Embedding(vocab_size, self.embed_dim, padding_idx=0),
        )
        # store components manually
        model.embed = model[0]
        model.convs = convs
        model.dropout = nn.Dropout(self.dropout)
        model.fc1 = nn.Linear(self.num_filters * len(self.kernel_sizes), self.hidden_dim)
        model.fc2 = nn.Linear(self.hidden_dim, 2)
        return model

    def _forward(self, model, x):
        import torch
        import torch.nn.functional as F

        emb = model.embed(x)
        emb = emb.transpose(1, 2)  # (B, E, L)
        conv_outs = []
        for conv in model.convs:
            feat = F.relu(conv(emb))
            feat = torch.max(feat, dim=2).values
            conv_outs.append(feat)
        features = torch.cat(conv_outs, dim=1)
        features = model.dropout(features)
        hidden = F.relu(model.fc1(features))
        hidden = model.dropout(hidden)
        logits = model.fc2(hidden)
        return logits

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        self._model = self._build_model().to(self.device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = torch.nn.CrossEntropyLoss()

        xs = np.stack([self.tokenizer.encode(t) for t in texts])
        ys = np.asarray(labels, dtype=np.int64)
        dataset = TensorDataset(torch.from_numpy(xs), torch.from_numpy(ys))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self._model.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self._forward(self._model, batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

    def predict(self, texts: Sequence[str], batch_size: int | None = None) -> List[int]:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if self._model is None:
            raise RuntimeError("Model not trained")

        batch_size = batch_size or self.batch_size
        xs = np.stack([self.tokenizer.encode(t) for t in texts])
        dataset = TensorDataset(torch.from_numpy(xs))
        loader = DataLoader(dataset, batch_size=batch_size)

        self._model.eval()
        preds: List[int] = []
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                logits = self._forward(self._model, batch_x)
                pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
                preds.extend(pred)
        return preds

    def predict_scores(self, texts: Sequence[str], batch_size: int | None = None):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if self._model is None:
            raise RuntimeError("Model not trained")

        batch_size = batch_size or self.batch_size
        xs = np.stack([self.tokenizer.encode(t) for t in texts])
        dataset = TensorDataset(torch.from_numpy(xs))
        loader = DataLoader(dataset, batch_size=batch_size)

        self._model.eval()
        scores: List[float] = []
        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                logits = self._forward(self._model, batch_x)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist()
                scores.extend(probs)
        return scores
