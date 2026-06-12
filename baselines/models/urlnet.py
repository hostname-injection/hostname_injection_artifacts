from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .base import BaselineModel
from .char_cnn import CharTokenizer, DEFAULT_ALPHABET, _resolve_device


TOKEN_SPLIT = re.compile(r"[./\\:_\-?=&%+@!#;:,\[\](){}<>\"'|*^`]+")


def _tokenize(text: str) -> List[str]:
    return [tok for tok in TOKEN_SPLIT.split(text) if tok]


@dataclass
class TokenVocab:
    max_tokens: int = 20_000

    def build(self, texts: Sequence[str]) -> None:
        counts: Dict[str, int] = {}
        for text in texts:
            for tok in _tokenize(text):
                counts[tok] = counts.get(tok, 0) + 1
        top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[: self.max_tokens]
        self.token_to_idx = {tok: i + 1 for i, (tok, _) in enumerate(top)}

    def encode(self, text: str, max_len: int) -> np.ndarray:
        tokens = _tokenize(text)[:max_len]
        arr = np.zeros(max_len, dtype=np.int64)
        for i, tok in enumerate(tokens):
            arr[i] = self.token_to_idx.get(tok, 0)
        return arr


class URLNetBaseline(BaselineModel):
    def __init__(
        self,
        *,
        char_max_len: int = 128,
        token_max_len: int = 32,
        embed_dim: int = 32,
        num_filters: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        epochs: int = 2,
        batch_size: int = 128,
        lr: float = 1e-3,
        device: str = "auto",
        repo_root: Optional[str] = None,
        use_official_repo: bool = False,
        allow_downloads: bool = False,
    ) -> None:
        self.char_tokenizer = CharTokenizer(max_len=char_max_len, alphabet=DEFAULT_ALPHABET)
        self.token_vocab = TokenVocab()
        self.token_max_len = token_max_len
        self.embed_dim = embed_dim
        self.num_filters = num_filters
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = _resolve_device(device)
        self._model = None
        self.repo_root = repo_root
        self.use_official_repo = use_official_repo
        self.allow_downloads = allow_downloads

        if self.use_official_repo and self.repo_root:
            self._ensure_repo()

    def _ensure_repo(self) -> None:
        from pathlib import Path

        from baselines.downloads import ensure_repo

        if not self.repo_root:
            return
        ensure_repo("urlnet", Path(self.repo_root), allow_downloads=self.allow_downloads)

    def _build_model(self, token_vocab_size: int):
        import torch
        import torch.nn as nn

        char_embed = nn.Embedding(len(self.char_tokenizer.alphabet) + 1, self.embed_dim, padding_idx=0)
        char_conv = nn.Conv1d(self.embed_dim, self.num_filters, kernel_size=3, padding=1)
        token_embed = nn.Embedding(token_vocab_size + 1, self.embed_dim, padding_idx=0)
        fc = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.num_filters + self.embed_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 2),
        )

        class URLNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.char_embed = char_embed
                self.char_conv = char_conv
                self.token_embed = token_embed
                self.fc = fc

            def forward(self, chars, tokens):
                char_emb = self.char_embed(chars).transpose(1, 2)
                char_feat = torch.relu(self.char_conv(char_emb))
                char_feat = torch.max(char_feat, dim=2).values

                token_emb = self.token_embed(tokens)
                token_feat = token_emb.mean(dim=1)

                combined = torch.cat([char_feat, token_feat], dim=1)
                return self.fc(combined)

        return URLNet()

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        self.token_vocab.build(texts)
        self._model = self._build_model(len(self.token_vocab.token_to_idx)).to(self.device)

        xs_char = np.stack([self.char_tokenizer.encode(t) for t in texts])
        xs_tok = np.stack([self.token_vocab.encode(t, self.token_max_len) for t in texts])
        ys = np.asarray(labels, dtype=np.int64)

        dataset = TensorDataset(torch.from_numpy(xs_char), torch.from_numpy(xs_tok), torch.from_numpy(ys))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = torch.nn.CrossEntropyLoss()

        self._model.train()
        for _ in range(self.epochs):
            for batch_char, batch_tok, batch_y in loader:
                batch_char = batch_char.to(self.device)
                batch_tok = batch_tok.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self._model(batch_char, batch_tok)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

    def predict(self, texts: Sequence[str], batch_size: int | None = None) -> List[int]:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if self._model is None:
            raise RuntimeError("Model not trained")

        batch_size = batch_size or self.batch_size
        xs_char = np.stack([self.char_tokenizer.encode(t) for t in texts])
        xs_tok = np.stack([self.token_vocab.encode(t, self.token_max_len) for t in texts])
        dataset = TensorDataset(torch.from_numpy(xs_char), torch.from_numpy(xs_tok))
        loader = DataLoader(dataset, batch_size=batch_size)

        self._model.eval()
        preds: List[int] = []
        with torch.no_grad():
            for batch_char, batch_tok in loader:
                batch_char = batch_char.to(self.device)
                batch_tok = batch_tok.to(self.device)
                logits = self._model(batch_char, batch_tok)
                pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
                preds.extend(pred)
        return preds

    def predict_scores(self, texts: Sequence[str], batch_size: int | None = None):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if self._model is None:
            raise RuntimeError("Model not trained")

        batch_size = batch_size or self.batch_size
        xs_char = np.stack([self.char_tokenizer.encode(t) for t in texts])
        xs_tok = np.stack([self.token_vocab.encode(t, self.token_max_len) for t in texts])
        dataset = TensorDataset(torch.from_numpy(xs_char), torch.from_numpy(xs_tok))
        loader = DataLoader(dataset, batch_size=batch_size)

        self._model.eval()
        scores: List[float] = []
        with torch.no_grad():
            for batch_char, batch_tok in loader:
                batch_char = batch_char.to(self.device)
                batch_tok = batch_tok.to(self.device)
                logits = self._model(batch_char, batch_tok)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist()
                scores.extend(probs)
        return scores
