from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .base import BaselineModel
from .char_cnn import _resolve_device


class URLBERTBaseline(BaselineModel):
    def __init__(
        self,
        *,
        model_name: str = "bert-base-uncased",
        max_length: int = 128,
        epochs: int = 1,
        batch_size: int = 16,
        lr: float = 2e-5,
        device: str = "auto",
        allow_downloads: bool = False,
        repo_root: Optional[str] = None,
        use_official_repo: bool = False,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = _resolve_device(device)
        self.allow_downloads = allow_downloads
        self.repo_root = repo_root
        self.use_official_repo = use_official_repo
        self._model = None
        self._tokenizer = None

    def _load(self):
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "transformers is required for URLBERT baselines. Install with conda: "
                "conda install -c conda-forge transformers"
            ) from exc

        model_name = self.model_name
        if self.use_official_repo and self.repo_root:
            from pathlib import Path

            from baselines.downloads import ensure_repo

            repo_path = ensure_repo("urlbert", Path(self.repo_root), allow_downloads=self.allow_downloads)
            model_name = str(repo_path)

        local_only = not self.allow_downloads
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_only)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            local_files_only=local_only,
        ).to(self.device)

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if self._model is None:
            self._load()

        tokens = self._tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        ys = torch.tensor(labels, dtype=torch.long)

        dataset = TensorDataset(tokens["input_ids"], tokens["attention_mask"], ys)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self._model.parameters(), lr=self.lr)
        self._model.train()

        for _ in range(self.epochs):
            for input_ids, attention_mask, batch_y in loader:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self._model(input_ids=input_ids, attention_mask=attention_mask, labels=batch_y)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

    def predict(self, texts: Sequence[str], batch_size: int | None = None) -> List[int]:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if self._model is None:
            self._load()

        batch_size = batch_size or self.batch_size
        tokens = self._tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        dataset = TensorDataset(tokens["input_ids"], tokens["attention_mask"])
        loader = DataLoader(dataset, batch_size=batch_size)

        self._model.eval()
        preds: List[int] = []
        with torch.no_grad():
            for input_ids, attention_mask in loader:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                logits = self._model(input_ids=input_ids, attention_mask=attention_mask).logits
                pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
                preds.extend(pred)
        return preds

    def predict_scores(self, texts: Sequence[str], batch_size: int | None = None):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        if self._model is None:
            self._load()

        batch_size = batch_size or self.batch_size
        tokens = self._tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        dataset = TensorDataset(tokens["input_ids"], tokens["attention_mask"])
        loader = DataLoader(dataset, batch_size=batch_size)

        self._model.eval()
        scores: List[float] = []
        with torch.no_grad():
            for input_ids, attention_mask in loader:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                logits = self._model(input_ids=input_ids, attention_mask=attention_mask).logits
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist()
                scores.extend(probs)
        return scores
