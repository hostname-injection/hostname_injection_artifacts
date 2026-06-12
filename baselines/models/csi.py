from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from ccd.augment import AugmentConfig, CAHOAugmenter

from .base import BaselineModel
from .char_cnn import _resolve_device


class CSIBaseline(BaselineModel):
    def __init__(
        self,
        *,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        epochs: int = 1,
        batch_size: int = 128,
        lr: float = 2e-5,
        max_pairs: int = 50_000,
        device: str = "auto",
        allow_downloads: bool = False,
        repo_root: Optional[str] = None,
        use_official_repo: bool = False,
    ) -> None:
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.max_pairs = max_pairs
        self.device = _resolve_device(device)
        self.allow_downloads = allow_downloads
        self.repo_root = repo_root
        self.use_official_repo = use_official_repo
        self._model = None
        self._clf = None

    def _load(self):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "sentence-transformers is required for CSI baseline. Install with conda: "
                "conda install -c conda-forge sentence-transformers"
            ) from exc

        if self.use_official_repo and self.repo_root:
            from pathlib import Path

            from baselines.downloads import ensure_repo

            ensure_repo("csi", Path(self.repo_root), allow_downloads=self.allow_downloads)

        if not self.allow_downloads:
            # prevent accidental HF downloads
            import os

            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        self._model = SentenceTransformer(self.model_name, device=self.device)

    def _contrastive_finetune(self, texts: Sequence[str]) -> None:
        from sentence_transformers import InputExample, losses
        from torch.utils.data import DataLoader

        augmenter = CAHOAugmenter(AugmentConfig())
        pairs = []
        for i, text in enumerate(texts):
            if len(pairs) >= self.max_pairs:
                break
            aug = augmenter.augment(text, is_malicious=False)
            pairs.append(InputExample(texts=[text, aug]))
        if not pairs:
            return

        loader = DataLoader(pairs, batch_size=self.batch_size, shuffle=True)
        loss_fn = losses.MultipleNegativesRankingLoss(self._model)
        self._model.fit(
            train_objectives=[(loader, loss_fn)],
            epochs=self.epochs,
            warmup_steps=0,
            show_progress_bar=False,
        )

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> None:
        try:
            from sklearn.linear_model import LogisticRegression
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "scikit-learn is required for CSI baseline. Install with conda: "
                "conda install -c conda-forge scikit-learn"
            ) from exc

        if self._model is None:
            self._load()

        if self.epochs > 0:
            self._contrastive_finetune(texts)

        embeddings = self._model.encode(list(texts), batch_size=self.batch_size, normalize_embeddings=True)
        self._clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        self._clf.fit(embeddings, labels)

    def predict(self, texts: Sequence[str], batch_size: int | None = None) -> List[int]:
        embeddings = self._model.encode(list(texts), batch_size=batch_size or self.batch_size, normalize_embeddings=True)
        preds = self._clf.predict(embeddings)
        return np.asarray(preds, dtype=int).tolist()

    def predict_scores(self, texts: Sequence[str], batch_size: int | None = None):
        embeddings = self._model.encode(list(texts), batch_size=batch_size or self.batch_size, normalize_embeddings=True)
        if hasattr(self._clf, "predict_proba"):
            return self._clf.predict_proba(embeddings)[:, 1].tolist()
        return None
