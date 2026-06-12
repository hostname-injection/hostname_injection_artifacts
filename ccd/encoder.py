from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np

from .config import EncoderConfig
from .utils import l2_normalize


def _resolve_device(name: Optional[str]) -> str:
    if not name or name == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            return "cpu"
        return "cpu"
    return name


class CahoEncoder:
    def __init__(self, config: Optional[EncoderConfig] = None):
        self.config = config or EncoderConfig()
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is required for the CAHO encoder. "
                "Install with `pip install sentence-transformers`"
            ) from exc
        device = _resolve_device(self.config.device)
        self._model = SentenceTransformer(self.config.model_name, device=device)
        self._model.max_seq_length = self.config.max_length
        if self.config.fp16:
            try:
                import torch

                self._model = self._model.to(torch.float16)
            except Exception:
                pass
        try:
            self._model.eval()
        except Exception:
            pass

    @property
    def model(self):
        self._load_model()
        return self._model

    def device_type(self) -> str:
        self._load_model()
        try:
            import torch

            params = self._model.parameters()
            first = next(params, None)
            if first is not None:
                return first.device.type
        except Exception:
            pass
        return "cpu"

    def encode(self, texts: List[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        self._load_model()
        try:
            import torch

            self._model.eval()
            context = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
            with context():
                embeddings = self._model.encode(
                    texts,
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=normalize,
                    show_progress_bar=False,
                )
        except Exception:
            embeddings = self._model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
                show_progress_bar=False,
            )
        if normalize:
            embeddings = l2_normalize(embeddings)
        return embeddings

    def encode_torch(self, texts: List[str], batch_size: int = 32, normalize: bool = True):
        self._load_model()
        try:
            import torch
            import torch.nn.functional as F

            self._model.eval()
            context = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
            with context():
                embeddings = self._model.encode(
                    texts,
                    batch_size=batch_size,
                    convert_to_tensor=True,
                    normalize_embeddings=normalize,
                    show_progress_bar=False,
                )
            if normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)
            return embeddings
        except Exception:
            import torch
            import torch.nn.functional as F

            embeddings = self.encode(texts, batch_size=batch_size, normalize=normalize)
            embeddings = torch.as_tensor(embeddings)
            if normalize:
                embeddings = F.normalize(embeddings, p=2, dim=1)
            return embeddings
