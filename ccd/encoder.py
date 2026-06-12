from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

from .config import EncoderConfig
from .utils import l2_normalize

LOCAL_HASH_ENCODER = "ccd-local-hash-encoder"
SENTENCE_TRANSFORMER_MODULES_FILE = "modules.json"


def require_trained_caho_checkpoint(model_name: str, *, purpose: str = "this command") -> Path:
    """Require an explicit filesystem CAHO checkpoint for reviewer-facing runs.

    Unit tests may still construct in-memory models directly, but command-line
    CCD training/scoring must not silently fall back to a base SentenceTransformer
    model or the deterministic local hash encoder.
    """
    value = str(model_name or "").strip()
    if not value:
        raise ValueError(f"{purpose} requires --encoder pointing to a trained CAHO checkpoint directory")
    if value == LOCAL_HASH_ENCODER:
        raise ValueError(
            f"{purpose} requires a trained CAHO checkpoint; {LOCAL_HASH_ENCODER!r} is reserved for unit tests."
        )
    path = Path(value).expanduser()
    if not path.exists():
        raise ValueError(
            f"{purpose} requires an existing trained CAHO checkpoint directory; got {value!r}"
        )
    if not path.is_dir():
        raise ValueError(f"{purpose} requires a trained CAHO checkpoint directory; got file {value!r}")
    modules_path = path / SENTENCE_TRANSFORMER_MODULES_FILE
    if not modules_path.exists():
        raise ValueError(
            f"{purpose} requires a SentenceTransformer CAHO checkpoint directory containing "
            f"{SENTENCE_TRANSFORMER_MODULES_FILE!r}; got {value!r}"
        )
    return path


def require_model_uses_trained_caho_checkpoint(model, *, purpose: str = "this command") -> None:
    config = getattr(model, "config", None)
    encoder_config = getattr(config, "encoder", None)
    model_name = getattr(encoder_config, "model_name", None)
    if model_name is None:
        return
    require_trained_caho_checkpoint(model_name, purpose=purpose)


class _HashingSentenceModel:
    """Deterministic local encoder used for offline smoke tests."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = int(dim)
        self.max_seq_length = 253

    def eval(self):
        return None

    def parameters(self):
        return iter(())

    def encode(
        self,
        texts: List[str],
        *,
        batch_size: int = 32,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ):
        del batch_size, show_progress_bar
        embeddings = np.vstack([self._embed(text) for text in texts]).astype(np.float32)
        if normalize_embeddings:
            embeddings = l2_normalize(embeddings)
        if convert_to_tensor:
            import torch

            return torch.as_tensor(embeddings)
        if convert_to_numpy:
            return embeddings
        return embeddings

    def _embed(self, text: str) -> np.ndarray:
        values = np.empty(self.dim, dtype=np.float32)
        seed = str(text).encode("utf-8", errors="ignore")
        for offset in range(0, self.dim, 16):
            digest = hashlib.blake2b(seed + offset.to_bytes(2, "little"), digest_size=16).digest()
            chunk = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
            values[offset : offset + len(chunk)] = (chunk / 127.5) - 1.0
        return values


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
        if self.config.model_name == LOCAL_HASH_ENCODER:
            self._model = _HashingSentenceModel(dim=384)
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
