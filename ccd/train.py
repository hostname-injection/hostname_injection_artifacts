from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .augment import CAHOAugmenter


CAHO_94GB_ACTUAL_BATCH_SIZE = 16_384
CAHO_94GB_GRAD_CACHE_BATCH_SIZE = 49_152
CAHO_94GB_GRAD_CACHE_CHUNK_SIZE = 8_192
CAHO_DEFAULT_EPOCHS = 20
CAHO_DEFAULT_LR = 1e-4
CAHO_DEFAULT_WEIGHT_DECAY = 1e-2


def resolve_caho_batch_size(batch_size: Optional[int], *, use_grad_cache: bool) -> int:
    """Resolve CAHO training batch defaults for a 94 GB CUDA card.

    The non-GradCache default keeps the full two-view contrastive graph resident
    while leaving headroom for optimizer state. GradCache can use a larger
    effective batch because encoder activations are replayed in chunks.
    """
    if batch_size is None:
        value = CAHO_94GB_GRAD_CACHE_BATCH_SIZE if use_grad_cache else CAHO_94GB_ACTUAL_BATCH_SIZE
    else:
        value = int(batch_size)
    if value <= 0:
        raise ValueError("CAHO batch size must be positive")
    return value


def _require_finite_positive(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _require_finite_nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


@dataclass(frozen=True)
class Sample:
    hostname: str
    is_malicious: bool
    family: Optional[str] = None


class CAHODataset:
    """Dataset wrapper that applies class-aware augmentation."""

    def __init__(
        self,
        samples: Sequence[Sample],
        augmenter: Optional[CAHOAugmenter] = None,
        include_original: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.samples = list(samples)
        self.augmenter = augmenter or CAHOAugmenter()
        self.include_original = include_original
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.samples)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rng_for_index(self, idx: int) -> random.Random:
        if self.seed is None:
            return random.Random()
        epoch_stride = max(1, len(self.samples))
        return random.Random(int(self.seed) + self.epoch * epoch_stride + int(idx))

    def __getitem__(self, idx: int) -> Tuple[str, str, Optional[str]]:
        sample = self.samples[idx]
        rng = self._rng_for_index(idx)
        if self.include_original:
            view1 = sample.hostname
            view2 = self.augmenter.augment(sample.hostname, is_malicious=sample.is_malicious, rng=rng)
        else:
            view1 = self.augmenter.augment(sample.hostname, is_malicious=sample.is_malicious, rng=rng)
            view2 = self.augmenter.augment(sample.hostname, is_malicious=sample.is_malicious, rng=rng)
        label = sample.family if sample.is_malicious else None
        return view1, view2, label


# Training utilities for SupCon loss

def supcon_loss(features: np.ndarray, labels: List[int], temperature: float = 0.07) -> float:
    """Compute supervised contrastive loss (Khosla et al.).

    Args:
        features: Array of shape (batch, n_views, dim).
        labels: Integer labels for each sample in the batch.
        temperature: Contrastive temperature.
    """
    import torch
    import torch.nn.functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feats = torch.tensor(features, device=device, dtype=torch.float32)
    batch, n_views, dim = feats.shape
    feats = F.normalize(feats, dim=-1)
    feats = feats.permute(1, 0, 2).reshape(batch * n_views, dim)

    labels_t = torch.tensor(labels, device=device).view(-1)
    labels_t = labels_t.repeat(n_views)
    mask = labels_t.view(-1, 1).eq(labels_t.view(1, -1)).float()

    logits = feats @ feats.T / temperature
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    logits_mask = torch.ones_like(mask)
    logits_mask.fill_diagonal_(0)
    mask = mask * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-12)
    loss = -mean_log_prob_pos
    loss = loss.view(n_views, batch).mean()
    return float(loss.item())


def pairwise_contrastive_loss(embeddings1, embeddings2, temperature: float = 0.1):
    """Pairwise contrastive loss for two augmented views (SimCSE-style)."""
    import torch
    import torch.nn.functional as F

    device = embeddings1.device
    n = embeddings1.size(0)
    embeddings = torch.cat([embeddings1, embeddings2], dim=0)
    embeddings = F.normalize(embeddings, dim=1)

    sim_matrix = embeddings @ embeddings.T / temperature
    mask = torch.eye(2 * n, dtype=torch.bool, device=device)
    sim_matrix = sim_matrix.masked_fill(mask, -9e15)

    labels = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)], dim=0).to(device)
    return F.cross_entropy(sim_matrix, labels)


def supervised_orbit_contrastive_loss(embeddings1, embeddings2, labels, temperature: float = 0.1):
    """Supervised two-view orbit-collapse loss with benign diversity preserved.

    Labels identify which examples should share an orbit. Callers can assign
    unique labels to benign examples and shared labels to executable-semantics
    families or to the available positive class.
    """
    import torch
    import torch.nn.functional as F

    device = embeddings1.device
    z = torch.cat([embeddings1, embeddings2], dim=0)
    z = F.normalize(z, dim=1)
    labels = torch.as_tensor(labels, device=device).view(-1)
    labels = torch.cat([labels, labels], dim=0)

    logits = z @ z.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    logits_mask = torch.ones_like(logits, dtype=torch.bool)
    logits_mask.fill_diagonal_(False)
    positive_mask = labels.view(-1, 1).eq(labels.view(1, -1)) & logits_mask

    exp_logits = torch.exp(logits) * logits_mask.float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
    positive_counts = positive_mask.sum(dim=1)
    anchors = positive_counts > 0
    if not torch.any(anchors):
        return pairwise_contrastive_loss(embeddings1, embeddings2, temperature=temperature)
    mean_log_prob_pos = (positive_mask.float() * log_prob).sum(dim=1) / positive_counts.clamp_min(1)
    return -mean_log_prob_pos[anchors].mean()


class ContrastiveLoss:
    """Learnable temperature contrastive loss (CLIP-style logit scale)."""

    def __init__(self, init_tau: float = 0.07, max_scale: float = 100.0, min_scale: float = 1.0) -> None:
        import torch

        self.logit_scale = torch.nn.Parameter(torch.tensor(float(np.log(1.0 / init_tau))))
        self.max_logit = float(np.log(max_scale))
        self.min_logit = float(np.log(min_scale))

    def parameters(self):
        return [self.logit_scale]

    def to(self, device):
        import torch

        self.logit_scale = torch.nn.Parameter(self.logit_scale.detach().to(device))
        return self

    def __call__(self, embeddings1, embeddings2, labels=None):
        import torch
        import torch.nn.functional as F

        device = embeddings1.device
        n = embeddings1.size(0)
        z = torch.cat([embeddings1, embeddings2], dim=0)
        z = F.normalize(z, dim=1)

        with torch.no_grad():
            self.logit_scale.clamp_(min=self.min_logit, max=self.max_logit)
        scale = self.logit_scale.exp()

        logits = (z @ z.T) * scale
        mask = torch.eye(2 * n, dtype=torch.bool, device=device)
        logits = logits.masked_fill(mask, -9e15)
        if labels is not None:
            labels_t = torch.as_tensor(labels, device=device).view(-1)
            labels_t = torch.cat([labels_t, labels_t], dim=0)
            positive_mask = labels_t.view(-1, 1).eq(labels_t.view(1, -1)) & ~mask
            log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
            positive_counts = positive_mask.sum(dim=1)
            anchors = positive_counts > 0
            if torch.any(anchors):
                mean_log_prob_pos = (
                    positive_mask.float() * log_prob
                ).sum(dim=1) / positive_counts.clamp_min(1)
                return -mean_log_prob_pos[anchors].mean()
        labels = torch.cat([torch.arange(n, 2 * n, device=device), torch.arange(0, n, device=device)])
        return F.cross_entropy(logits, labels)

    @property
    def tau(self) -> float:
        import torch

        return float(torch.exp(-self.logit_scale).item())


def split_input_fn(model_input: List[str], chunk_size: int) -> List[Dict[str, List[str]]]:
    """Split a list of strings into chunked inputs for GradCache."""
    if not isinstance(model_input, list) or not all(isinstance(x, str) for x in model_input):
        raise ValueError("Expected model_input to be a list of strings.")
    return [{"view": model_input[i:i + chunk_size]} for i in range(0, len(model_input), chunk_size)]


def seed_training(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass


def torch_generator(seed: Optional[int]):
    if seed is None:
        return None
    import torch

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


class CAHOTrainer:
    """Trainer for CAHO using a SentenceTransformers model."""

    def __init__(
        self,
        model,
        batch_size: int = CAHO_94GB_ACTUAL_BATCH_SIZE,
        temperature: float = 0.07,
        lr: float = CAHO_DEFAULT_LR,
        weight_decay: float = CAHO_DEFAULT_WEIGHT_DECAY,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize trainer.

        Args:
            model: sentence_transformers.SentenceTransformer
            batch_size: Batch size per step.
            temperature: SupCon temperature.
            lr: AdamW learning rate.
            weight_decay: AdamW weight decay.
            seed: Optional deterministic seed for training order and augmentations.
        """
        self.model = model
        self.batch_size = batch_size
        self.temperature = temperature
        self.lr = _require_finite_positive(lr, "lr")
        self.weight_decay = _require_finite_nonnegative(weight_decay, "weight_decay")
        self.seed = seed

    def fit(self, dataset: CAHODataset, epochs: int = CAHO_DEFAULT_EPOCHS) -> None:
        import torch
        from torch.utils.data import DataLoader

        seed_training(self.seed)
        optim = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.model.train()

        def collate(batch):
            v1 = [b[0] for b in batch]
            v2 = [b[1] for b in batch]
            labels = [b[2] for b in batch]
            return v1, v2, labels

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate,
            generator=torch_generator(self.seed),
        )
        for _epoch in range(epochs):
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(_epoch)
            for v1, v2, labels in loader:
                label_ids: List[int] = []
                family_map: Dict[str, int] = {}
                next_id = 0
                for i, fam in enumerate(labels):
                    if fam is None:
                        label_ids.append(10_000_000 + i)
                    else:
                        if fam not in family_map:
                            family_map[fam] = next_id
                            next_id += 1
                        label_ids.append(family_map[fam])

                feats1 = self.model(self.model.tokenize(v1))["sentence_embedding"]
                feats2 = self.model(self.model.tokenize(v2))["sentence_embedding"]
                loss_t = supervised_orbit_contrastive_loss(
                    feats1,
                    feats2,
                    label_ids,
                    temperature=self.temperature,
                )

                optim.zero_grad()
                loss_t.backward()
                optim.step()


class ContrastiveTrainer:
    """Pairwise contrastive trainer matching the CAHO contrastive training script logic."""

    def __init__(
        self,
        model,
        batch_size: int = CAHO_94GB_ACTUAL_BATCH_SIZE,
        temperature: float = 0.1,
        lr: float = CAHO_DEFAULT_LR,
        weight_decay: float = CAHO_DEFAULT_WEIGHT_DECAY,
        max_grad_norm: float = 1.0,
        scheduler: str = "cosine",
        min_lr: float = 1e-5,
        use_grad_cache: bool = False,
        grad_cache_chunk_size: int = CAHO_94GB_GRAD_CACHE_CHUNK_SIZE,
        num_workers: int = 0,
        empty_cache: bool = False,
        loss_mode: str = "fixed",
        loss_max_scale: float = 100.0,
        loss_min_scale: float = 1.0,
        optimize_loss: bool = False,
        save_best: bool = False,
        save_best_path: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.temperature = temperature
        self.lr = _require_finite_positive(lr, "lr")
        self.weight_decay = _require_finite_nonnegative(weight_decay, "weight_decay")
        self.max_grad_norm = max_grad_norm
        self.scheduler = scheduler
        self.min_lr = min_lr
        self.use_grad_cache = use_grad_cache
        self.grad_cache_chunk_size = grad_cache_chunk_size
        self.num_workers = num_workers
        self.empty_cache = empty_cache
        self.loss_mode = loss_mode
        self.loss_max_scale = loss_max_scale
        self.loss_min_scale = loss_min_scale
        self.optimize_loss = optimize_loss
        self.save_best = save_best
        self.save_best_path = save_best_path
        self.seed = seed
        self._loss_module = None

    def _embed_view(self, view):
        import torch

        device = next(self.model.parameters()).device
        tokenized = self.model.tokenize(
            view,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=64,
        )
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        model_output = self.model(tokenized)
        return model_output["sentence_embedding"]

    def fit(self, dataset: CAHODataset, epochs: int = CAHO_DEFAULT_EPOCHS) -> None:
        import torch
        from torch.utils.data import DataLoader
        from torch.optim.lr_scheduler import CosineAnnealingLR

        seed_training(self.seed)
        loss_params = []
        if self.loss_mode == "learnable":
            self._loss_module = ContrastiveLoss(
                init_tau=self.temperature,
                max_scale=self.loss_max_scale,
                min_scale=self.loss_min_scale,
            )
            if self.optimize_loss:
                loss_params = list(self._loss_module.parameters())

        optim = torch.optim.AdamW(
            list(self.model.parameters()) + loss_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.model.train()

        def collate(batch):
            v1 = [b[0] for b in batch]
            v2 = [b[1] for b in batch]
            return v1, v2

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=self.num_workers,
            generator=torch_generator(self.seed),
        )
        total_steps = len(loader) * max(1, epochs)
        lr_sched = None
        if self.scheduler == "cosine" and total_steps > 0:
            lr_sched = CosineAnnealingLR(optim, T_max=total_steps, eta_min=self.min_lr)

        gc_module = None
        if self.use_grad_cache:
            try:
                from grad_cache import GradCache
            except Exception as exc:
                raise ImportError(
                    "grad-cache is required for --grad-cache training. "
                    "Install with `pip install 'GradCache @ git+https://github.com/luyug/GradCache.git'`."
                ) from exc

            def model_embedding(view):
                return self._embed_view(view)

            if self._loss_module is not None:
                loss_fn = self._loss_module
            else:
                def loss_fn(e1, e2):
                    return pairwise_contrastive_loss(e1, e2, temperature=self.temperature)

            gc_module = GradCache(
                models=[model_embedding, model_embedding],
                chunk_sizes=self.grad_cache_chunk_size,
                loss_fn=loss_fn,
                split_input_fn=split_input_fn,
            )

        if self._loss_module is not None:
            self._loss_module.to(next(self.model.parameters()).device)

        best_loss = float("inf")
        for _epoch in range(epochs):
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(_epoch)
            total_loss = 0.0
            steps = 0
            for v1, v2 in loader:
                optim.zero_grad()
                if gc_module is not None:
                    loss_t = gc_module(v1, v2)
                else:
                    feats1 = self._embed_view(v1)
                    feats2 = self._embed_view(v2)
                    if self._loss_module is not None:
                        loss_t = self._loss_module(feats1, feats2)
                    else:
                        loss_t = pairwise_contrastive_loss(feats1, feats2, temperature=self.temperature)

                if gc_module is None:
                    loss_t.backward()
                if self.max_grad_norm and self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                optim.step()
                if lr_sched is not None:
                    lr_sched.step()
                if self.empty_cache and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                loss_val = float(loss_t.item()) if hasattr(loss_t, "item") else float(loss_t)
                total_loss += loss_val
                steps += 1

            avg_loss = total_loss / max(steps, 1)
            if self.save_best and self.save_best_path is not None and avg_loss < best_loss:
                best_loss = avg_loss
                self.model.save(self.save_best_path)
