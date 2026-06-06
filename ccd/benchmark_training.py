from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .augment import AugmentConfig, CAHOAugmenter, WeightedAugmentConfig
from .benchmark_dataset import (
    BenchmarkFamily,
    BenchmarkLabelMethod,
    BenchmarkTextField,
    HostnameCommandInjectionBenchmarkDataset,
)
from .preprocess import normalize_hostname
from .train import (
    CAHO_94GB_GRAD_CACHE_BATCH_SIZE,
    CAHO_94GB_GRAD_CACHE_CHUNK_SIZE,
    CAHO_DEFAULT_AUGMENTER,
    CAHO_DEFAULT_EPOCHS,
    CAHO_DEFAULT_LR,
    CAHO_DEFAULT_USE_GRAD_CACHE,
    CAHO_DEFAULT_WEIGHT_DECAY,
    CAHO_DEFAULT_BINARY_HIDDEN_DIM,
    CAHO_DEFAULT_BINARY_LOSS_WEIGHT,
    CAHO_DEFAULT_CONTRASTIVE_LOSS_WEIGHT,
    ContrastiveLoss,
    _model_embedding_dimension,
    pairwise_contrastive_loss,
    resolve_caho_batch_size,
    seed_training,
    split_input_fn,
    supervised_orbit_contrastive_loss,
    torch_generator,
    training_default_values,
    warn_if_caho_training_defaults_changed,
)


@dataclass(frozen=True)
class BenchmarkTrainingConfig:
    root: str
    model: str
    out: str
    epochs: int
    batch_size: int
    lr: float
    temperature: float
    max_grad_norm: float
    scheduler: str
    min_lr: float
    grad_cache: bool
    grad_cache_chunk_size: int
    num_workers: int
    device: str
    normalize_text: bool
    augmenter: str
    weighted_num_augs: int
    weighted_max_attempts: int
    weighted_retry_on_no_change: bool
    contrastive_loss: str
    contrastive_max_scale: float
    contrastive_min_scale: float
    optimize_contrastive_scale: bool
    binary_loss_weight: float = 1.0
    contrastive_loss_weight: float = 1.0
    binary_hidden_dim: int = 256
    weight_decay: float = CAHO_DEFAULT_WEIGHT_DECAY
    log_every: int = 100
    max_steps: Optional[int] = None
    checkpoint_every_steps: int = 5000
    checkpoint_dir: Optional[str] = None
    seed: Optional[int] = 13
    validation_root: Optional[str] = None
    validation_target_fpr: float = 1e-4
    restore_best_validation: bool = False


class BenchmarkCAHOViewDataset:
    """CAHO two-view dataset backed by the external benchmark CSV chunks.

    This wrapper intentionally keeps the benchmark map-style Dataset underneath
    so rows are read chunk-by-chunk. It does not materialize all hostnames.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        label_method: BenchmarkLabelMethod | str = BenchmarkLabelMethod.ANY_MALICIOUS_ELSE_BENIGN,
        normalize_text: bool = False,
        augmenter: Optional[CAHOAugmenter] = None,
        include_original: bool = True,
        max_rows: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.base = HostnameCommandInjectionBenchmarkDataset(
            root,
            family=BenchmarkFamily.BOTH,
            label_method=label_method,
            drop_unknown=False,
            include_explanations=False,
            include_metadata=False,
            return_dict=True,
            text_field=BenchmarkTextField.AUTO,
            normalize_text=normalize_text,
            max_rows=max_rows,
            cache_chunks=1,
        )
        self.augmenter = augmenter or CAHOAugmenter()
        self.include_original = include_original
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.base)

    @property
    def stats(self):
        return self.base.stats

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rng_for_index(self, idx: int) -> random.Random:
        if self.seed is None:
            return random.Random()
        epoch_stride = max(1, len(self))
        return random.Random(int(self.seed) + self.epoch * epoch_stride + int(idx))

    def __getitem__(self, idx: int) -> Tuple[str, str, int]:
        item = self.base[idx]
        text = str(item["text"])
        label = int(item["label"])
        is_malicious = label == 1
        rng = self._rng_for_index(idx)
        if self.include_original:
            view1 = text
            view2 = self.augmenter.augment(text, is_malicious=is_malicious, rng=rng)
        else:
            view1 = self.augmenter.augment(text, is_malicious=is_malicious, rng=rng)
            view2 = self.augmenter.augment(text, is_malicious=is_malicious, rng=rng)
        return view1, view2, label


class BenchmarkChunkShuffleSampler:
    """Shuffle benchmark chunks while keeping row access mostly chunk-local."""

    def __init__(self, dataset: BenchmarkCAHOViewDataset, *, seed: Optional[int] = None) -> None:
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0
        self.groups = self._build_groups(dataset)

    def __iter__(self):
        rng = random.Random(None if self.seed is None else self.seed + self.epoch)
        self.epoch += 1
        groups = list(self.groups)
        rng.shuffle(groups)
        for group in groups:
            if isinstance(group, tuple):
                start, end = group
                indices = list(range(start, end))
            else:
                indices = list(group)
            rng.shuffle(indices)
            yield from indices

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def _build_groups(dataset: BenchmarkCAHOViewDataset) -> List[Tuple[int, int] | List[int]]:
        base = dataset.base
        selected = getattr(base, "_selected", None)
        if selected is not None:
            grouped: Dict[int, List[int]] = {}
            for dataset_index, (chunk_index, _row_index) in enumerate(selected):
                grouped.setdefault(int(chunk_index), []).append(dataset_index)
            return [group for _chunk_index, group in sorted(grouped.items()) if group]

        groups: List[Tuple[int, int] | List[int]] = []
        start = 0
        remaining = len(dataset)
        for _family, _path, rows in getattr(base, "_chunks", []):
            if remaining <= 0:
                break
            end = min(start + int(rows), len(dataset))
            if end > start:
                groups.append((start, end))
            remaining -= max(0, end - start)
            start = end
        if not groups and len(dataset):
            groups.append((0, len(dataset)))
        return groups


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _require_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _require_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _require_finite_positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _require_finite_nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def build_augmenter(
    *,
    mode: str,
    normalize_text: bool,
    weighted_num_augs: int,
    weighted_max_attempts: int,
    weighted_retry_on_no_change: bool,
) -> CAHOAugmenter:
    normalize_in_aug = normalize_text
    if mode in {"weighted", "hybrid"}:
        normalize_in_aug = False
    weighted_config = WeightedAugmentConfig(
        num_augs=weighted_num_augs,
        retry_on_no_change=weighted_retry_on_no_change,
        max_attempts=weighted_max_attempts,
    )
    aug_config = AugmentConfig(
        normalize_input=normalize_in_aug,
        use_edit_model=mode in {"edit", "hybrid"},
        use_weighted_augs=mode in {"weighted", "hybrid"},
        weighted=weighted_config,
    )
    return CAHOAugmenter(config=aug_config)


class BenchmarkContrastiveTrainer:
    """Regular CAHO contrastive trainer over benchmark-backed two-view rows."""

    def __init__(
        self,
        model,
        *,
        batch_size: int,
        temperature: float,
        lr: float,
        max_grad_norm: float,
        scheduler: str,
        min_lr: float,
        weight_decay: float = CAHO_DEFAULT_WEIGHT_DECAY,
        use_grad_cache: bool,
        grad_cache_chunk_size: int,
        num_workers: int,
        loss_mode: str,
        loss_max_scale: float,
        loss_min_scale: float,
        optimize_loss: bool,
        binary_loss_weight: float = CAHO_DEFAULT_BINARY_LOSS_WEIGHT,
        contrastive_loss_weight: float = CAHO_DEFAULT_CONTRASTIVE_LOSS_WEIGHT,
        binary_hidden_dim: int = CAHO_DEFAULT_BINARY_HIDDEN_DIM,
        log_every: int = 100,
        max_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.model = model
        self.batch_size = _require_positive_int(batch_size, "batch_size")
        self.temperature = _require_finite_positive(temperature, "temperature")
        self.lr = _require_finite_positive(lr, "lr")
        self.max_grad_norm = _require_finite_nonnegative(max_grad_norm, "max_grad_norm")
        self.scheduler = scheduler
        if self.scheduler not in {"cosine", "none"}:
            raise ValueError("scheduler must be 'cosine' or 'none'")
        self.min_lr = _require_finite_nonnegative(min_lr, "min_lr")
        self.weight_decay = _require_finite_nonnegative(weight_decay, "weight_decay")
        self.use_grad_cache = use_grad_cache
        self.grad_cache_chunk_size = _require_positive_int(grad_cache_chunk_size, "grad_cache_chunk_size")
        self.num_workers = _require_nonnegative_int(num_workers, "num_workers")
        self.loss_mode = loss_mode
        if self.loss_mode not in {"fixed", "learnable"}:
            raise ValueError("loss_mode must be 'fixed' or 'learnable'")
        self.loss_max_scale = _require_finite_positive(loss_max_scale, "loss_max_scale")
        self.loss_min_scale = _require_finite_positive(loss_min_scale, "loss_min_scale")
        if self.loss_max_scale < self.loss_min_scale:
            raise ValueError("loss_max_scale must be greater than or equal to loss_min_scale")
        self.optimize_loss = optimize_loss
        self.binary_loss_weight = _require_finite_positive(binary_loss_weight, "binary_loss_weight")
        self.contrastive_loss_weight = _require_finite_positive(
            contrastive_loss_weight,
            "contrastive_loss_weight",
        )
        self.binary_hidden_dim = _require_positive_int(binary_hidden_dim, "binary_hidden_dim")
        self.log_every = _require_nonnegative_int(log_every, "log_every")
        self.max_steps = None if max_steps is None else _require_positive_int(max_steps, "max_steps")
        self.seed = seed
        self._loss_module = None
        self.classifier = None

    def _embed_view(self, view: List[str]):
        device = next(self.model.parameters()).device
        tokenized = self.model.tokenize(view)
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        return self.model(tokenized)["sentence_embedding"]

    def _build_classifier(self, embedding_dim: int):
        import torch

        return torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, self.binary_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.binary_hidden_dim, 1),
        )

    def fit(self, dataset: BenchmarkCAHOViewDataset, *, epochs: int = CAHO_DEFAULT_EPOCHS) -> Dict[str, Any]:
        import torch
        from torch.optim.lr_scheduler import CosineAnnealingLR
        from torch.utils.data import DataLoader

        seed_training(self.seed)
        model_params = list(self.model.parameters())
        if not model_params:
            raise ValueError("CAHO model must expose trainable parameters")
        device = model_params[0].device
        self.classifier = self._build_classifier(_model_embedding_dimension(self.model)).to(device)
        classifier_params = list(self.classifier.parameters())

        loss_params = []
        if self.loss_mode == "learnable":
            self._loss_module = ContrastiveLoss(
                init_tau=self.temperature,
                max_scale=self.loss_max_scale,
                min_scale=self.loss_min_scale,
            )
            self._loss_module.to(device)
            if self.optimize_loss:
                loss_params = list(self._loss_module.parameters())

        trainable_params = model_params + classifier_params + loss_params
        optim = torch.optim.AdamW(trainable_params, lr=self.lr, weight_decay=self.weight_decay)
        sampler = BenchmarkChunkShuffleSampler(dataset, seed=self.seed) if isinstance(dataset, BenchmarkCAHOViewDataset) else None
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            collate_fn=_collate_views,
            num_workers=self.num_workers,
            pin_memory=device.type == "cuda",
            generator=torch_generator(self.seed) if sampler is None else None,
        )
        total_steps = len(loader) * max(1, epochs)
        if self.max_steps is not None:
            total_steps = min(total_steps, self.max_steps * max(1, epochs))
        lr_sched = CosineAnnealingLR(optim, T_max=max(total_steps, 1), eta_min=self.min_lr) if self.scheduler == "cosine" else None
        gc_module = self._build_grad_cache() if self.use_grad_cache else None

        self.model.train()
        self.classifier.train()
        steps = 0
        total_loss = 0.0
        for _epoch in range(epochs):
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(_epoch)
            for v1, v2, labels in loader:
                optim.zero_grad()
                if gc_module is not None:
                    loss_t = gc_module(v1, v2, labels=labels)
                else:
                    e1 = self._embed_view(v1)
                    e2 = self._embed_view(v2)
                    loss_t = self._training_loss(e1, e2, labels)
                    loss_t.backward()
                if self.max_grad_norm and self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=self.max_grad_norm)
                optim.step()
                if lr_sched is not None:
                    lr_sched.step()
                steps += 1
                loss_value = float(loss_t.item()) if hasattr(loss_t, "item") else float(loss_t)
                total_loss += loss_value
                if self.log_every and steps % self.log_every == 0:
                    print(
                        f"step={steps} avg_loss={total_loss / max(steps, 1):.6f}",
                        flush=True,
                    )
                if self.max_steps is not None and steps >= self.max_steps:
                    return {
                        "steps": steps,
                        "avg_loss": total_loss / max(steps, 1),
                    }
        return {
            "steps": steps,
            "avg_loss": total_loss / max(steps, 1),
        }

    def _contrastive_loss(self, e1, e2, labels=None):
        if labels is not None:
            labels = _orbit_labels(labels, device=e1.device)
        if self._loss_module is not None:
            return self._loss_module(e1, e2, labels) * self.contrastive_loss_weight
        if labels is not None:
            return supervised_orbit_contrastive_loss(
                e1,
                e2,
                labels,
                temperature=self.temperature,
            ) * self.contrastive_loss_weight
        return pairwise_contrastive_loss(e1, e2, temperature=self.temperature) * self.contrastive_loss_weight

    def _binary_auxiliary_loss(self, e1, e2, labels):
        if self.classifier is None:
            raise ValueError("binary auxiliary classifier is not initialized")

        import torch
        import torch.nn.functional as F

        labels_t = torch.as_tensor(labels, device=e1.device, dtype=torch.float32).view(-1)
        binary_embeddings = torch.cat([F.normalize(e1, dim=1), F.normalize(e2, dim=1)], dim=0)
        logits = self.classifier(binary_embeddings).squeeze(-1)
        return F.binary_cross_entropy_with_logits(logits, labels_t.repeat(2)) * self.binary_loss_weight

    def _training_loss(self, e1, e2, labels=None):
        loss_t = self._contrastive_loss(e1, e2, labels)
        if labels is not None:
            loss_t = loss_t + self._binary_auxiliary_loss(e1, e2, labels)
        return loss_t

    def _build_grad_cache(self):
        try:
            from grad_cache import GradCache
        except Exception as exc:
            raise ImportError(
                "GradCache is required for CAHO training. Install GradCache from "
                "https://github.com/luyug/GradCache before running this mode."
            ) from exc

        def model_embedding(view):
            return self._embed_view(view)

        return GradCache(
            models=[model_embedding, model_embedding],
            chunk_sizes=self.grad_cache_chunk_size,
            loss_fn=self._training_loss,
            split_input_fn=split_input_fn,
        )


def _collate_views(batch):
    import torch

    v1 = [b[0] for b in batch]
    v2 = [b[1] for b in batch]
    labels = torch.tensor([int(b[2]) for b in batch], dtype=torch.long)
    return v1, v2, labels


def _orbit_labels(labels, *, device=None):
    import torch

    labels = torch.as_tensor(labels, dtype=torch.long, device=device)
    orbit = labels.clone()
    benign = labels != 1
    if torch.any(benign):
        unique_benign = -torch.arange(1, int(benign.sum().item()) + 1, device=labels.device, dtype=torch.long)
        orbit[benign] = unique_benign
    return orbit


def _write_report(out: Path, config: BenchmarkTrainingConfig, train_summary: Mapping[str, Any]) -> None:
    report = {
        "config": asdict(config),
        "label_policy": {
            "method": BenchmarkLabelMethod.ANY_MALICIOUS_ELSE_BENIGN.value,
            "meaning": "M if GPT-5.5 or Claude Opus 4.8 is M; otherwise B, including U/U.",
        },
        "contrastive_objective": {
            "name": "supervised_orbit_contrastive_loss",
            "benign_orbits": "unique_per_batch_item",
            "positive_orbits": "shared_positive_label_when_family_labels_are_unavailable",
            "binary_auxiliary_head_views": "both_l2_normalized_views",
            "grad_cache_boundary": "GradCache carries supervised orbit labels and binary labels through the training loss.",
            "deterministic_seed": config.seed,
        },
        "train_summary": dict(train_summary),
    }
    (out / "benchmark_training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

def save_encoder_only(model, out: Path, config: BenchmarkTrainingConfig, train_summary: Mapping[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    _write_report(out, config, train_summary)
