from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from .augment import AugmentConfig, CAHOAugmenter, WeightedAugmentConfig
from .benchmark_dataset import (
    BenchmarkFamily,
    BenchmarkLabelMethod,
    BenchmarkTextField,
    HostnameCommandInjectionBenchmarkDataset,
)
from .calibration import split_conformal_threshold_metadata
from .preprocess import normalize_hostname
from .train import (
    ContrastiveLoss,
    pairwise_contrastive_loss,
    seed_training,
    split_input_fn,
    supervised_orbit_contrastive_loss,
    torch_generator,
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
    weight_decay: float = 0.01
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
        weight_decay: float = 0.01,
        use_grad_cache: bool,
        grad_cache_chunk_size: int,
        num_workers: int,
        loss_mode: str,
        loss_max_scale: float,
        loss_min_scale: float,
        optimize_loss: bool,
        log_every: int = 100,
        max_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.temperature = temperature
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.scheduler = scheduler
        self.min_lr = min_lr
        self.weight_decay = float(weight_decay)
        self.use_grad_cache = use_grad_cache
        self.grad_cache_chunk_size = grad_cache_chunk_size
        self.num_workers = num_workers
        self.loss_mode = loss_mode
        self.loss_max_scale = loss_max_scale
        self.loss_min_scale = loss_min_scale
        self.optimize_loss = optimize_loss
        self.log_every = max(0, int(log_every))
        self.max_steps = max_steps
        self.seed = seed
        self._loss_module = None

    def _embed_view(self, view: List[str]):
        device = next(self.model.parameters()).device
        tokenized = self.model.tokenize(view)
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        return self.model(tokenized)["sentence_embedding"]

    def fit(self, dataset: BenchmarkCAHOViewDataset, *, epochs: int = 1) -> Dict[str, Any]:
        import torch
        from torch.optim.lr_scheduler import CosineAnnealingLR
        from torch.utils.data import DataLoader

        seed_training(self.seed)
        loss_params = []
        if self.loss_mode == "learnable":
            self._loss_module = ContrastiveLoss(
                init_tau=self.temperature,
                max_scale=self.loss_max_scale,
                min_scale=self.loss_min_scale,
            )
            self._loss_module.to(next(self.model.parameters()).device)
            if self.optimize_loss:
                loss_params = list(self._loss_module.parameters())

        optim = torch.optim.AdamW(list(self.model.parameters()) + loss_params, lr=self.lr, weight_decay=self.weight_decay)
        sampler = BenchmarkChunkShuffleSampler(dataset, seed=self.seed) if isinstance(dataset, BenchmarkCAHOViewDataset) else None
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            collate_fn=_collate_views,
            num_workers=self.num_workers,
            pin_memory=next(self.model.parameters()).device.type == "cuda",
            generator=torch_generator(self.seed) if sampler is None else None,
        )
        total_steps = len(loader) * max(1, epochs)
        if self.max_steps is not None:
            total_steps = min(total_steps, self.max_steps * max(1, epochs))
        lr_sched = CosineAnnealingLR(optim, T_max=max(total_steps, 1), eta_min=self.min_lr) if self.scheduler == "cosine" else None
        gc_module = self._build_grad_cache() if self.use_grad_cache else None

        self.model.train()
        steps = 0
        total_loss = 0.0
        for _epoch in range(epochs):
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(_epoch)
            for v1, v2, labels in loader:
                optim.zero_grad()
                if gc_module is not None:
                    loss_t = gc_module(v1, v2)
                else:
                    e1 = self._embed_view(v1)
                    e2 = self._embed_view(v2)
                    loss_t = self._contrastive_loss(e1, e2, labels)
                    loss_t.backward()
                if self.max_grad_norm and self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                optim.step()
                if lr_sched is not None:
                    lr_sched.step()
                steps += 1
                total_loss += float(loss_t.item()) if hasattr(loss_t, "item") else float(loss_t)
                if self.log_every and steps % self.log_every == 0:
                    print(
                        f"step={steps} avg_loss={total_loss / max(steps, 1):.6f}",
                        flush=True,
                    )
                if self.max_steps is not None and steps >= self.max_steps:
                    return {"steps": steps, "avg_loss": total_loss / max(steps, 1)}
        return {"steps": steps, "avg_loss": total_loss / max(steps, 1)}

    def _contrastive_loss(self, e1, e2, labels=None):
        if labels is not None:
            labels = _orbit_labels(labels, device=e1.device)
        if self._loss_module is not None:
            return self._loss_module(e1, e2, labels)
        if labels is not None:
            return supervised_orbit_contrastive_loss(
                e1,
                e2,
                labels,
                temperature=self.temperature,
            )
        return pairwise_contrastive_loss(e1, e2, temperature=self.temperature)

    def _build_grad_cache(self):
        try:
            from grad_cache import GradCache
        except Exception as exc:
            raise ImportError(
                "grad-cache is required for --grad-cache training. Install GradCache from "
                "https://github.com/luyug/GradCache before running this mode."
            ) from exc

        def model_embedding(view):
            return self._embed_view(view)

        return GradCache(
            models=[model_embedding, model_embedding],
            chunk_sizes=self.grad_cache_chunk_size,
            loss_fn=self._contrastive_loss,
            split_input_fn=split_input_fn,
        )


class BenchmarkBinaryContrastiveTrainer(BenchmarkContrastiveTrainer):
    """Train CAHO contrastive loss plus a binary classifier over embeddings."""

    def __init__(
        self,
        *args,
        binary_loss_weight: float = 1.0,
        contrastive_loss_weight: float = 1.0,
        binary_hidden_dim: int = 256,
        binary_classifier_path: Optional[str | Path] = None,
        checkpoint_every_steps: int = 0,
        checkpoint_dir: Optional[str | Path] = None,
        checkpoint_config: Optional[BenchmarkTrainingConfig] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.binary_loss_weight = binary_loss_weight
        self.contrastive_loss_weight = contrastive_loss_weight
        self.binary_hidden_dim = int(binary_hidden_dim)
        self.binary_classifier_path = Path(binary_classifier_path) if binary_classifier_path is not None else None
        self.checkpoint_every_steps = max(0, int(checkpoint_every_steps))
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.checkpoint_config = checkpoint_config
        self.best_checkpoint_avg_loss: Optional[float] = None
        self.classifier = None

    def _build_classifier(self, embedding_dim: int):
        import torch

        hidden_dim = max(1, self.binary_hidden_dim)
        return torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def fit(
        self,
        dataset: BenchmarkCAHOViewDataset,
        *,
        epochs: int = 1,
        validation_dataset: Optional[BenchmarkCAHOViewDataset] = None,
        validation_target_fpr: float = 1e-4,
        restore_best_validation: bool = False,
    ) -> Dict[str, Any]:
        import torch
        import torch.nn.functional as F
        from torch.optim.lr_scheduler import CosineAnnealingLR
        from torch.utils.data import DataLoader

        seed_training(self.seed)
        if self.use_grad_cache:
            raise ValueError(
                "GradCache is not supported for BenchmarkBinaryContrastiveTrainer because "
                "the deployed Appendix C objective requires supervised orbit labels in the "
                "contrastive loss. Run the binary trainer without --grad-cache, or use the "
                "regular pairwise CAHO trainer for GradCache experiments."
            )

        emb_dim = int(self.model.get_sentence_embedding_dimension())
        self.classifier = self._build_classifier(emb_dim).to(next(self.model.parameters()).device)
        self._load_binary_classifier_if_requested(emb_dim)

        loss_params = []
        if self.loss_mode == "learnable":
            self._loss_module = ContrastiveLoss(
                init_tau=self.temperature,
                max_scale=self.loss_max_scale,
                min_scale=self.loss_min_scale,
            )
            self._loss_module.to(next(self.model.parameters()).device)
            if self.optimize_loss:
                loss_params = list(self._loss_module.parameters())

        optim = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.classifier.parameters()) + loss_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        sampler = BenchmarkChunkShuffleSampler(dataset, seed=self.seed) if isinstance(dataset, BenchmarkCAHOViewDataset) else None
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            collate_fn=_collate_views,
            num_workers=self.num_workers,
            pin_memory=next(self.model.parameters()).device.type == "cuda",
            generator=torch_generator(self.seed) if sampler is None else None,
        )
        total_steps = len(loader) * max(1, epochs)
        if self.max_steps is not None:
            total_steps = min(total_steps, self.max_steps * max(1, epochs))
        lr_sched = CosineAnnealingLR(optim, T_max=max(total_steps, 1), eta_min=self.min_lr) if self.scheduler == "cosine" else None

        self.model.train()
        self.classifier.train()
        steps = 0
        total_loss = 0.0
        total_contrastive = 0.0
        total_binary = 0.0
        recent_losses = deque(maxlen=self.checkpoint_every_steps or 1)
        validation_history: list[dict[str, Any]] = []
        best_validation: Optional[dict[str, Any]] = None
        best_validation_state: Optional[dict[str, dict[str, Any]]] = None
        stopped_early = False
        for _epoch in range(epochs):
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(_epoch)
            for v1, v2, labels in loader:
                optim.zero_grad()
                e1 = self._embed_view(v1)
                e2 = self._embed_view(v2)
                contrastive_loss = self._contrastive_loss(e1, e2, labels)
                labels_t = labels.to(e1.device, dtype=torch.float32)
                binary_embeddings = torch.cat([F.normalize(e1, dim=1), F.normalize(e2, dim=1)], dim=0)
                binary_labels = labels_t.repeat(2)
                logits = self.classifier(binary_embeddings).squeeze(-1)
                binary_loss = F.binary_cross_entropy_with_logits(logits, binary_labels) * self.binary_loss_weight
                combined_loss = contrastive_loss + binary_loss
                combined_loss.backward()
                loss_for_log = float(combined_loss.item())
                if self.max_grad_norm and self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(list(self.model.parameters()) + list(self.classifier.parameters()), max_norm=self.max_grad_norm)
                optim.step()
                if lr_sched is not None:
                    lr_sched.step()
                steps += 1
                total_loss += loss_for_log
                total_contrastive += _as_float(contrastive_loss)
                total_binary += _as_float(binary_loss)
                recent_losses.append(loss_for_log)
                if self.log_every and steps % self.log_every == 0:
                    print(
                        "step="
                        f"{steps} "
                        f"avg_loss={total_loss / max(steps, 1):.6f} "
                        f"avg_contrastive={total_contrastive / max(steps, 1):.6f} "
                        f"avg_binary={total_binary / max(steps, 1):.6f}",
                        flush=True,
                    )
                if self._should_checkpoint(steps, recent_losses):
                    window_avg_loss = sum(recent_losses) / len(recent_losses)
                    self._save_improved_checkpoint(
                        steps=steps,
                        window_avg_loss=window_avg_loss,
                        train_summary=_training_summary(steps, total_loss, total_contrastive, total_binary),
                    )
                if self.max_steps is not None and steps >= self.max_steps:
                    stopped_early = True
                    break

            if validation_dataset is not None:
                validation_report = self.evaluate_fixed_fpr(
                    validation_dataset,
                    target_fpr=validation_target_fpr,
                )
                validation_report["epoch"] = _epoch + 1
                validation_history.append(validation_report)
                if validation_report.get("status") == "pass":
                    current_metric = float(validation_report["tpr_at_target_fpr"])
                    if best_validation is None or current_metric > float(best_validation["tpr_at_target_fpr"]):
                        best_validation = dict(validation_report)
                        if restore_best_validation:
                            best_validation_state = self._capture_model_selection_state()

            if stopped_early:
                break

        summary: Dict[str, Any] = _training_summary(steps, total_loss, total_contrastive, total_binary)
        if validation_history:
            if restore_best_validation and best_validation_state is not None:
                self._restore_model_selection_state(best_validation_state)
            summary["validation_model_selection"] = {
                "metric": "tpr_at_target_fpr",
                "target_fpr": float(validation_target_fpr),
                "selection_rule": "maximize_tpr_at_target_fpr;ties_keep_earliest",
                "score_source": "binary_auxiliary_head_sigmoid",
                "score_view": "canonical_view1_only",
                "best_epoch": None if best_validation is None else int(best_validation["epoch"]),
                "best_validation_tpr_at_target_fpr": None
                if best_validation is None
                else float(best_validation["tpr_at_target_fpr"]),
                "restored_best_validation_checkpoint": bool(restore_best_validation and best_validation_state is not None),
                "history": validation_history,
            }
        return summary

    def evaluate_fixed_fpr(
        self,
        dataset: BenchmarkCAHOViewDataset,
        *,
        target_fpr: float = 1e-4,
    ) -> dict[str, Any]:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader

        if self.classifier is None:
            raise ValueError("classifier must be initialized before validation evaluation")
        if not 0.0 < float(target_fpr) < 1.0:
            raise ValueError("target_fpr must be in (0, 1)")

        model_was_training = bool(getattr(self.model, "training", False))
        classifier_was_training = bool(getattr(self.classifier, "training", False))
        self.model.eval()
        self.classifier.eval()
        scores: list[float] = []
        labels_out: list[int] = []
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=_collate_views,
            num_workers=self.num_workers,
            pin_memory=next(self.model.parameters()).device.type == "cuda",
        )
        with torch.no_grad():
            for v1, _v2, labels in loader:
                embeddings = F.normalize(self._embed_view(v1), dim=1)
                logits = self.classifier(embeddings).squeeze(-1)
                scores.extend(torch.sigmoid(logits).detach().cpu().numpy().astype(float).tolist())
                labels_out.extend(labels.detach().cpu().numpy().astype(int).tolist())

        if model_was_training:
            self.model.train()
        if classifier_was_training:
            self.classifier.train()

        score_arr = np.asarray(scores, dtype=np.float64)
        label_arr = np.asarray(labels_out, dtype=np.int64)
        benign_scores = score_arr[label_arr == 0]
        positive_scores = score_arr[label_arr == 1]
        if benign_scores.size == 0 or positive_scores.size == 0:
            return {
                "status": "unavailable",
                "reason": "validation set must contain at least one benign and one positive row",
                "target_fpr": float(target_fpr),
                "score_source": "binary_auxiliary_head_sigmoid",
                "score_view": "canonical_view1_only",
                "embedding_normalization": "l2",
                "n_validation_rows": int(score_arr.size),
                "n_validation_benign": int(benign_scores.size),
                "n_validation_positive": int(positive_scores.size),
            }

        calibration_metadata = split_conformal_threshold_metadata(benign_scores, alpha=float(target_fpr))
        threshold = float(calibration_metadata["threshold"])
        predictions = score_arr > threshold
        benign = label_arr == 0
        positive = label_arr == 1
        fp = int(np.sum(predictions & benign))
        tp = int(np.sum(predictions & positive))
        return {
            "status": "pass",
            "target_fpr": float(target_fpr),
            **calibration_metadata,
            "threshold_source": "validation_benign_scores",
            "score_source": "binary_auxiliary_head_sigmoid",
            "score_view": "canonical_view1_only",
            "embedding_normalization": "l2",
            "n_validation_rows": int(score_arr.size),
            "n_validation_benign": int(np.sum(benign)),
            "n_validation_positive": int(np.sum(positive)),
            "false_positives": fp,
            "true_positives": tp,
            "observed_fpr": fp / max(int(np.sum(benign)), 1),
            "tpr_at_target_fpr": tp / max(int(np.sum(positive)), 1),
        }

    def _capture_model_selection_state(self) -> dict[str, dict[str, Any]]:
        assert self.classifier is not None
        return {
            "model": {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()},
            "classifier": {
                key: value.detach().cpu().clone()
                for key, value in self.classifier.state_dict().items()
            },
        }

    def _restore_model_selection_state(self, state: Mapping[str, Mapping[str, Any]]) -> None:
        assert self.classifier is not None
        device = next(self.model.parameters()).device
        self.model.load_state_dict({key: value.to(device) for key, value in state["model"].items()})
        self.classifier.load_state_dict({key: value.to(device) for key, value in state["classifier"].items()})

    def _should_checkpoint(self, steps: int, recent_losses) -> bool:
        return (
            self.checkpoint_every_steps > 0
            and self.checkpoint_dir is not None
            and steps % self.checkpoint_every_steps == 0
            and len(recent_losses) == self.checkpoint_every_steps
        )

    def _save_improved_checkpoint(self, *, steps: int, window_avg_loss: float, train_summary: Mapping[str, Any]) -> None:
        previous_best = self.best_checkpoint_avg_loss
        if previous_best is not None and window_avg_loss >= previous_best:
            print(
                f"step={steps} checkpoint_skipped window_avg_loss={window_avg_loss:.6f} "
                f"best_checkpoint_avg_loss={previous_best:.6f}",
                flush=True,
            )
            return
        self.best_checkpoint_avg_loss = window_avg_loss
        checkpoint_out = self.checkpoint_dir / f"step-{steps:08d}"
        checkpoint_summary = dict(train_summary)
        checkpoint_summary.update(
            {
                "checkpoint_step": steps,
                "checkpoint_window_steps": self.checkpoint_every_steps,
                "checkpoint_window_avg_loss": window_avg_loss,
                "previous_best_checkpoint_avg_loss": previous_best,
            }
        )
        config = self.checkpoint_config or _minimal_checkpoint_config(str(checkpoint_out))
        self.save(checkpoint_out, config, checkpoint_summary)
        print(
            f"step={steps} checkpoint_saved path={checkpoint_out} "
            f"window_avg_loss={window_avg_loss:.6f}",
            flush=True,
        )

    def _contrastive_loss(self, e1, e2, labels=None):
        return super()._contrastive_loss(e1, e2, labels) * self.contrastive_loss_weight

    def _load_binary_classifier_if_requested(self, embedding_dim: int) -> None:
        if self.binary_classifier_path is None:
            return

        import torch

        assert self.classifier is not None
        checkpoint = torch.load(self.binary_classifier_path, map_location=next(self.classifier.parameters()).device)
        checkpoint_embedding_dim = int(checkpoint.get("embedding_dim", embedding_dim))
        checkpoint_hidden_dim = int(checkpoint.get("hidden_dim", self.binary_hidden_dim))
        if checkpoint_embedding_dim != embedding_dim:
            raise ValueError(
                f"Binary classifier embedding_dim mismatch: checkpoint has {checkpoint_embedding_dim}, "
                f"model has {embedding_dim}."
            )
        if checkpoint_hidden_dim != self.binary_hidden_dim:
            raise ValueError(
                f"Binary classifier hidden_dim mismatch: checkpoint has {checkpoint_hidden_dim}, "
                f"configured hidden dim is {self.binary_hidden_dim}."
            )
        self.classifier.load_state_dict(checkpoint["state_dict"])
        print(f"Loaded binary classifier from {self.binary_classifier_path}", flush=True)

    def save(self, out: Path, config: BenchmarkTrainingConfig, train_summary: Mapping[str, Any]) -> None:
        import torch

        assert self.classifier is not None
        out.mkdir(parents=True, exist_ok=True)
        self.model.save(str(out))
        torch.save(
            {
                "state_dict": self.classifier.state_dict(),
                "embedding_dim": int(self.model.get_sentence_embedding_dimension()),
                "hidden_dim": self.binary_hidden_dim,
                "architecture": "linear_relu_linear",
                "label_policy": BenchmarkLabelMethod.ANY_MALICIOUS_ELSE_BENIGN.value,
            },
            out / "binary_classifier.pt",
        )
        _write_report(out, config, train_summary)


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


def _as_float(value: Any) -> float:
    return float(value.item()) if hasattr(value, "item") else float(value)


def _training_summary(steps: int, total_loss: float, total_contrastive: float, total_binary: float) -> Dict[str, float | int]:
    return {
        "steps": steps,
        "avg_loss": total_loss / max(steps, 1),
        "avg_contrastive_loss": total_contrastive / max(steps, 1),
        "avg_binary_loss": total_binary / max(steps, 1),
    }


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
            "grad_cache_boundary": "BenchmarkBinaryContrastiveTrainer fails closed for GradCache because supervised orbit labels must reach the contrastive objective.",
            "deterministic_seed": config.seed,
        },
        "train_summary": dict(train_summary),
    }
    (out / "benchmark_training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def _minimal_checkpoint_config(out: str) -> BenchmarkTrainingConfig:
    return BenchmarkTrainingConfig(
        root="",
        model="",
        out=out,
        epochs=0,
        batch_size=0,
        lr=0.0,
        temperature=0.0,
        max_grad_norm=0.0,
        scheduler="none",
        min_lr=0.0,
        grad_cache=False,
        grad_cache_chunk_size=0,
        num_workers=0,
        device="",
        normalize_text=False,
        augmenter="weighted",
        weighted_num_augs=0,
        weighted_max_attempts=0,
        weighted_retry_on_no_change=False,
        contrastive_loss="fixed",
        contrastive_max_scale=0.0,
        contrastive_min_scale=0.0,
        optimize_contrastive_scale=False,
        weight_decay=0.01,
        checkpoint_every_steps=0,
    )


def save_encoder_only(model, out: Path, config: BenchmarkTrainingConfig, train_summary: Mapping[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    _write_report(out, config, train_summary)
