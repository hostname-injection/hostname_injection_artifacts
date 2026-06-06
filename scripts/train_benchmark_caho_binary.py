#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccd.benchmark_training import (
    CAHO_DEFAULT_EPOCHS,
    CAHO_DEFAULT_LR,
    CAHO_DEFAULT_WEIGHT_DECAY,
    CAHO_94GB_GRAD_CACHE_BATCH_SIZE,
    CAHO_94GB_GRAD_CACHE_CHUNK_SIZE,
    BenchmarkBinaryContrastiveTrainer,
    BenchmarkCAHOViewDataset,
    BenchmarkTrainingConfig,
    build_augmenter,
    resolve_device,
    training_default_values,
    warn_if_caho_training_defaults_changed,
)


BENCHMARK_BINARY_CAHO_TRAINING_SETTING_FIELDS = (
    "model",
    "epochs",
    "batch_size",
    "lr",
    "weight_decay",
    "temperature",
    "max_grad_norm",
    "scheduler",
    "min_lr",
    "grad_cache",
    "grad_cache_chunk_size",
    "num_workers",
    "device",
    "require_cuda",
    "augmenter",
    "weighted_num_augs",
    "weighted_max_attempts",
    "weighted_no_retry",
    "contrastive_loss",
    "contrastive_max_scale",
    "contrastive_min_scale",
    "optimize_contrastive_scale",
    "binary_loss_weight",
    "contrastive_loss_weight",
    "binary_hidden_dim",
    "normalize_text",
    "resume",
    "binary_classifier",
    "log_every",
    "seed",
    "checkpoint_every_steps",
    "validation_root",
    "validation_max_rows",
    "validation_target_fpr",
    "restore_best_validation",
    "max_rows",
    "max_steps",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CAHO plus a binary classification head over the full benchmark Dataset.")
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("HIB_BENCHMARK_ROOT", "HostnameCommandInjectionBenchmark")))
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=CAHO_DEFAULT_EPOCHS)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=CAHO_94GB_GRAD_CACHE_BATCH_SIZE,
        help=(
            "Effective CAHO batch size. Defaults to "
            f"{CAHO_94GB_GRAD_CACHE_BATCH_SIZE} with required GradCache for 94 GB VRAM."
        ),
    )
    parser.add_argument("--lr", type=float, default=CAHO_DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=CAHO_DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--scheduler", choices=["cosine", "none"], default="cosine")
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.set_defaults(
        grad_cache=True,
        grad_cache_chunk_size=CAHO_94GB_GRAD_CACHE_CHUNK_SIZE,
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail instead of falling back to CPU/MPS when reproducing a GPU training run.",
    )
    parser.add_argument("--augmenter", choices=["edit", "weighted", "hybrid"], default="weighted")
    parser.add_argument("--weighted-num-augs", type=int, default=2)
    parser.add_argument("--weighted-max-attempts", type=int, default=3)
    parser.add_argument("--weighted-no-retry", action="store_true")
    parser.add_argument("--contrastive-loss", choices=["fixed", "learnable"], default="learnable")
    parser.add_argument("--contrastive-max-scale", type=float, default=100.0)
    parser.add_argument("--contrastive-min-scale", type=float, default=1.0)
    parser.add_argument("--optimize-contrastive-scale", action="store_true")
    parser.add_argument("--binary-loss-weight", type=float, default=1.0)
    parser.add_argument("--contrastive-loss-weight", type=float, default=1.0)
    parser.add_argument("--binary-hidden-dim", type=int, default=256)
    parser.add_argument("--normalize-text", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.set_defaults(binary_classifier=None)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=13, help="Deterministic seed for augmentation and training order.")
    parser.add_argument("--checkpoint-every-steps", type=int, default=5000)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory for improved intermediate model checkpoints. Defaults to OUT's run directory/checkpoints.",
    )
    parser.add_argument(
        "--validation-root",
        type=Path,
        default=None,
        help="Optional validation-only benchmark root for Appendix C fixed-FPR model selection.",
    )
    parser.add_argument("--validation-max-rows", type=int, default=None, help="Debug only: limit validation rows.")
    parser.add_argument("--validation-target-fpr", type=float, default=1e-4)
    parser.add_argument(
        "--restore-best-validation",
        action="store_true",
        help="Save the epoch with best validation TPR at --validation-target-fpr instead of the final epoch.",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Debug only: limit rows loaded by the benchmark Dataset.")
    parser.add_argument("--max-steps", type=int, default=None, help="Debug only: stop after this many optimizer steps.")
    return parser


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    if not 0.0 < float(args.validation_target_fpr) < 1.0:
        raise RuntimeError("--validation-target-fpr must be in (0, 1).")
    if args.restore_best_validation and args.validation_root is None:
        raise RuntimeError("--restore-best-validation requires --validation-root.")
    return args


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    warn_if_caho_training_defaults_changed(
        args,
        defaults=training_default_values(parser, BENCHMARK_BINARY_CAHO_TRAINING_SETTING_FIELDS),
        fields=BENCHMARK_BINARY_CAHO_TRAINING_SETTING_FIELDS,
        label="scripts/train_benchmark_caho_binary.py",
    )
    validate_args(args)

    device = resolve_device(args.device)
    if args.require_cuda and device != "cuda":
        raise RuntimeError(f"CUDA training was requested, but resolved device is {device!r}.")

    from sentence_transformers import SentenceTransformer

    model_path = str(args.out) if args.resume and args.out.exists() else args.model
    model = SentenceTransformer(model_path).to(device)
    augmenter = build_augmenter(
        mode=args.augmenter,
        normalize_text=args.normalize_text,
        weighted_num_augs=args.weighted_num_augs,
        weighted_max_attempts=args.weighted_max_attempts,
        weighted_retry_on_no_change=not args.weighted_no_retry,
    )
    dataset = BenchmarkCAHOViewDataset(
        args.root,
        normalize_text=args.normalize_text,
        augmenter=augmenter,
        include_original=True,
        max_rows=args.max_rows,
        seed=args.seed,
    )
    validation_dataset = None
    if args.validation_root is not None:
        validation_dataset = BenchmarkCAHOViewDataset(
            args.validation_root,
            normalize_text=args.normalize_text,
            augmenter=augmenter,
            include_original=True,
            max_rows=args.validation_max_rows,
            seed=args.seed,
        )
    config = BenchmarkTrainingConfig(
        root=str(args.root),
        model=args.model,
        out=str(args.out),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        max_grad_norm=args.max_grad_norm,
        scheduler=args.scheduler,
        min_lr=args.min_lr,
        grad_cache=args.grad_cache,
        grad_cache_chunk_size=args.grad_cache_chunk_size,
        num_workers=args.num_workers,
        device=device,
        normalize_text=args.normalize_text,
        augmenter=args.augmenter,
        weighted_num_augs=args.weighted_num_augs,
        weighted_max_attempts=args.weighted_max_attempts,
        weighted_retry_on_no_change=not args.weighted_no_retry,
        contrastive_loss=args.contrastive_loss,
        contrastive_max_scale=args.contrastive_max_scale,
        contrastive_min_scale=args.contrastive_min_scale,
        optimize_contrastive_scale=args.optimize_contrastive_scale,
        binary_loss_weight=args.binary_loss_weight,
        contrastive_loss_weight=args.contrastive_loss_weight,
        binary_hidden_dim=args.binary_hidden_dim,
        log_every=args.log_every,
        max_steps=args.max_steps,
        checkpoint_every_steps=args.checkpoint_every_steps,
        checkpoint_dir=str(args.checkpoint_dir or (args.out.parent / "checkpoints")),
        seed=args.seed,
        validation_root=None if args.validation_root is None else str(args.validation_root),
        validation_target_fpr=args.validation_target_fpr,
        restore_best_validation=args.restore_best_validation,
    )
    trainer = BenchmarkBinaryContrastiveTrainer(
        model,
        batch_size=args.batch_size,
        temperature=args.temperature,
        lr=args.lr,
        max_grad_norm=args.max_grad_norm,
        scheduler=args.scheduler,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        use_grad_cache=args.grad_cache,
        grad_cache_chunk_size=args.grad_cache_chunk_size,
        num_workers=args.num_workers,
        loss_mode=args.contrastive_loss,
        loss_max_scale=args.contrastive_max_scale,
        loss_min_scale=args.contrastive_min_scale,
        optimize_loss=args.optimize_contrastive_scale,
        log_every=args.log_every,
        max_steps=args.max_steps,
        binary_loss_weight=args.binary_loss_weight,
        contrastive_loss_weight=args.contrastive_loss_weight,
        binary_hidden_dim=args.binary_hidden_dim,
        binary_classifier_path=args.binary_classifier,
        checkpoint_every_steps=args.checkpoint_every_steps,
        checkpoint_dir=args.checkpoint_dir or (args.out.parent / "checkpoints"),
        checkpoint_config=config,
        seed=args.seed,
    )
    summary = trainer.fit(
        dataset,
        epochs=args.epochs,
        validation_dataset=validation_dataset,
        validation_target_fpr=args.validation_target_fpr,
        restore_best_validation=args.restore_best_validation,
    )
    trainer.save(args.out, config, summary)
    print(f"Saved benchmark CAHO+binary model to {args.out}")
    print(summary)


if __name__ == "__main__":
    main()
