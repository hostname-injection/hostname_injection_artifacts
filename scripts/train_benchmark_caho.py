#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from ccd.benchmark_training import (
    CAHO_94GB_ACTUAL_BATCH_SIZE,
    CAHO_DEFAULT_EPOCHS,
    CAHO_DEFAULT_LR,
    CAHO_DEFAULT_WEIGHT_DECAY,
    CAHO_94GB_GRAD_CACHE_BATCH_SIZE,
    CAHO_94GB_GRAD_CACHE_CHUNK_SIZE,
    BenchmarkCAHOViewDataset,
    BenchmarkContrastiveTrainer,
    BenchmarkTrainingConfig,
    build_augmenter,
    resolve_caho_batch_size,
    resolve_device,
    save_encoder_only,
    training_default_values,
    warn_if_caho_training_defaults_changed,
)


BENCHMARK_CAHO_TRAINING_SETTING_FIELDS = (
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
    "augmenter",
    "weighted_num_augs",
    "weighted_max_attempts",
    "weighted_no_retry",
    "contrastive_loss",
    "contrastive_max_scale",
    "contrastive_min_scale",
    "optimize_contrastive_scale",
    "normalize_text",
    "resume",
    "log_every",
    "seed",
    "max_rows",
    "max_steps",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train regular CAHO over the full benchmark Dataset.")
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("HIB_BENCHMARK_ROOT", "HostnameCommandInjectionBenchmark")))
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=CAHO_DEFAULT_EPOCHS)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Effective contrastive batch size. Defaults to "
            f"{CAHO_94GB_GRAD_CACHE_BATCH_SIZE} with --grad-cache and "
            f"{CAHO_94GB_ACTUAL_BATCH_SIZE} otherwise for 94 GB VRAM."
        ),
    )
    parser.add_argument("--lr", type=float, default=CAHO_DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=CAHO_DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--scheduler", choices=["cosine", "none"], default="cosine")
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--grad-cache", action="store_true")
    parser.add_argument("--grad-cache-chunk-size", type=int, default=CAHO_94GB_GRAD_CACHE_CHUNK_SIZE)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--augmenter", choices=["edit", "weighted", "hybrid"], default="weighted")
    parser.add_argument("--weighted-num-augs", type=int, default=2)
    parser.add_argument("--weighted-max-attempts", type=int, default=3)
    parser.add_argument("--weighted-no-retry", action="store_true")
    parser.add_argument("--contrastive-loss", choices=["fixed", "learnable"], default="learnable")
    parser.add_argument("--contrastive-max-scale", type=float, default=100.0)
    parser.add_argument("--contrastive-min-scale", type=float, default=1.0)
    parser.add_argument("--optimize-contrastive-scale", action="store_true")
    parser.add_argument("--normalize-text", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=13, help="Deterministic seed for augmentation and training order.")
    parser.add_argument("--max-rows", type=int, default=None, help="Debug only: limit rows loaded by the benchmark Dataset.")
    parser.add_argument("--max-steps", type=int, default=None, help="Debug only: stop after this many optimizer steps.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    warn_if_caho_training_defaults_changed(
        args,
        defaults=training_default_values(parser, BENCHMARK_CAHO_TRAINING_SETTING_FIELDS),
        fields=BENCHMARK_CAHO_TRAINING_SETTING_FIELDS,
        label="scripts/train_benchmark_caho.py",
    )
    args.batch_size = resolve_caho_batch_size(args.batch_size, use_grad_cache=args.grad_cache)

    device = resolve_device(args.device)
    if device != "cuda":
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
        log_every=args.log_every,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    trainer = BenchmarkContrastiveTrainer(
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
        seed=args.seed,
    )
    summary = trainer.fit(dataset, epochs=args.epochs)
    save_encoder_only(model, args.out, config, summary)
    print(f"Saved benchmark CAHO encoder to {args.out}")
    print(summary)


if __name__ == "__main__":
    main()
