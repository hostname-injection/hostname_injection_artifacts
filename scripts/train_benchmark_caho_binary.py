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
    BenchmarkBinaryContrastiveTrainer,
    BenchmarkCAHOViewDataset,
    BenchmarkTrainingConfig,
    build_augmenter,
    resolve_device,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CAHO plus a binary classification head over the full benchmark Dataset.")
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("HIB_BENCHMARK_ROOT", "HostnameCommandInjectionBenchmark")))
    parser.add_argument("--model", default="caho_model_checkpoint")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--scheduler", choices=["cosine", "none"], default="cosine")
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--grad-cache", action="store_true")
    parser.add_argument("--grad-cache-chunk-size", type=int, default=128)
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
    parser.add_argument(
        "--binary-classifier",
        type=Path,
        default=None,
        help="Optional binary_classifier.pt checkpoint to load before training.",
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--checkpoint-every-steps", type=int, default=5000)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory for improved intermediate model checkpoints. Defaults to OUT's run directory/checkpoints.",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Debug only: limit rows loaded by the benchmark Dataset.")
    parser.add_argument("--max-steps", type=int, default=None, help="Debug only: stop after this many optimizer steps.")
    return parser


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.grad_cache:
        raise RuntimeError(
            "--grad-cache is not supported by train_benchmark_caho_binary.py because "
            "the Appendix C binary trainer requires supervised orbit labels in the "
            "contrastive loss. Omit --grad-cache for paper-aligned binary training."
        )
    return args


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
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
    )
    summary = trainer.fit(dataset, epochs=args.epochs)
    trainer.save(args.out, config, summary)
    print(f"Saved benchmark CAHO+binary model to {args.out}")
    print(summary)


if __name__ == "__main__":
    main()
