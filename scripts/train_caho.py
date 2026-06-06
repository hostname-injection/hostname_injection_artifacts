#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from sentence_transformers import SentenceTransformer

from ccd.augment import CAHOAugmenter, AugmentConfig, WeightedAugmentConfig
from ccd.csv_io import iter_malicious_csv_rows
from ccd.preprocess import normalize_hostname
from ccd.train import (
    CAHO_DEFAULT_EPOCHS,
    CAHO_DEFAULT_AUGMENTER,
    CAHO_DEFAULT_LOSS,
    CAHO_DEFAULT_LR,
    CAHO_DEFAULT_USE_GRAD_CACHE,
    CAHO_DEFAULT_WEIGHT_DECAY,
    CAHO_94GB_GRAD_CACHE_BATCH_SIZE,
    CAHO_94GB_GRAD_CACHE_CHUNK_SIZE,
    CAHO_TRAINING_SETTING_FIELDS,
    CAHODataset,
    CAHOTrainer,
    ContrastiveTrainer,
    Sample,
    resolve_caho_batch_size,
    training_default_values,
    warn_if_caho_training_defaults_changed,
)


def read_lines(path: Path):
    return [line.strip() for line in path.read_text(errors="ignore").splitlines() if line.strip()]


def read_malicious_csv(path: Path):
    return [
        Sample(host, is_malicious=True, family=family)
        for host, family in iter_malicious_csv_rows(path)
    ]


def maybe_normalize(samples, enable: bool):
    if not enable:
        return samples
    normed = []
    for s in samples:
        normed.append(Sample(normalize_hostname(s.hostname), s.is_malicious, s.family))
    return normed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benign", required=True, type=Path)
    parser.add_argument("--malicious", required=True, type=Path)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=CAHO_DEFAULT_EPOCHS)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Effective CAHO batch size. Defaults to "
            f"{CAHO_94GB_GRAD_CACHE_BATCH_SIZE} with required GradCache for 94 GB VRAM."
        ),
    )
    parser.add_argument("--lr", type=float, default=CAHO_DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=CAHO_DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--loss", choices=["supcon", "contrastive"], default=CAHO_DEFAULT_LOSS)
    parser.add_argument("--augmenter", choices=["edit", "weighted", "hybrid"], default=CAHO_DEFAULT_AUGMENTER)
    parser.add_argument("--weighted-num-augs", type=int, default=2)
    parser.add_argument("--weighted-max-attempts", type=int, default=3)
    parser.add_argument("--weighted-no-retry", action="store_true")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--scheduler", choices=["cosine", "none"], default="cosine")
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.set_defaults(grad_cache=CAHO_DEFAULT_USE_GRAD_CACHE)
    parser.add_argument("--grad-cache-chunk-size", type=int, default=CAHO_94GB_GRAD_CACHE_CHUNK_SIZE)
    parser.add_argument("--contrastive-loss", choices=["fixed", "learnable"], default="fixed")
    parser.add_argument("--contrastive-max-scale", type=float, default=100.0)
    parser.add_argument("--contrastive-min-scale", type=float, default=1.0)
    parser.add_argument("--optimize-contrastive-scale", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--empty-cache", action="store_true")
    parser.add_argument("--device", default="auto", help="Training device: auto|cpu|cuda")
    parser.add_argument("--resume", action="store_true", help="Load model from --out if it exists")
    parser.add_argument("--save-best", action="store_true")
    parser.add_argument("--no-save-final", action="store_true")
    parser.add_argument("--no-normalize", action="store_true", help="Skip hostname normalization")
    parser.add_argument("--seed", type=int, default=13, help="Deterministic seed for augmentation and training order")
    args = parser.parse_args()
    warn_if_caho_training_defaults_changed(
        args,
        defaults=training_default_values(parser, CAHO_TRAINING_SETTING_FIELDS),
        label="scripts/train_caho.py",
    )
    args.batch_size = resolve_caho_batch_size(args.batch_size, use_grad_cache=args.grad_cache)

    benign_hosts = read_lines(args.benign)
    benign_samples = [Sample(h, is_malicious=False, family=None) for h in benign_hosts]
    malicious_samples = read_malicious_csv(args.malicious)

    samples = benign_samples + malicious_samples
    samples = maybe_normalize(samples, enable=not args.no_normalize)

    model_path = args.model
    if args.resume and args.out.exists():
        model_path = str(args.out)
    model = SentenceTransformer(model_path)
    device = args.device
    if device == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        except Exception:
            device = "cpu"
    if device:
        try:
            model = model.to(device)
        except Exception:
            pass
    normalize_in_aug = not args.no_normalize
    if args.augmenter in {"weighted", "hybrid"}:
        normalize_in_aug = False
    weighted_config = WeightedAugmentConfig(
        num_augs=args.weighted_num_augs,
        retry_on_no_change=not args.weighted_no_retry,
        max_attempts=args.weighted_max_attempts,
    )
    aug_config = AugmentConfig(
        normalize_input=normalize_in_aug,
        use_edit_model=args.augmenter in {"edit", "hybrid"},
        use_weighted_augs=args.augmenter in {"weighted", "hybrid"},
        weighted=weighted_config,
    )
    dataset = CAHODataset(
        samples,
        augmenter=CAHOAugmenter(config=aug_config),
        include_original=args.loss == "contrastive",
        seed=args.seed,
    )
    if args.loss == "contrastive":
        trainer = ContrastiveTrainer(
            model,
            batch_size=args.batch_size,
            temperature=args.temperature,
            lr=args.lr,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            scheduler=args.scheduler,
            min_lr=args.min_lr,
            use_grad_cache=args.grad_cache,
            grad_cache_chunk_size=args.grad_cache_chunk_size,
            num_workers=args.num_workers,
            empty_cache=args.empty_cache,
            loss_mode=args.contrastive_loss,
            loss_max_scale=args.contrastive_max_scale,
            loss_min_scale=args.contrastive_min_scale,
            optimize_loss=args.optimize_contrastive_scale,
            save_best=args.save_best,
            save_best_path=str(args.out) if args.save_best else None,
            seed=args.seed,
        )
    else:
        trainer = CAHOTrainer(
            model,
            batch_size=args.batch_size,
            temperature=args.temperature,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
        )
    trainer.fit(dataset, epochs=args.epochs)

    if not args.no_save_final:
        model.save(str(args.out))
    print(f"Saved CAHO encoder to {args.out}")


if __name__ == "__main__":
    main()
