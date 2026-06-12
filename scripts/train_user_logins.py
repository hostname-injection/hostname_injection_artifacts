#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ccd.config import CCDConfig
from ccd.encoder import CahoEncoder
from ccd.io import save_model
from ccd.augment import CAHOAugmenter, AugmentConfig, WeightedAugmentConfig
from ccd.train import (
    CAHO_94GB_ACTUAL_BATCH_SIZE,
    CAHO_94GB_GRAD_CACHE_BATCH_SIZE,
    CAHO_94GB_GRAD_CACHE_CHUNK_SIZE,
    CAHODataset,
    CAHOTrainer,
    ContrastiveTrainer,
    resolve_caho_batch_size,
)
from ccd.user_logins import (
    DEFAULT_HOSTNAME_COLUMN,
    DEFAULT_USER_LOGINS_COLUMN,
    DEFAULT_OPUS_COLUMN,
    DEFAULT_SONNET_COLUMN,
    DEFAULT_OPUS_CONF_COLUMN,
    DEFAULT_SONNET_CONF_COLUMN,
    LABEL_POLICY_DESCRIPTIONS,
    LabelPolicy,
    build_priors_from_user_logins,
    collect_caho_samples_from_user_logins,
    collect_label_stats_from_user_logins,
)


def _label_policy_help() -> str:
    details = " ".join(f"{name}: {desc}" for name, desc in LABEL_POLICY_DESCRIPTIONS.items())
    return f"How to combine GPT 5.5 / Claude Opus 4.8 labels. {details}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--user-logins-dir",
        type=Path,
        default=Path("hostname_injection_benchmark/user_logins"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--label-policy",
        choices=[policy.value for policy in LabelPolicy],
        default=LabelPolicy.BOTH_M.value,
        help=_label_policy_help(),
    )
    parser.add_argument("--malicious-family", default="dns_cmd_injection")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--encoder", type=str, default=None, help="Override encoder model path/name")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=2048,
        help="Hostnames to buffer per class before encoding",
    )
    parser.add_argument("--no-normalize", action="store_true", help="Skip hostname normalization")
    parser.add_argument("--hostname-col", default=DEFAULT_USER_LOGINS_COLUMN)
    parser.add_argument("--sonnet-col", default=DEFAULT_SONNET_COLUMN)
    parser.add_argument("--opus-col", default=DEFAULT_OPUS_COLUMN)
    parser.add_argument("--sonnet-conf-col", default=DEFAULT_SONNET_CONF_COLUMN)
    parser.add_argument("--opus-conf-col", default=DEFAULT_OPUS_CONF_COLUMN)
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument("--min-sonnet-confidence", type=float, default=None)
    parser.add_argument("--min-opus-confidence", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Only report label counts")
    parser.add_argument("--train-caho", action="store_true", help="Fine-tune CAHO encoder first")
    parser.add_argument("--caho-out", type=Path, default=None)
    parser.add_argument(
        "--caho-model",
        type=str,
        default=None,
        help="Base SentenceTransformer model for CAHO fine-tuning",
    )
    parser.add_argument("--caho-epochs", type=int, default=1)
    parser.add_argument(
        "--caho-batch-size",
        type=int,
        default=None,
        help=(
            "Effective CAHO batch size. Defaults to "
            f"{CAHO_94GB_GRAD_CACHE_BATCH_SIZE} with --caho-grad-cache and "
            f"{CAHO_94GB_ACTUAL_BATCH_SIZE} otherwise for 94 GB VRAM."
        ),
    )
    parser.add_argument("--caho-lr", type=float, default=2e-5)
    parser.add_argument("--caho-temperature", type=float, default=0.07)
    parser.add_argument("--caho-loss", choices=["supcon", "contrastive"], default="supcon")
    parser.add_argument("--caho-augmenter", choices=["edit", "weighted", "hybrid"], default="edit")
    parser.add_argument("--caho-weighted-num-augs", type=int, default=2)
    parser.add_argument("--caho-weighted-max-attempts", type=int, default=3)
    parser.add_argument("--caho-weighted-no-retry", action="store_true")
    parser.add_argument("--caho-max-grad-norm", type=float, default=1.0)
    parser.add_argument("--caho-scheduler", choices=["cosine", "none"], default="cosine")
    parser.add_argument("--caho-min-lr", type=float, default=1e-5)
    parser.add_argument("--caho-grad-cache", action="store_true", help="Enable GradCache for CAHO training")
    parser.add_argument("--caho-grad-cache-chunk-size", type=int, default=CAHO_94GB_GRAD_CACHE_CHUNK_SIZE)
    parser.add_argument("--caho-contrastive-loss", choices=["fixed", "learnable"], default="fixed")
    parser.add_argument("--caho-contrastive-max-scale", type=float, default=100.0)
    parser.add_argument("--caho-contrastive-min-scale", type=float, default=1.0)
    parser.add_argument("--caho-optimize-contrastive-scale", action="store_true")
    parser.add_argument("--caho-num-workers", type=int, default=0)
    parser.add_argument("--caho-empty-cache", action="store_true")
    parser.add_argument("--caho-device", default="auto", help="Training device: auto|cpu|cuda")
    parser.add_argument("--caho-resume", action="store_true")
    parser.add_argument("--caho-save-best", action="store_true")
    parser.add_argument("--caho-no-save-final", action="store_true")
    parser.add_argument(
        "--caho-sample",
        type=int,
        default=None,
        help="Reservoir sample size per class for CAHO training",
    )
    parser.add_argument("--caho-seed", type=int, default=13)
    args = parser.parse_args()

    config = CCDConfig()
    if args.config and args.config.exists():
        config = CCDConfig.from_dict(json.loads(args.config.read_text()))

    if args.train_caho and args.encoder:
        raise ValueError("Use either --train-caho or --encoder, not both.")

    if args.encoder:
        config.encoder.model_name = args.encoder

    min_sonnet_conf = args.min_sonnet_confidence
    min_opus_conf = args.min_opus_confidence
    if args.min_confidence is not None:
        if min_sonnet_conf is None:
            min_sonnet_conf = args.min_confidence
        if min_opus_conf is None:
            min_opus_conf = args.min_confidence

    if args.dry_run:
        stats = collect_label_stats_from_user_logins(
            args.user_logins_dir,
            label_policy=LabelPolicy(args.label_policy),
            min_sonnet_confidence=min_sonnet_conf,
            min_opus_confidence=min_opus_conf,
            normalize=not args.no_normalize,
            hostname_col=args.hostname_col,
            sonnet_col=args.sonnet_col,
            opus_col=args.opus_col,
            sonnet_conf_col=args.sonnet_conf_col,
            opus_conf_col=args.opus_conf_col,
        )
        print(
            "Rows: "
            f"total={stats.total_rows}, "
            f"benign={stats.used_benign}, "
            f"malicious={stats.used_malicious}, "
            f"dropped={stats.dropped_rows}"
        )
        combos = ", ".join(f"{k}:{v}" for k, v in sorted(stats.combo_counts.items()))
        if combos:
            print(f"Label combos (sonnet/opus): {combos}")
        return

    if args.output is None:
        raise ValueError("--output is required unless --dry-run is set.")

    label_policy = LabelPolicy(args.label_policy)
    if args.train_caho:
        from sentence_transformers import SentenceTransformer

        caho_out = args.caho_out or Path("caho_encoder_user_logins")
        base_model = args.caho_model or config.encoder.model_name
        samples, sample_stats = collect_caho_samples_from_user_logins(
            args.user_logins_dir,
            label_policy=label_policy,
            min_sonnet_confidence=min_sonnet_conf,
            min_opus_confidence=min_opus_conf,
            normalize=not args.no_normalize,
            malicious_family=args.malicious_family,
            hostname_col=args.hostname_col,
            sonnet_col=args.sonnet_col,
            opus_col=args.opus_col,
            sonnet_conf_col=args.sonnet_conf_col,
            opus_conf_col=args.opus_conf_col,
            sample_per_class=args.caho_sample,
            seed=args.caho_seed,
        )
        if not samples:
            raise ValueError("No samples available for CAHO training after filtering.")

        if args.caho_resume and caho_out.exists():
            base_model = str(caho_out)
        model = SentenceTransformer(base_model)
        device = args.caho_device
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
        if args.caho_augmenter in {"weighted", "hybrid"}:
            normalize_in_aug = False
        weighted_config = WeightedAugmentConfig(
            num_augs=args.caho_weighted_num_augs,
            retry_on_no_change=not args.caho_weighted_no_retry,
            max_attempts=args.caho_weighted_max_attempts,
        )
        aug_config = AugmentConfig(
            normalize_input=normalize_in_aug,
            use_edit_model=args.caho_augmenter in {"edit", "hybrid"},
            use_weighted_augs=args.caho_augmenter in {"weighted", "hybrid"},
            weighted=weighted_config,
        )
        dataset = CAHODataset(
            samples,
            augmenter=CAHOAugmenter(config=aug_config),
            include_original=args.caho_loss == "contrastive",
            seed=args.caho_seed,
        )
        caho_batch_size = resolve_caho_batch_size(
            args.caho_batch_size,
            use_grad_cache=args.caho_grad_cache,
        )
        if args.caho_loss == "contrastive":
            trainer = ContrastiveTrainer(
                model,
                batch_size=caho_batch_size,
                temperature=args.caho_temperature,
                lr=args.caho_lr,
                max_grad_norm=args.caho_max_grad_norm,
                scheduler=args.caho_scheduler,
                min_lr=args.caho_min_lr,
                use_grad_cache=args.caho_grad_cache,
                grad_cache_chunk_size=args.caho_grad_cache_chunk_size,
                num_workers=args.caho_num_workers,
                empty_cache=args.caho_empty_cache,
                loss_mode=args.caho_contrastive_loss,
                loss_max_scale=args.caho_contrastive_max_scale,
                loss_min_scale=args.caho_contrastive_min_scale,
                optimize_loss=args.caho_optimize_contrastive_scale,
                save_best=args.caho_save_best,
                save_best_path=str(caho_out) if args.caho_save_best else None,
                seed=args.caho_seed,
            )
        else:
            trainer = CAHOTrainer(
                model,
                batch_size=caho_batch_size,
                temperature=args.caho_temperature,
                lr=args.caho_lr,
                seed=args.caho_seed,
            )
        trainer.fit(dataset, epochs=args.caho_epochs)
        if not args.caho_no_save_final:
            model.save(str(caho_out))
        print(
            "CAHO samples: "
            f"benign={sample_stats.used_benign}, "
            f"malicious={sample_stats.used_malicious}, "
            f"dropped={sample_stats.dropped_rows}"
        )
        print(f"Saved CAHO encoder to {caho_out}")
        config.encoder.model_name = str(caho_out)

    encoder = CahoEncoder(config.encoder)
    label_policy = LabelPolicy(args.label_policy)
    bundle, stats = build_priors_from_user_logins(
        args.user_logins_dir,
        config,
        encoder=encoder,
        label_policy=label_policy,
        min_sonnet_confidence=min_sonnet_conf,
        min_opus_confidence=min_opus_conf,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        normalize=not args.no_normalize,
        malicious_family=args.malicious_family,
        hostname_col=args.hostname_col,
        sonnet_col=args.sonnet_col,
        opus_col=args.opus_col,
        sonnet_conf_col=args.sonnet_conf_col,
        opus_conf_col=args.opus_conf_col,
    )

    save_model(args.output, bundle)
    print(f"Saved model to {args.output}")
    print(
        "Rows: "
        f"total={stats.total_rows}, "
        f"benign={stats.used_benign}, "
        f"malicious={stats.used_malicious}, "
        f"dropped={stats.dropped_rows}"
    )
    combos = ", ".join(f"{k}:{v}" for k, v in sorted(stats.combo_counts.items()))
    if combos:
        print(f"Label combos (sonnet/opus): {combos}")


if __name__ == "__main__":
    main()
