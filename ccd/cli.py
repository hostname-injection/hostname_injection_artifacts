from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from .config import CCDConfig
from .corpus import (
    filter_hostnames,
    read_hostnames_from_benign_dir,
    read_hostnames_from_jsonl_dir,
    read_hostnames_from_txt_dir,
)
from .csv_io import iter_malicious_csv_rows, read_malicious_csv_map, write_score_csv
from .io import load_model, ModelBundle, save_model
from .line_io import read_nonempty_lines, read_parallel_lines
from .cone import ConePartition
from .encoder import CahoEncoder, require_model_uses_trained_caho_checkpoint, require_trained_caho_checkpoint
from .augment import CAHOAugmenter, AugmentConfig, WeightedAugmentConfig
from .edit_model import EditModel
from .explain import add_explain_arguments, run as run_explain
from .preprocess import normalize_hostname, normalization_trace
from .priors import build_benign_prior, build_malicious_priors
from .train import (
    CAHO_DEFAULT_EPOCHS,
    CAHO_DEFAULT_AUGMENTER,
    CAHO_94GB_GRAD_CACHE_BATCH_SIZE,
    CAHO_94GB_GRAD_CACHE_CHUNK_SIZE,
    CAHO_DEFAULT_LOSS,
    CAHODataset,
    CAHO_DEFAULT_LR,
    CAHO_DEFAULT_WEIGHT_DECAY,
    CAHO_DEFAULT_USE_GRAD_CACHE,
    CAHO_DEFAULT_BINARY_HIDDEN_DIM,
    CAHO_DEFAULT_BINARY_LOSS_WEIGHT,
    CAHO_DEFAULT_CONTRASTIVE_LOSS_WEIGHT,
    CAHO_TRAINING_SETTING_FIELDS,
    ContrastiveTrainer,
    Sample,
    resolve_caho_batch_size,
    training_default_values,
    warn_if_caho_training_defaults_changed,
)
import numpy as np
from .calibration import (
    calibrate_thresholds_by_group,
    require_calibrated_threshold,
    split_conformal_threshold_metadata,
    threshold_for_group,
)


def _read_lines(path: Path) -> List[str]:
    return read_nonempty_lines(path)


def _read_parallel_lines(path: Path, expected_len: int, *, field_name: str) -> List[str]:
    return read_parallel_lines(path, expected_len, field_name)


def _read_malicious_csv(path: Path) -> Dict[str, List[str]]:
    return read_malicious_csv_map(path)


def _apply_normalization(hosts: List[str]) -> List[str]:
    return [normalize_hostname(h) for h in hosts]


def _resolve_device(name: str) -> str:
    if name == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        except Exception:
            return "cpu"
    return name


def _require_model_bundle_caho_checkpoint(model, *, purpose: str) -> None:
    require_model_uses_trained_caho_checkpoint(model, purpose=purpose)


def _require_caho_training_samples(samples: List[Sample], *, source: str) -> None:
    benign_count = sum(1 for sample in samples if not sample.is_malicious)
    malicious_count = sum(1 for sample in samples if sample.is_malicious)
    if benign_count == 0:
        raise ValueError(f"{source} requires at least one benign hostname")
    if malicious_count == 0:
        raise ValueError(f"{source} requires at least one malicious hostname")


def _train_caho_samples(args: argparse.Namespace, samples: List[Sample], out_path: Path) -> None:
    _require_caho_training_samples(samples, source="CAHO training")

    from sentence_transformers import SentenceTransformer

    defaults = getattr(args, "_caho_training_defaults", None)
    if defaults:
        warn_if_caho_training_defaults_changed(
            args,
            defaults=defaults,
            fields=getattr(args, "_caho_training_warning_fields", CAHO_TRAINING_SETTING_FIELDS),
            label=getattr(args, "_caho_training_warning_label", "CAHO training"),
        )

    model_path = args.model
    if args.resume and out_path.exists():
        model_path = str(out_path)

    model = SentenceTransformer(model_path)
    device = _resolve_device(args.device)
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
    augmenter = CAHOAugmenter(config=aug_config)
    dataset = CAHODataset(
        samples,
        augmenter=augmenter,
        include_original=True,
        seed=args.seed,
    )
    batch_size = resolve_caho_batch_size(args.batch_size, use_grad_cache=args.grad_cache)

    trainer = ContrastiveTrainer(
        model,
        batch_size=batch_size,
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
        binary_loss_weight=args.binary_loss_weight,
        contrastive_loss_weight=args.contrastive_loss_weight,
        binary_hidden_dim=args.binary_hidden_dim,
        save_best=args.save_best,
        save_best_path=str(out_path) if args.save_best else None,
        seed=args.seed,
    )
    trainer.fit(dataset, epochs=args.epochs)

    if not args.no_save_final:
        model.save(str(out_path))

def cmd_train_caho(args: argparse.Namespace) -> None:
    benign_hosts = _read_lines(args.benign)
    benign_samples = [Sample(h, is_malicious=False, family=None) for h in benign_hosts]
    malicious_samples = [
        Sample(host, is_malicious=True, family=family)
        for host, family in iter_malicious_csv_rows(args.malicious)
    ]

    samples = benign_samples + malicious_samples
    _require_caho_training_samples(samples, source="ccd train-caho")
    if not args.no_normalize:
        samples = [Sample(normalize_hostname(s.hostname), s.is_malicious, s.family) for s in samples]

    _train_caho_samples(args, samples, args.out)
    print(f"Saved CAHO encoder to {args.out}")


def cmd_train_caho_corpus(args: argparse.Namespace) -> None:
    if not args.benign_dir.is_dir():
        raise ValueError(f"--benign-dir must be an existing directory: {args.benign_dir}")
    if not args.malicious_jsonl_dir.is_dir():
        raise ValueError(f"--malicious-jsonl-dir must be an existing directory: {args.malicious_jsonl_dir}")
    if not args.malicious_txt_dir.is_dir():
        raise ValueError(f"--malicious-txt-dir must be an existing directory: {args.malicious_txt_dir}")

    benign_hosts = read_hostnames_from_benign_dir(args.benign_dir)
    benign_hosts = filter_hostnames(benign_hosts, min_length=args.min_length, dedup=False)

    malicious_jsonl_hosts = read_hostnames_from_jsonl_dir(args.malicious_jsonl_dir, key=args.jsonl_key)
    malicious_txt_hosts = read_hostnames_from_txt_dir(
        args.malicious_txt_dir,
        include_csv=True,
        csv_column=args.csv_hostname_col,
    )
    malicious_hosts = filter_hostnames(
        malicious_jsonl_hosts + malicious_txt_hosts,
        min_length=args.min_length,
        dedup=False,
    )

    if not benign_hosts:
        raise ValueError("No benign hostnames loaded. Check --benign-dir.")
    if not filter_hostnames(malicious_jsonl_hosts, min_length=args.min_length, dedup=False):
        raise ValueError("No JSONL malicious hostnames loaded. Check --malicious-jsonl-dir and --jsonl-key.")
    if not filter_hostnames(malicious_txt_hosts, min_length=args.min_length, dedup=False):
        raise ValueError(
            "No TXT/CSV malicious hostnames loaded. Check --malicious-txt-dir and --csv-hostname-col."
        )

    benign_samples = [Sample(h, is_malicious=False, family=None) for h in benign_hosts]
    malicious_samples = [
        Sample(h, is_malicious=True, family=args.malicious_family) for h in malicious_hosts
    ]
    samples = benign_samples + malicious_samples
    _require_caho_training_samples(samples, source="ccd train-caho-corpus")
    if not args.no_normalize:
        samples = [Sample(normalize_hostname(s.hostname), s.is_malicious, s.family) for s in samples]

    _train_caho_samples(args, samples, args.out)
    print(f"Saved CAHO encoder to {args.out}")


def cmd_eval_caho(args: argparse.Namespace) -> None:
    if not getattr(args, "_allow_test_encoder", False):
        require_trained_caho_checkpoint(args.model, purpose="ccd eval-caho")
    hostnames = _read_lines(args.input)
    if args.normalize:
        hostnames = [normalize_hostname(h) for h in hostnames]

    config = CCDConfig().encoder
    config.model_name = args.model
    config.device = args.device
    encoder = CahoEncoder(config)
    embeddings = encoder.encode(hostnames, batch_size=args.batch_size, normalize=args.embed_normalize)

    fmt = args.format
    if fmt is None:
        if args.output.suffix.lower() == ".csv":
            fmt = "csv"
        else:
            fmt = "npz"

    if fmt == "csv":
        with args.output.open("w") as handle:
            cols = ["hostname"] + [f"e{i}" for i in range(embeddings.shape[1])]
            handle.write(",".join(cols) + "\n")
            for host, vec in zip(hostnames, embeddings):
                row = [host] + [f"{v:.6f}" for v in vec.tolist()]
                handle.write(",".join(row) + "\n")
    else:
        np.savez(args.output, hostnames=np.array(hostnames), embeddings=embeddings)

    print(f"Wrote embeddings to {args.output}")


def cmd_train_priors(args: argparse.Namespace) -> None:
    config = CCDConfig()
    if args.config and args.config.exists():
        config = CCDConfig.from_dict(json.loads(args.config.read_text()))

    checkpoint = require_trained_caho_checkpoint(args.encoder, purpose="ccd train-priors")
    config.encoder.model_name = str(checkpoint)

    encoder = CahoEncoder(config.encoder)
    benign = _read_lines(args.benign)
    malicious = _read_malicious_csv(args.malicious)

    if not args.no_normalize:
        benign = _apply_normalization(benign)
        malicious = {fam: _apply_normalization(hosts) for fam, hosts in malicious.items()}

    benign_emb = encoder.encode(benign, batch_size=args.batch_size)
    mal_emb = {fam: encoder.encode(hosts, batch_size=args.batch_size) for fam, hosts in malicious.items()}

    cones = ConePartition.build(config.cone)
    benign_prior = build_benign_prior(benign_emb, cones, config.prior)
    mal_priors = build_malicious_priors(mal_emb, cones, config.prior)

    bundle = ModelBundle(
        axes=cones.axes,
        benign_prior=benign_prior,
        malicious_priors=mal_priors,
        config=config,
    )
    save_model(args.output, bundle)
    print(f"Saved model to {args.output}")


def cmd_score(args: argparse.Namespace) -> None:
    model = load_model(args.model)
    _require_model_bundle_caho_checkpoint(model, purpose="ccd score")
    hostnames = _read_lines(args.input)
    if not args.no_normalize:
        hostnames = [normalize_hostname(h) for h in hostnames]
    groups = _read_parallel_lines(args.groups, len(hostnames), field_name="groups") if args.groups else None
    threshold = require_calibrated_threshold(model, purpose="ccd score")
    grouped_thresholds = getattr(model, "grouped_thresholds", None)

    scores = model.score(
        hostnames,
        batch_size=args.batch_size,
        normalize=False,
        approximate=args.approximate,
        approximate_k=args.approximate_k,
    )
    if groups is not None:
        row_thresholds = np.array(
            [
                threshold_for_group(
                    group,
                    threshold,
                    grouped_thresholds,
                    missing="error" if args.require_group_thresholds else "default",
                )
                for group in groups
            ],
            dtype=np.float64,
        )
    else:
        row_thresholds = np.full(len(scores), threshold, dtype=np.float64)
    preds = scores > row_thresholds

    write_score_csv(
        args.output,
        hostnames,
        scores,
        preds,
        groups=groups,
        thresholds=row_thresholds,
    )
    print(f"Wrote scores to {args.output}")


def cmd_calibrate(args: argparse.Namespace) -> None:
    model = load_model(args.model)
    _require_model_bundle_caho_checkpoint(model, purpose="ccd calibrate")
    hostnames = _read_lines(args.benign)
    if not args.no_normalize:
        hostnames = [normalize_hostname(h) for h in hostnames]
    groups = _read_parallel_lines(args.groups, len(hostnames), field_name="groups") if args.groups else None

    scores = model.score(
        hostnames,
        batch_size=args.batch_size,
        normalize=False,
        approximate=args.approximate,
        approximate_k=args.approximate_k,
    )
    alpha = args.alpha if args.alpha is not None else model.config.calibration.alpha
    calibration_metadata = split_conformal_threshold_metadata(scores, alpha)
    threshold = calibration_metadata["threshold"]
    grouped_thresholds = calibrate_thresholds_by_group(scores, groups, alpha) if groups is not None else {}
    model.threshold = threshold
    if hasattr(model, "grouped_thresholds"):
        model.grouped_thresholds = grouped_thresholds or None

    output = {
        **calibration_metadata,
        "threshold_source": "grouped_benign_calibration_scores" if groups is not None else "benign_calibration_scores",
        "grouped_thresholds": grouped_thresholds,
        "n_calibration_groups": len(grouped_thresholds),
        "score_path": {
            "approximate": bool(args.approximate),
            "approximate_k": args.approximate_k,
            "normalized_inputs": not args.no_normalize,
        },
    }
    args.output.write_text(json.dumps(output, indent=2))
    save_model(
        args.save_model,
        ModelBundle(
            axes=model.cones.axes,
            benign_prior=model.benign_prior,
            malicious_priors=model.malicious_priors,
            config=model.config,
            threshold=threshold,
            grouped_thresholds=grouped_thresholds or None,
        ),
    )
    print(f"Wrote calibration to {args.output}")


def cmd_refresh_benign(args: argparse.Namespace) -> None:
    model = load_model(args.model)
    _require_model_bundle_caho_checkpoint(model, purpose="ccd refresh-benign")
    require_calibrated_threshold(model, purpose="ccd refresh-benign")
    hostnames = _read_lines(args.benign)
    if not args.no_normalize:
        hostnames = [normalize_hostname(h) for h in hostnames]
    if not hostnames:
        raise ValueError("benign refresh file contains no hostnames")
    groups = _read_parallel_lines(args.groups, len(hostnames), field_name="groups") if args.groups else None

    embeddings = model.encoder.encode(hostnames, batch_size=args.batch_size, normalize=True)
    alpha = args.alpha if args.alpha is not None else model.config.calibration.alpha
    report = model.refresh_benign_reference(
        embeddings,
        alpha=alpha,
        calibration_groups=groups,
        drop_grouped_thresholds=args.drop_grouped_thresholds,
        approximate=args.approximate,
        approximate_k=args.approximate_k,
    )
    report["score_path"]["normalized_inputs"] = not args.no_normalize
    report["input"] = {
        "benign_path": str(args.benign),
        "num_hostnames": len(hostnames),
    }

    save_model(
        args.output,
        ModelBundle(
            axes=model.cones.axes,
            benign_prior=model.benign_prior,
            malicious_priors=model.malicious_priors,
            config=model.config,
            threshold=model.threshold,
            grouped_thresholds=model.grouped_thresholds,
        ),
    )
    if args.report:
        args.report.write_text(json.dumps(report, indent=2))
    print(f"Saved benign-refreshed model to {args.output}")


def cmd_certify(args: argparse.Namespace) -> None:
    model = load_model(args.model)
    _require_model_bundle_caho_checkpoint(model, purpose="ccd certify")
    hostnames = _read_lines(args.input)
    threshold = require_calibrated_threshold(model, purpose="ccd certify")
    threshold_source = "model_bundle_threshold"
    grouped_thresholds = getattr(model, "grouped_thresholds", None)
    grouped_thresholds_source = "model_bundle_grouped_thresholds" if grouped_thresholds else "none"
    groups = _read_parallel_lines(args.groups, len(hostnames), field_name="groups") if args.groups else None

    edit_model = None
    if args.edits:
        edits = [name.strip() for name in args.edits.split(",") if name.strip()]
        edit_model = EditModel(edits=edits)
    edit_manifest = edit_model or EditModel()

    certificates = []
    for index, hostname in enumerate(hostnames):
        row_threshold = threshold
        row_threshold_source = threshold_source
        if groups is not None:
            group_name = str(groups[index]).strip()
            if grouped_thresholds and group_name in grouped_thresholds:
                row_threshold_source = grouped_thresholds_source
            elif not args.require_group_thresholds:
                row_threshold_source = f"{threshold_source}_fallback"
            row_threshold = threshold_for_group(
                group_name,
                threshold,
                grouped_thresholds,
                missing="error" if args.require_group_thresholds else "default",
            )
        cert = model.certify(
            hostname,
            radius=args.radius,
            threshold=row_threshold,
            edit_model=edit_model,
            normalize=not args.no_normalize,
            batch_size=args.batch_size,
            max_nodes=args.max_nodes,
            method=args.cert_method,
            sketch_lipschitz=args.sketch_lipschitz,
            embedding_rotation_bound=args.embedding_rotation_bound,
            eps=args.cert_eps,
        )
        row = asdict(cert)
        row["hostname"] = hostname
        if not args.no_normalize:
            trace = normalization_trace(hostname)
            row["normalized_hostname"] = trace["normalized_hostname"]
            row["normalization_trace"] = trace
        else:
            row["normalized_hostname"] = hostname
            row["normalization_trace"] = {
                "enabled": False,
                "raw_input": hostname,
                "normalized_hostname": hostname,
            }
        row["threshold_source"] = row_threshold_source
        row["decision_rule"] = "score > threshold"
        if groups is not None:
            row["calibration_group"] = groups[index]
        certificates.append(row)

    payload = {
        "radius": args.radius,
        "threshold": threshold,
        "threshold_source": threshold_source,
        "decision_rule": "score > threshold",
        "grouped_thresholds_source": grouped_thresholds_source,
        "grouped_thresholds_used": groups is not None,
        "cert_method": args.cert_method,
        "score_path": {
            "exact_all_cones": True,
            "exact_all_cones_meaning": (
                "all cone axes are scanned exactly before selecting the "
                "deployed top-R cone sketch; scoring still uses the frozen "
                "active-cone statistic"
            ),
            "score_statistic": "deployed_top_r_cone_sketch",
            "active_cones": getattr(getattr(model, "cones", None), "config", None).active_cones
            if getattr(getattr(model, "cones", None), "config", None) is not None
            else None,
            "num_cones": getattr(getattr(model, "cones", None), "config", None).num_cones
            if getattr(getattr(model, "cones", None), "config", None) is not None
            else None,
            "effective_count": getattr(getattr(model, "config", None), "scoring", None).effective_count
            if getattr(getattr(model, "config", None), "scoring", None) is not None
            else None,
            "lsh_bypassed": True,
            "approximate": False,
            "normalized_inputs": not args.no_normalize,
        },
        "normalizer": {
            "enabled": not args.no_normalize,
            "function": "ccd.preprocess.normalize_hostname" if not args.no_normalize else None,
            "unicode_form": "NFKC" if not args.no_normalize else None,
            "decode_percent": True if not args.no_normalize else None,
            "decode_utf8_percent_runs": True if not args.no_normalize else None,
            "idna_roundtrip": True if not args.no_normalize else None,
            "per_certificate_trace": not args.no_normalize,
        },
        "edit_manifest": {
            "version": edit_manifest.version,
            "edits": [op.name for op in edit_manifest.edits],
        },
        "calibrated_margin_bounds": {
            "sketch_lipschitz": args.sketch_lipschitz,
            "embedding_rotation_bound": args.embedding_rotation_bound,
            "eps": args.cert_eps,
        },
        "count": len(certificates),
        "certificates": certificates,
    }
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"Wrote certificates to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ccd")
    sub = parser.add_subparsers(dest="command", required=True)

    train_caho = sub.add_parser("train-caho", help="Fine-tune CAHO encoder")
    train_caho.add_argument("--benign", required=True, type=Path)
    train_caho.add_argument("--malicious", required=True, type=Path)
    train_caho.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    train_caho.add_argument("--out", required=True, type=Path)
    train_caho.add_argument("--epochs", type=int, default=CAHO_DEFAULT_EPOCHS)
    train_caho.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Effective CAHO batch size. Defaults to "
            f"{CAHO_94GB_GRAD_CACHE_BATCH_SIZE} with required GradCache for 94 GB VRAM."
        ),
    )
    train_caho.add_argument("--lr", type=float, default=CAHO_DEFAULT_LR)
    train_caho.add_argument("--weight-decay", type=float, default=CAHO_DEFAULT_WEIGHT_DECAY)
    train_caho.add_argument("--temperature", type=float, default=0.07)
    train_caho.add_argument("--loss", choices=["contrastive"], default=CAHO_DEFAULT_LOSS)
    train_caho.add_argument("--augmenter", choices=["edit", "weighted", "hybrid"], default=CAHO_DEFAULT_AUGMENTER)
    train_caho.add_argument("--weighted-num-augs", type=int, default=2)
    train_caho.add_argument("--weighted-max-attempts", type=int, default=3)
    train_caho.add_argument("--weighted-no-retry", action="store_true")
    train_caho.add_argument("--max-grad-norm", type=float, default=1.0)
    train_caho.add_argument("--scheduler", choices=["cosine", "none"], default="cosine")
    train_caho.add_argument("--min-lr", type=float, default=1e-5)
    train_caho.set_defaults(grad_cache=CAHO_DEFAULT_USE_GRAD_CACHE)
    train_caho.add_argument("--grad-cache-chunk-size", type=int, default=CAHO_94GB_GRAD_CACHE_CHUNK_SIZE)
    train_caho.add_argument("--contrastive-loss", choices=["fixed", "learnable"], default="fixed")
    train_caho.add_argument("--contrastive-max-scale", type=float, default=100.0)
    train_caho.add_argument("--contrastive-min-scale", type=float, default=1.0)
    train_caho.add_argument("--optimize-contrastive-scale", action="store_true")
    train_caho.add_argument("--binary-loss-weight", type=float, default=CAHO_DEFAULT_BINARY_LOSS_WEIGHT)
    train_caho.add_argument("--contrastive-loss-weight", type=float, default=CAHO_DEFAULT_CONTRASTIVE_LOSS_WEIGHT)
    train_caho.add_argument("--binary-hidden-dim", type=int, default=CAHO_DEFAULT_BINARY_HIDDEN_DIM)
    train_caho.add_argument("--num-workers", type=int, default=0)
    train_caho.add_argument("--empty-cache", action="store_true")
    train_caho.add_argument("--device", default="auto", help="Training device: auto|cpu|cuda")
    train_caho.add_argument("--resume", action="store_true", help="Load model from --out if it exists")
    train_caho.add_argument("--save-best", action="store_true")
    train_caho.add_argument("--no-save-final", action="store_true")
    train_caho.add_argument("--no-normalize", action="store_true")
    train_caho.add_argument("--seed", type=int, default=13)
    train_caho.set_defaults(
        func=cmd_train_caho,
        _caho_training_defaults=training_default_values(train_caho, CAHO_TRAINING_SETTING_FIELDS),
        _caho_training_warning_fields=CAHO_TRAINING_SETTING_FIELDS,
        _caho_training_warning_label="ccd train-caho",
    )

    train_caho_corpus = sub.add_parser(
        "train-caho-corpus",
        help="Fine-tune CAHO encoder from corpus directories",
    )
    train_caho_corpus.add_argument("--benign-dir", required=True, type=Path)
    train_caho_corpus.add_argument("--malicious-jsonl-dir", required=True, type=Path)
    train_caho_corpus.add_argument("--malicious-txt-dir", required=True, type=Path)
    train_caho_corpus.add_argument("--jsonl-key", default="hostname")
    train_caho_corpus.add_argument("--csv-hostname-col", default="Hostname")
    train_caho_corpus.add_argument("--min-length", type=int, default=5)
    train_caho_corpus.add_argument("--malicious-family", default="corpus")
    train_caho_corpus.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    train_caho_corpus.add_argument("--out", required=True, type=Path)
    train_caho_corpus.add_argument("--epochs", type=int, default=CAHO_DEFAULT_EPOCHS)
    train_caho_corpus.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Effective CAHO batch size. Defaults to "
            f"{CAHO_94GB_GRAD_CACHE_BATCH_SIZE} with required GradCache for 94 GB VRAM."
        ),
    )
    train_caho_corpus.add_argument("--lr", type=float, default=CAHO_DEFAULT_LR)
    train_caho_corpus.add_argument("--weight-decay", type=float, default=CAHO_DEFAULT_WEIGHT_DECAY)
    train_caho_corpus.add_argument("--temperature", type=float, default=0.07)
    train_caho_corpus.add_argument("--loss", choices=["contrastive"], default=CAHO_DEFAULT_LOSS)
    train_caho_corpus.add_argument("--augmenter", choices=["edit", "weighted", "hybrid"], default=CAHO_DEFAULT_AUGMENTER)
    train_caho_corpus.add_argument("--weighted-num-augs", type=int, default=2)
    train_caho_corpus.add_argument("--weighted-max-attempts", type=int, default=3)
    train_caho_corpus.add_argument("--weighted-no-retry", action="store_true")
    train_caho_corpus.add_argument("--max-grad-norm", type=float, default=1.0)
    train_caho_corpus.add_argument("--scheduler", choices=["cosine", "none"], default="cosine")
    train_caho_corpus.add_argument("--min-lr", type=float, default=1e-5)
    train_caho_corpus.set_defaults(grad_cache=CAHO_DEFAULT_USE_GRAD_CACHE)
    train_caho_corpus.add_argument("--grad-cache-chunk-size", type=int, default=CAHO_94GB_GRAD_CACHE_CHUNK_SIZE)
    train_caho_corpus.add_argument("--contrastive-loss", choices=["fixed", "learnable"], default="fixed")
    train_caho_corpus.add_argument("--contrastive-max-scale", type=float, default=100.0)
    train_caho_corpus.add_argument("--contrastive-min-scale", type=float, default=1.0)
    train_caho_corpus.add_argument("--optimize-contrastive-scale", action="store_true")
    train_caho_corpus.add_argument("--binary-loss-weight", type=float, default=CAHO_DEFAULT_BINARY_LOSS_WEIGHT)
    train_caho_corpus.add_argument("--contrastive-loss-weight", type=float, default=CAHO_DEFAULT_CONTRASTIVE_LOSS_WEIGHT)
    train_caho_corpus.add_argument("--binary-hidden-dim", type=int, default=CAHO_DEFAULT_BINARY_HIDDEN_DIM)
    train_caho_corpus.add_argument("--num-workers", type=int, default=0)
    train_caho_corpus.add_argument("--empty-cache", action="store_true")
    train_caho_corpus.add_argument("--device", default="auto", help="Training device: auto|cpu|cuda")
    train_caho_corpus.add_argument("--resume", action="store_true", help="Load model from --out if it exists")
    train_caho_corpus.add_argument("--save-best", action="store_true")
    train_caho_corpus.add_argument("--no-save-final", action="store_true")
    train_caho_corpus.add_argument("--no-normalize", action="store_true")
    train_caho_corpus.add_argument("--seed", type=int, default=13)
    train_caho_corpus.set_defaults(
        func=cmd_train_caho_corpus,
        _caho_training_defaults=training_default_values(train_caho_corpus, CAHO_TRAINING_SETTING_FIELDS),
        _caho_training_warning_fields=CAHO_TRAINING_SETTING_FIELDS,
        _caho_training_warning_label="ccd train-caho-corpus",
    )

    eval_caho = sub.add_parser("eval-caho", help="Encode hostnames with a CAHO encoder")
    eval_caho.add_argument("--model", required=True, help="Trained CAHO checkpoint directory")
    eval_caho.add_argument("--input", required=True, type=Path)
    eval_caho.add_argument("--output", required=True, type=Path)
    eval_caho.add_argument("--format", choices=["npz", "csv"], default=None)
    eval_caho.add_argument("--batch-size", type=int, default=64)
    eval_caho.add_argument("--device", default="auto", help="Inference device: auto|cpu|cuda|mps")
    eval_caho.add_argument("--normalize", action="store_true", help="Apply hostname normalization")
    eval_caho.add_argument("--embed-normalize", action="store_true", help="Normalize embeddings")
    eval_caho.set_defaults(func=cmd_eval_caho)

    train_priors = sub.add_parser("train-priors", help="Build priors + cone partition")
    train_priors.add_argument("--benign", required=True, type=Path)
    train_priors.add_argument("--malicious", required=True, type=Path)
    train_priors.add_argument("--output", required=True, type=Path)
    train_priors.add_argument("--config", type=Path, default=None)
    train_priors.add_argument("--encoder", type=str, required=True, help="Trained CAHO checkpoint directory")
    train_priors.add_argument("--batch-size", type=int, default=64)
    train_priors.add_argument("--no-normalize", action="store_true")
    train_priors.set_defaults(func=cmd_train_priors)

    score = sub.add_parser("score", help="Score hostnames with CCD model")
    score.add_argument("--model", required=True, type=Path)
    score.add_argument("--input", required=True, type=Path)
    score.add_argument("--output", required=True, type=Path)
    score.add_argument("--groups", type=Path, default=None, help="Optional one-calibration-group-per-input-row file.")
    score.add_argument(
        "--require-group-thresholds",
        action="store_true",
        help="Fail if --groups contains a group missing from grouped calibration output.",
    )
    score.add_argument("--batch-size", type=int, default=64)
    score.add_argument(
        "--approximate",
        action="store_true",
        help="Use fast approximate scoring (hard-cone).",
    )
    score.add_argument(
        "--approximate-k",
        type=int,
        default=None,
        help="Top-k cones to use for approximate scoring (implies --approximate).",
    )
    score.add_argument("--no-normalize", action="store_true")
    score.set_defaults(func=cmd_score)

    explain = sub.add_parser("explain", help="Explain CCD predictions")
    add_explain_arguments(explain)
    explain.set_defaults(func=run_explain)

    calibrate = sub.add_parser("calibrate", help="Calibrate a fixed-FPR threshold")
    calibrate.add_argument("--model", required=True, type=Path)
    calibrate.add_argument("--benign", required=True, type=Path)
    calibrate.add_argument("--output", required=True, type=Path)
    calibrate.add_argument("--groups", type=Path, default=None, help="Optional one-calibration-group-per-benign-row file.")
    calibrate.add_argument(
        "--save-model",
        type=Path,
        required=True,
        help="Path for the calibrated model bundle with the calibrated threshold embedded.",
    )
    calibrate.add_argument("--alpha", type=float, default=None)
    calibrate.add_argument("--batch-size", type=int, default=64)
    calibrate.add_argument(
        "--approximate",
        action="store_true",
        help="Use fast approximate scoring (hard-cone) for calibration.",
    )
    calibrate.add_argument(
        "--approximate-k",
        type=int,
        default=None,
        help="Top-k cones to use for approximate scoring (implies --approximate).",
    )
    calibrate.add_argument("--no-normalize", action="store_true")
    calibrate.set_defaults(func=cmd_calibrate)

    refresh_benign = sub.add_parser(
        "refresh-benign",
        help="Refresh only P_B and the fixed-FPR threshold from a benign window",
    )
    refresh_benign.add_argument("--model", required=True, type=Path)
    refresh_benign.add_argument("--benign", required=True, type=Path)
    refresh_benign.add_argument("--output", required=True, type=Path)
    refresh_benign.add_argument("--report", type=Path, default=None)
    refresh_benign.add_argument("--groups", type=Path, default=None, help="Optional one-calibration-group-per-benign-row file.")
    refresh_benign.add_argument(
        "--drop-grouped-thresholds",
        action="store_true",
        help=(
            "Allow refreshing a model that already has grouped thresholds "
            "without replacement --groups, discarding grouped thresholds and "
            "keeping only the refreshed global threshold."
        ),
    )
    refresh_benign.add_argument("--alpha", type=float, default=None)
    refresh_benign.add_argument("--batch-size", type=int, default=64)
    refresh_benign.add_argument(
        "--approximate",
        action="store_true",
        help="Use fast approximate scoring when recalibrating the refreshed threshold.",
    )
    refresh_benign.add_argument(
        "--approximate-k",
        type=int,
        default=None,
        help="Top-k cones to use for approximate refreshed-threshold scoring.",
    )
    refresh_benign.add_argument("--no-normalize", action="store_true")
    refresh_benign.set_defaults(func=cmd_refresh_benign)

    certify = sub.add_parser("certify", help="Check finite-edit decision stability")
    certify.add_argument("--model", required=True, type=Path)
    certify.add_argument("--input", required=True, type=Path)
    certify.add_argument("--output", required=True, type=Path)
    certify.add_argument("--radius", type=int, required=True)
    certify.add_argument("--groups", type=Path, default=None, help="Optional one-calibration-group-per-input-row file.")
    certify.add_argument(
        "--require-group-thresholds",
        action="store_true",
        help="Fail if --groups contains a group missing from grouped calibration output.",
    )
    certify.add_argument("--edits", type=str, default=None, help="Comma-separated edit manifest subset, e.g. E3_delimiter,E5_case")
    certify.add_argument("--max-nodes", type=int, default=10000)
    certify.add_argument("--batch-size", type=int, default=64)
    certify.add_argument(
        "--cert-method",
        choices=["enumeration", "calibrated-margin", "combined"],
        default="enumeration",
        help="Certificate method. combined tries calibrated-margin first, then deterministic enumeration.",
    )
    certify.add_argument("--sketch-lipschitz", type=float, default=None)
    certify.add_argument("--embedding-rotation-bound", type=float, default=None)
    certify.add_argument("--cert-eps", type=float, default=1e-12)
    certify.add_argument("--no-normalize", action="store_true")
    certify.set_defaults(func=cmd_certify)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
