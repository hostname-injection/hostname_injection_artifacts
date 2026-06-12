from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from sentence_transformers import SentenceTransformer

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
from .encoder import CahoEncoder
from .augment import CAHOAugmenter, AugmentConfig, WeightedAugmentConfig
from .edit_model import EditModel
from .preprocess import normalize_hostname, normalization_trace
from .priors import build_benign_prior, build_malicious_priors
from .train import (
    CAHO_94GB_ACTUAL_BATCH_SIZE,
    CAHO_94GB_GRAD_CACHE_BATCH_SIZE,
    CAHO_94GB_GRAD_CACHE_CHUNK_SIZE,
    CAHODataset,
    CAHOTrainer,
    ContrastiveTrainer,
    Sample,
    resolve_caho_batch_size,
)
import numpy as np
from .calibration import (
    calibrate_thresholds_by_group,
    coerce_finite_threshold,
    split_conformal_threshold_metadata,
    threshold_for_group,
)
from .user_logins import (
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


def _default_caho_checkpoint() -> str:
    checkpoint = Path("caho_model_checkpoint")
    if checkpoint.exists():
        return str(checkpoint)
    return "sentence-transformers/all-MiniLM-L6-v2"


def _train_caho_samples(args: argparse.Namespace, samples: List[Sample], out_path: Path) -> None:
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
        include_original=args.loss == "contrastive",
        seed=args.seed,
    )
    batch_size = resolve_caho_batch_size(args.batch_size, use_grad_cache=args.grad_cache)

    if args.loss == "contrastive":
        trainer = ContrastiveTrainer(
            model,
            batch_size=batch_size,
            temperature=args.temperature,
            lr=args.lr,
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
            save_best_path=str(out_path) if args.save_best else None,
            seed=args.seed,
        )
        trainer.fit(dataset, epochs=args.epochs)
    else:
        trainer = CAHOTrainer(
            model,
            batch_size=batch_size,
            temperature=args.temperature,
            lr=args.lr,
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
    if not args.no_normalize:
        samples = [Sample(normalize_hostname(s.hostname), s.is_malicious, s.family) for s in samples]

    _train_caho_samples(args, samples, args.out)
    print(f"Saved CAHO encoder to {args.out}")


def cmd_train_caho_corpus(args: argparse.Namespace) -> None:
    benign_hosts = read_hostnames_from_benign_dir(args.benign_dir)
    benign_hosts = filter_hostnames(benign_hosts, min_length=args.min_length, dedup=not args.no_dedup)

    malicious_hosts: List[str] = []
    if args.malicious_jsonl_dir and args.malicious_jsonl_dir.is_dir():
        malicious_hosts.extend(read_hostnames_from_jsonl_dir(args.malicious_jsonl_dir, key=args.jsonl_key))
    if args.malicious_txt_dir and args.malicious_txt_dir.is_dir():
        malicious_hosts.extend(
            read_hostnames_from_txt_dir(
                args.malicious_txt_dir,
                include_csv=True,
                csv_column=args.csv_hostname_col,
            )
        )
    malicious_hosts = filter_hostnames(malicious_hosts, min_length=args.min_length, dedup=not args.no_dedup)

    if not benign_hosts and not malicious_hosts:
        raise ValueError("No hostnames loaded. Check input paths.")

    benign_samples = [Sample(h, is_malicious=False, family=None) for h in benign_hosts]
    malicious_samples = [
        Sample(h, is_malicious=True, family=args.malicious_family) for h in malicious_hosts
    ]
    samples = benign_samples + malicious_samples
    if not args.no_normalize:
        samples = [Sample(normalize_hostname(s.hostname), s.is_malicious, s.family) for s in samples]

    _train_caho_samples(args, samples, args.out)
    print(f"Saved CAHO encoder to {args.out}")


def cmd_eval_caho(args: argparse.Namespace) -> None:
    model = SentenceTransformer(args.model)
    device = _resolve_device(args.device)
    if device:
        try:
            model = model.to(device)
        except Exception:
            pass
    try:
        model.eval()
    except Exception:
        pass

    hostnames = _read_lines(args.input)
    if args.normalize:
        hostnames = [normalize_hostname(h) for h in hostnames]

    try:
        import torch

        model.eval()
        context = torch.inference_mode if hasattr(torch, "inference_mode") else torch.no_grad
        with context():
            embeddings = model.encode(
                hostnames,
                batch_size=args.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=args.embed_normalize,
                show_progress_bar=False,
            )
    except Exception:
        embeddings = model.encode(
            hostnames,
            batch_size=args.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=args.embed_normalize,
            show_progress_bar=False,
        )

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

    if args.encoder:
        config.encoder.model_name = args.encoder

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
    hostnames = _read_lines(args.input)
    if not args.no_normalize:
        hostnames = [normalize_hostname(h) for h in hostnames]
    groups = _read_parallel_lines(args.groups, len(hostnames), field_name="groups") if args.groups else None
    threshold = coerce_finite_threshold(
        args.threshold if args.threshold is not None else (model.threshold if model.threshold is not None else 0.0)
    )
    grouped_thresholds = getattr(model, "grouped_thresholds", None)
    if args.calibration:
        calib = json.loads(args.calibration.read_text())
        threshold = coerce_finite_threshold(calib.get("threshold", threshold))
        grouped_thresholds = calib.get("grouped_thresholds", grouped_thresholds)

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
    if args.save_model:
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
    hostnames = _read_lines(args.input)
    threshold = coerce_finite_threshold(
        args.threshold if args.threshold is not None else (model.threshold if model.threshold is not None else 0.0)
    )
    threshold_source = (
        "cli_threshold"
        if args.threshold is not None
        else ("model_bundle_threshold" if model.threshold is not None else "default_zero_threshold")
    )
    grouped_thresholds = getattr(model, "grouped_thresholds", None)
    grouped_thresholds_source = "model_bundle_grouped_thresholds" if grouped_thresholds else "none"
    if args.calibration:
        calib = json.loads(args.calibration.read_text())
        if "threshold" in calib:
            threshold = coerce_finite_threshold(calib["threshold"])
            threshold_source = "calibration_file_threshold"
        if "grouped_thresholds" in calib:
            grouped_thresholds = calib.get("grouped_thresholds", grouped_thresholds)
            grouped_thresholds_source = "calibration_file_grouped_thresholds" if grouped_thresholds else "none"
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


def _label_policy_help() -> str:
    details = " ".join(f"{name}: {desc}" for name, desc in LABEL_POLICY_DESCRIPTIONS.items())
    return f"How to combine GPT 5.5 / Claude Opus 4.8 labels. {details}"


def cmd_train_user_logins(args: argparse.Namespace) -> None:
    config = CCDConfig()
    if args.config and args.config.exists():
        config = CCDConfig.from_dict(json.loads(args.config.read_text()))

    if args.train_caho and args.encoder:
        raise ValueError("Use either --train-caho or --encoder, not both.")

    if args.encoder:
        config.encoder.model_name = args.encoder

    label_policy = LabelPolicy(args.label_policy)
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
            label_policy=label_policy,
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

    if args.train_caho:
        caho_out = args.caho_out or Path("caho_encoder_user_logins")
        base_model = args.caho_model or config.encoder.model_name
        if args.caho_resume and caho_out.exists():
            base_model = str(caho_out)
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

        model = SentenceTransformer(base_model)
        device = _resolve_device(args.caho_device)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ccd")
    sub = parser.add_subparsers(dest="command", required=True)

    train_caho = sub.add_parser("train-caho", help="Fine-tune CAHO encoder")
    train_caho.add_argument("--benign", required=True, type=Path)
    train_caho.add_argument("--malicious", required=True, type=Path)
    train_caho.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    train_caho.add_argument("--out", required=True, type=Path)
    train_caho.add_argument("--epochs", type=int, default=1)
    train_caho.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Effective CAHO batch size. Defaults to "
            f"{CAHO_94GB_GRAD_CACHE_BATCH_SIZE} with --grad-cache and "
            f"{CAHO_94GB_ACTUAL_BATCH_SIZE} otherwise for 94 GB VRAM."
        ),
    )
    train_caho.add_argument("--lr", type=float, default=2e-5)
    train_caho.add_argument("--temperature", type=float, default=0.07)
    train_caho.add_argument("--loss", choices=["supcon", "contrastive"], default="supcon")
    train_caho.add_argument("--augmenter", choices=["edit", "weighted", "hybrid"], default="edit")
    train_caho.add_argument("--weighted-num-augs", type=int, default=2)
    train_caho.add_argument("--weighted-max-attempts", type=int, default=3)
    train_caho.add_argument("--weighted-no-retry", action="store_true")
    train_caho.add_argument("--max-grad-norm", type=float, default=1.0)
    train_caho.add_argument("--scheduler", choices=["cosine", "none"], default="cosine")
    train_caho.add_argument("--min-lr", type=float, default=1e-5)
    train_caho.add_argument("--grad-cache", action="store_true", help="Enable GradCache for large batches")
    train_caho.add_argument("--grad-cache-chunk-size", type=int, default=CAHO_94GB_GRAD_CACHE_CHUNK_SIZE)
    train_caho.add_argument("--contrastive-loss", choices=["fixed", "learnable"], default="fixed")
    train_caho.add_argument("--contrastive-max-scale", type=float, default=100.0)
    train_caho.add_argument("--contrastive-min-scale", type=float, default=1.0)
    train_caho.add_argument("--optimize-contrastive-scale", action="store_true")
    train_caho.add_argument("--num-workers", type=int, default=0)
    train_caho.add_argument("--empty-cache", action="store_true")
    train_caho.add_argument("--device", default="auto", help="Training device: auto|cpu|cuda")
    train_caho.add_argument("--resume", action="store_true", help="Load model from --out if it exists")
    train_caho.add_argument("--save-best", action="store_true")
    train_caho.add_argument("--no-save-final", action="store_true")
    train_caho.add_argument("--no-normalize", action="store_true")
    train_caho.add_argument("--seed", type=int, default=13)
    train_caho.set_defaults(func=cmd_train_caho)

    train_caho_corpus = sub.add_parser(
        "train-caho-corpus",
        help="Fine-tune CAHO encoder from corpus directories",
    )
    train_caho_corpus.add_argument("--benign-dir", required=True, type=Path)
    train_caho_corpus.add_argument("--malicious-jsonl-dir", type=Path, default=None)
    train_caho_corpus.add_argument("--malicious-txt-dir", type=Path, default=None)
    train_caho_corpus.add_argument("--jsonl-key", default="hostname")
    train_caho_corpus.add_argument("--csv-hostname-col", default="Hostname")
    train_caho_corpus.add_argument("--min-length", type=int, default=5)
    train_caho_corpus.add_argument("--no-dedup", action="store_true")
    train_caho_corpus.add_argument("--malicious-family", default="corpus")
    train_caho_corpus.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    train_caho_corpus.add_argument("--out", required=True, type=Path)
    train_caho_corpus.add_argument("--epochs", type=int, default=1)
    train_caho_corpus.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Effective CAHO batch size. Defaults to "
            f"{CAHO_94GB_GRAD_CACHE_BATCH_SIZE} with --grad-cache and "
            f"{CAHO_94GB_ACTUAL_BATCH_SIZE} otherwise for 94 GB VRAM."
        ),
    )
    train_caho_corpus.add_argument("--lr", type=float, default=2e-5)
    train_caho_corpus.add_argument("--temperature", type=float, default=0.07)
    train_caho_corpus.add_argument("--loss", choices=["supcon", "contrastive"], default="supcon")
    train_caho_corpus.add_argument("--augmenter", choices=["edit", "weighted", "hybrid"], default="edit")
    train_caho_corpus.add_argument("--weighted-num-augs", type=int, default=2)
    train_caho_corpus.add_argument("--weighted-max-attempts", type=int, default=3)
    train_caho_corpus.add_argument("--weighted-no-retry", action="store_true")
    train_caho_corpus.add_argument("--max-grad-norm", type=float, default=1.0)
    train_caho_corpus.add_argument("--scheduler", choices=["cosine", "none"], default="cosine")
    train_caho_corpus.add_argument("--min-lr", type=float, default=1e-5)
    train_caho_corpus.add_argument("--grad-cache", action="store_true", help="Enable GradCache for large batches")
    train_caho_corpus.add_argument("--grad-cache-chunk-size", type=int, default=CAHO_94GB_GRAD_CACHE_CHUNK_SIZE)
    train_caho_corpus.add_argument("--contrastive-loss", choices=["fixed", "learnable"], default="fixed")
    train_caho_corpus.add_argument("--contrastive-max-scale", type=float, default=100.0)
    train_caho_corpus.add_argument("--contrastive-min-scale", type=float, default=1.0)
    train_caho_corpus.add_argument("--optimize-contrastive-scale", action="store_true")
    train_caho_corpus.add_argument("--num-workers", type=int, default=0)
    train_caho_corpus.add_argument("--empty-cache", action="store_true")
    train_caho_corpus.add_argument("--device", default="auto", help="Training device: auto|cpu|cuda")
    train_caho_corpus.add_argument("--resume", action="store_true", help="Load model from --out if it exists")
    train_caho_corpus.add_argument("--save-best", action="store_true")
    train_caho_corpus.add_argument("--no-save-final", action="store_true")
    train_caho_corpus.add_argument("--no-normalize", action="store_true")
    train_caho_corpus.add_argument("--seed", type=int, default=13)
    train_caho_corpus.set_defaults(func=cmd_train_caho_corpus)

    eval_caho = sub.add_parser("eval-caho", help="Encode hostnames with a CAHO checkpoint")
    eval_caho.add_argument("--model", default=_default_caho_checkpoint())
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
    train_priors.add_argument("--encoder", type=str, default=None, help="Override encoder model path/name")
    train_priors.add_argument("--batch-size", type=int, default=64)
    train_priors.add_argument("--no-normalize", action="store_true")
    train_priors.set_defaults(func=cmd_train_priors)

    score = sub.add_parser("score", help="Score hostnames with CCD model")
    score.add_argument("--model", required=True, type=Path)
    score.add_argument("--input", required=True, type=Path)
    score.add_argument("--output", required=True, type=Path)
    score.add_argument("--threshold", type=float, default=None)
    score.add_argument("--calibration", type=Path, default=None, help="JSON file from `ccd calibrate`")
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

    calibrate = sub.add_parser("calibrate", help="Calibrate a fixed-FPR threshold")
    calibrate.add_argument("--model", required=True, type=Path)
    calibrate.add_argument("--benign", required=True, type=Path)
    calibrate.add_argument("--output", required=True, type=Path)
    calibrate.add_argument("--groups", type=Path, default=None, help="Optional one-calibration-group-per-benign-row file.")
    calibrate.add_argument(
        "--save-model",
        type=Path,
        default=None,
        help="Optional path for a model bundle with the calibrated threshold embedded.",
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
    certify.add_argument("--threshold", type=float, default=None)
    certify.add_argument("--calibration", type=Path, default=None, help="JSON file from `ccd calibrate`")
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

    train_user_logins = sub.add_parser(
        "train-user-logins",
        help="Build CCD priors directly from user_logins CSVs",
    )
    train_user_logins.add_argument(
        "--user-logins-dir",
        type=Path,
        default=Path("hostname_injection_benchmark/user_logins"),
    )
    train_user_logins.add_argument("--output", type=Path)
    train_user_logins.add_argument(
        "--label-policy",
        choices=[policy.value for policy in LabelPolicy],
        default=LabelPolicy.BOTH_M.value,
        help=_label_policy_help(),
    )
    train_user_logins.add_argument("--malicious-family", default="dns_cmd_injection")
    train_user_logins.add_argument("--config", type=Path, default=None)
    train_user_logins.add_argument(
        "--encoder",
        type=str,
        default=None,
        help="Override encoder model path/name",
    )
    train_user_logins.add_argument("--batch-size", type=int, default=64)
    train_user_logins.add_argument(
        "--buffer-size",
        type=int,
        default=2048,
        help="Hostnames to buffer per class before encoding",
    )
    train_user_logins.add_argument("--no-normalize", action="store_true")
    train_user_logins.add_argument("--hostname-col", default=DEFAULT_USER_LOGINS_COLUMN)
    train_user_logins.add_argument("--sonnet-col", default=DEFAULT_SONNET_COLUMN)
    train_user_logins.add_argument("--opus-col", default=DEFAULT_OPUS_COLUMN)
    train_user_logins.add_argument("--sonnet-conf-col", default=DEFAULT_SONNET_CONF_COLUMN)
    train_user_logins.add_argument("--opus-conf-col", default=DEFAULT_OPUS_CONF_COLUMN)
    train_user_logins.add_argument("--min-confidence", type=float, default=None)
    train_user_logins.add_argument("--min-sonnet-confidence", type=float, default=None)
    train_user_logins.add_argument("--min-opus-confidence", type=float, default=None)
    train_user_logins.add_argument("--dry-run", action="store_true", help="Only report label counts")
    train_user_logins.add_argument("--train-caho", action="store_true", help="Fine-tune CAHO encoder first")
    train_user_logins.add_argument("--caho-out", type=Path, default=None)
    train_user_logins.add_argument(
        "--caho-model",
        type=str,
        default=None,
        help="Base SentenceTransformer model for CAHO fine-tuning",
    )
    train_user_logins.add_argument("--caho-epochs", type=int, default=1)
    train_user_logins.add_argument(
        "--caho-batch-size",
        type=int,
        default=None,
        help=(
            "Effective CAHO batch size. Defaults to "
            f"{CAHO_94GB_GRAD_CACHE_BATCH_SIZE} with --caho-grad-cache and "
            f"{CAHO_94GB_ACTUAL_BATCH_SIZE} otherwise for 94 GB VRAM."
        ),
    )
    train_user_logins.add_argument("--caho-lr", type=float, default=2e-5)
    train_user_logins.add_argument("--caho-temperature", type=float, default=0.07)
    train_user_logins.add_argument("--caho-loss", choices=["supcon", "contrastive"], default="supcon")
    train_user_logins.add_argument("--caho-augmenter", choices=["edit", "weighted", "hybrid"], default="edit")
    train_user_logins.add_argument("--caho-weighted-num-augs", type=int, default=2)
    train_user_logins.add_argument("--caho-weighted-max-attempts", type=int, default=3)
    train_user_logins.add_argument("--caho-weighted-no-retry", action="store_true")
    train_user_logins.add_argument("--caho-max-grad-norm", type=float, default=1.0)
    train_user_logins.add_argument("--caho-scheduler", choices=["cosine", "none"], default="cosine")
    train_user_logins.add_argument("--caho-min-lr", type=float, default=1e-5)
    train_user_logins.add_argument(
        "--caho-grad-cache",
        action="store_true",
        help="Enable GradCache for CAHO training",
    )
    train_user_logins.add_argument("--caho-grad-cache-chunk-size", type=int, default=CAHO_94GB_GRAD_CACHE_CHUNK_SIZE)
    train_user_logins.add_argument("--caho-contrastive-loss", choices=["fixed", "learnable"], default="fixed")
    train_user_logins.add_argument("--caho-contrastive-max-scale", type=float, default=100.0)
    train_user_logins.add_argument("--caho-contrastive-min-scale", type=float, default=1.0)
    train_user_logins.add_argument("--caho-optimize-contrastive-scale", action="store_true")
    train_user_logins.add_argument("--caho-num-workers", type=int, default=0)
    train_user_logins.add_argument("--caho-empty-cache", action="store_true")
    train_user_logins.add_argument("--caho-device", default="auto", help="Training device: auto|cpu|cuda")
    train_user_logins.add_argument("--caho-resume", action="store_true")
    train_user_logins.add_argument("--caho-save-best", action="store_true")
    train_user_logins.add_argument("--caho-no-save-final", action="store_true")
    train_user_logins.add_argument(
        "--caho-sample",
        type=int,
        default=None,
        help="Reservoir sample size per class for CAHO training",
    )
    train_user_logins.add_argument("--caho-seed", type=int, default=13)
    train_user_logins.set_defaults(func=cmd_train_user_logins)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
