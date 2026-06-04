from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

from baselines.dataset import load_benchmark_dataset, train_test_split
from baselines.downloads import ensure_repos_for_baselines
from baselines.latency import measure_latency
from baselines.metrics import classification_metrics
from baselines.models.registry import BASELINE_SPECS, get_baseline, list_baselines
from ccd.benchmark_dataset import BenchmarkLabelMethod


def _parse_baselines(arg: str) -> List[str]:
    if not arg:
        return []
    return [name.strip() for name in arg.split(",") if name.strip()]


def _build_baseline(name: str, args: argparse.Namespace):
    kwargs: Dict[str, object] = {}
    if name in {"char-cnn", "urlnet"}:
        kwargs.update(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
        )
        if name == "urlnet":
            kwargs.update(
                repo_root=str(args.repo_root),
                use_official_repo=args.use_official_repos,
                allow_downloads=args.allow_downloads,
            )
    if name == "urlbert":
        kwargs.update(
            model_name=args.urlbert_model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            allow_downloads=args.allow_downloads,
            repo_root=str(args.repo_root),
            use_official_repo=args.use_official_repos,
        )
    if name == "csi":
        kwargs.update(
            model_name=args.encoder_model,
            epochs=args.csi_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            allow_downloads=args.allow_downloads,
            repo_root=str(args.repo_root),
            use_official_repo=args.use_official_repos,
        )
    if name in {"knn-density", "mahalanobis", "deep-sad", "drocc", "deep-svdd", "deep-one-class", "t-mahalanobis"}:
        kwargs.update(
            model_name=args.encoder_model,
            batch_size=args.batch_size,
            device=args.device,
        )
        if name in {"deep-sad", "drocc", "deep-svdd", "deep-one-class"}:
            kwargs.update(epochs=args.epochs, lr=args.lr)
        if name in {"deep-sad", "drocc", "deep-svdd", "deep-one-class"}:
            kwargs.update(
                repo_root=str(args.repo_root),
                use_official_repo=args.use_official_repos,
                allow_downloads=args.allow_downloads,
            )
    if name.startswith("tfidf"):
        kwargs.update(random_state=args.seed)

    return get_baseline(name, **kwargs)


def _sample_latency_inputs(texts: List[str], sample_size: int, seed: int) -> List[str]:
    if len(texts) <= sample_size:
        return texts
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(texts), size=sample_size, replace=False)
    return [texts[i] for i in idx]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run baseline models for hostname command injection.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("HostnameCommandInjectionBenchmark"),
        help="Benchmark root containing manifest.json. All benchmark families are loaded.",
    )
    parser.add_argument(
        "--label-method",
        type=str,
        default=BenchmarkLabelMethod.ANY_MALICIOUS_ELSE_BENIGN.value,
        choices=[p.value for p in BenchmarkLabelMethod],
        help="How to resolve GPT 5.5 / Claude Opus 4.8 benchmark labels.",
    )
    parser.add_argument("--sample-per-class", type=int, default=None, help="Reservoir sample per class.")
    parser.add_argument("--deduplicate", action="store_true", help="Deduplicate texts before training.")
    parser.add_argument("--max-rows", type=int, default=None, help="Debug only: limit benchmark rows after manifest order.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed.")

    parser.add_argument("--baselines", type=str, default="", help="Comma-separated list of baselines to run.")
    parser.add_argument("--list", action="store_true", help="List available baselines and exit.")
    parser.add_argument("--allow-downloads", action="store_true", help="Allow downloading external models.")
    parser.add_argument(
        "--download-repos",
        action="store_true",
        help="Download official baseline repos into --repo-root before running.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("baselines/downloads"),
        help="Where to store downloaded baseline repos.",
    )
    parser.add_argument(
        "--use-official-repos",
        action="store_true",
        help="When available, load models from the downloaded official repos.",
    )

    parser.add_argument("--encoder-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--urlbert-model", type=str, default="bert-base-uncased")

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--csi-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--latency-samples", type=int, default=2000)
    parser.add_argument("--latency-repeats", type=int, default=3)
    parser.add_argument("--latency-warmup", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("baselines/outputs/results.csv"))

    args = parser.parse_args()

    if args.list:
        for spec in list_baselines():
            print(f"{spec.name}: {spec.description}")
        return 0

    selected = _parse_baselines(args.baselines)
    if not selected:
        selected = list(BASELINE_SPECS.keys())

    if args.download_repos or (args.use_official_repos and args.allow_downloads):
        try:
            ensure_repos_for_baselines(selected, args.repo_root, allow_downloads=True)
        except Exception as exc:
            print(f"Failed to download repos: {exc}")

    texts, labels, stats = load_benchmark_dataset(
        args.data_dir,
        label_method=BenchmarkLabelMethod(args.label_method),
        sample_per_class=args.sample_per_class,
        deduplicate=args.deduplicate,
        seed=args.seed,
        max_rows=args.max_rows,
    )

    if len(texts) == 0:
        print("No data loaded. Check dataset path and filters.")
        return 1

    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=args.test_size, seed=args.seed
    )

    results: List[Dict[str, object]] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for name in selected:
        spec = BASELINE_SPECS.get(name)
        if spec is None:
            print(f"Unknown baseline: {name}")
            continue
        if spec.needs_download and not args.allow_downloads:
            print(f"Skipping {name}: downloads disabled (pass --allow-downloads).")
            continue

        print(f"\n=== {name} ===")
        try:
            baseline = _build_baseline(name, args)
            baseline.fit(x_train, y_train)
            preds = baseline.predict(x_test, batch_size=args.batch_size)
            metrics = classification_metrics(y_test, preds)

            latency_inputs = _sample_latency_inputs(x_test, args.latency_samples, args.seed)
            latency = measure_latency(
                lambda batch: baseline.predict(batch, batch_size=args.batch_size),
                latency_inputs,
                repeats=args.latency_repeats,
                warmup=args.latency_warmup,
                device=args.device,
            )

            row = {
                "baseline": name,
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1": metrics.f1,
                "tp": metrics.tp,
                "fp": metrics.fp,
                "tn": metrics.tn,
                "fn": metrics.fn,
                "train_size": len(x_train),
                "test_size": len(x_test),
                "latency_ms": latency.ms_per_sample,
                "throughput": latency.samples_per_s,
                "notes": "",
            }
            results.append(row)
            print(
                f"accuracy={metrics.accuracy:.4f} f1={metrics.f1:.4f} "
                f"latency_ms={latency.ms_per_sample:.2f} samples/s={latency.samples_per_s:.1f}"
            )
        except Exception as exc:  # pragma: no cover - execution path
            print(f"Baseline {name} failed: {exc}")
            results.append({"baseline": name, "notes": f"error: {exc}"})

    if results:
        fieldnames = sorted({key for row in results for key in row.keys()})
        with args.output.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        print(f"\nSaved results to {args.output}")

    print(
        f"Dataset stats: total_rows={stats.total_rows} used_rows={stats.used_rows} "
        f"benign={stats.benign} malicious={stats.malicious}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
