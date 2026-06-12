#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ccd.config import CCDConfig, ConeConfig
from ccd.encoder import CahoEncoder, require_trained_caho_checkpoint
from ccd.scoring import ccd_scores_logpriors_topk
from ccd.utils import l2_normalize, stable_log


def _read_inputs(path: Path, num_samples: int) -> list[str]:
    if num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    rows = [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{path} contains no input rows")
    return [rows[i % len(rows)] for i in range(num_samples)]


def _sync_device(device: str | None) -> None:
    if not device:
        return
    try:
        import torch

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elif device == "mps" and hasattr(torch, "mps"):
            try:
                torch.mps.synchronize()
            except Exception:
                pass
    except Exception:
        return


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (q / 100.0)
    lower = int(np.floor(position))
    upper = int(np.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _benchmark(
    fn: Callable[[], Any],
    *,
    samples: int,
    repeats: int,
    warmup: int,
    device: str | None = None,
) -> dict[str, Any]:
    if repeats <= 0:
        raise ValueError("--repeats must be positive")
    if warmup < 0:
        raise ValueError("--warmup must be non-negative")

    for _ in range(warmup):
        _ = fn()
        _sync_device(device)

    durations: list[float] = []
    for _ in range(repeats):
        _sync_device(device)
        start = time.perf_counter()
        _ = fn()
        _sync_device(device)
        durations.append(time.perf_counter() - start)

    median_s = _percentile(durations, 50)
    p95_s = _percentile(durations, 95)
    p99_s = _percentile(durations, 99)
    samples = max(samples, 1)
    return {
        "samples": samples,
        "repeats": repeats,
        "warmup": warmup,
        "duration_s_median": median_s,
        "duration_s_p95": p95_s,
        "duration_s_p99": p99_s,
        "ms_per_sample_median": 1000.0 * median_s / samples,
        "ms_per_sample_p95": 1000.0 * p95_s / samples,
        "ms_per_sample_p99": 1000.0 * p99_s / samples,
        "samples_per_s_median": samples / max(median_s, 1e-12),
    }


def _random_prior(rng: np.random.Generator, n: int) -> np.ndarray:
    values = rng.random(n, dtype=np.float32) + np.float32(1e-3)
    return (values / values.sum()).astype(np.float32)


def benchmark_scoring_kernel(args: argparse.Namespace) -> dict[str, Any]:
    rng = np.random.default_rng(args.seed)
    cone_config = ConeConfig(
        dim=args.dim,
        num_cones=args.num_cones,
        active_cones=args.active_cones,
        temperature=args.temperature,
        seed=args.seed,
    )
    embeddings = l2_normalize(rng.standard_normal((args.num_samples, args.dim)).astype(np.float32))
    axes = l2_normalize(rng.standard_normal((args.num_cones, args.dim)).astype(np.float32))
    benign_log_prior = stable_log(_random_prior(rng, args.num_cones))
    malicious_log_priors = {
        "shell": stable_log(_random_prior(rng, args.num_cones)),
        "query": stable_log(_random_prior(rng, args.num_cones)),
        "template": stable_log(_random_prior(rng, args.num_cones)),
    }

    def score_once() -> np.ndarray:
        return ccd_scores_logpriors_topk(
            embeddings,
            axes,
            benign_log_prior,
            malicious_log_priors,
            cone_config.temperature,
            cone_config.active_cones,
            effective_count=1.0,
        )

    metrics = _benchmark(
        score_once,
        samples=args.num_samples,
        repeats=args.repeats,
        warmup=args.warmup,
    )
    metrics.update(
        {
            "dim": args.dim,
            "num_cones": args.num_cones,
            "active_cones": args.active_cones,
            "temperature": args.temperature,
            "families": sorted(malicious_log_priors),
            "path": "ccd_scores_logpriors_topk",
        }
    )
    return metrics


def benchmark_encoder(args: argparse.Namespace, hostnames: list[str]) -> dict[str, Any] | None:
    if args.skip_encoder:
        return None
    config = CCDConfig()
    config.encoder.model_name = str(require_trained_caho_checkpoint(args.checkpoint, purpose="benchmark_artifact_latency.py"))
    config.encoder.device = args.device
    encoder = CahoEncoder(config.encoder)
    device_type = encoder.device_type()

    def encode_once():
        return encoder.encode_torch(hostnames, batch_size=args.batch_size, normalize=True)

    metrics = _benchmark(
        encode_once,
        samples=len(hostnames),
        repeats=args.repeats,
        warmup=args.warmup,
        device=device_type,
    )
    metrics.update(
        {
            "checkpoint": args.checkpoint,
            "batch_size": args.batch_size,
            "device": device_type,
            "path": "CahoEncoder.encode_torch",
        }
    )
    return metrics


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    hostnames = _read_inputs(Path(args.input), args.num_samples)
    encoder_metrics = benchmark_encoder(args, hostnames)
    scoring_metrics = benchmark_scoring_kernel(args)
    return {
        "status": "pass",
        "hardware_dependent": True,
        "paper_latency_note": (
            "This smoke benchmark exercises the artifact's local encoder and CCD scoring-kernel paths. "
            "It is not expected to reproduce the paper's production p95/p99/p99.9 or 0.60 ms amortized "
            "latency numbers on arbitrary evaluator hardware."
        ),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "processor": platform.processor(),
        },
        "input": {
            "path": args.input,
            "num_samples": args.num_samples,
        },
        "encoder": encoder_metrics,
        "scoring_kernel": scoring_metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a local CCD latency smoke benchmark.")
    parser.add_argument("--checkpoint", default=None, help="Trained CAHO checkpoint directory")
    parser.add_argument("--input", default="examples/queries.txt")
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-encoder", action="store_true", help="Only benchmark the deterministic scoring kernel.")
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--num-cones", type=int, default=256)
    parser.add_argument("--active-cones", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if not args.skip_encoder and not args.checkpoint:
        raise ValueError("--checkpoint is required unless --skip-encoder is set")

    report = build_report(args)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
