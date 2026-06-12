"""Device/throughput diagnostics for CCD inference."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

from .config import CCDConfig
from .encoder import CahoEncoder


def _make_inputs(n: int) -> List[str]:
    return [f"host-{i}.example.com" for i in range(n)]


def _sync_device(device: str) -> None:
    if device == "cuda":
        try:
            import torch

            torch.cuda.synchronize()
        except Exception:
            pass
    if device == "mps":
        try:
            import torch

            if hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
        except Exception:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CCD device/throughput diagnostics.")
    parser.add_argument(
        "--checkpoint",
        default="ccd-local-hash-encoder",
        help="CAHO encoder path or SentenceTransformer name.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto|cpu|cuda|mps",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for encoding.")
    parser.add_argument("--num-samples", type=int, default=2048, help="Number of synthetic hostnames.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")

    config = CCDConfig()
    config.encoder.model_name = args.checkpoint
    config.encoder.device = args.device

    encoder = CahoEncoder(config.encoder)
    device_type = encoder.device_type()
    hostnames = _make_inputs(args.num_samples)

    # Warmup
    for _ in range(max(0, args.warmup)):
        _ = encoder.encode_torch(hostnames[: args.batch_size], batch_size=args.batch_size, normalize=True)
        _sync_device(device_type)

    start = time.perf_counter()
    _ = encoder.encode_torch(hostnames, batch_size=args.batch_size, normalize=True)
    _sync_device(device_type)
    elapsed = time.perf_counter() - start

    throughput = args.num_samples / elapsed if elapsed > 0 else 0.0

    print("CCD diagnostics")
    print(f"Device: {device_type}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Batch size: {args.batch_size}")
    print(f"Samples: {args.num_samples}")
    print(f"Elapsed: {elapsed:.4f}s")
    print(f"Throughput: {throughput:.2f} samples/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
