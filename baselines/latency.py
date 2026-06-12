from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

import numpy as np


@dataclass
class LatencyMetrics:
    samples: int
    duration_s: float
    ms_per_sample: float
    samples_per_s: float

    def to_dict(self):
        return {
            "samples": self.samples,
            "duration_s": self.duration_s,
            "ms_per_sample": self.ms_per_sample,
            "samples_per_s": self.samples_per_s,
        }


def _maybe_sync(device: Optional[str] = None) -> None:
    try:
        import torch

        if device is None:
            return
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elif device == "mps" and hasattr(torch, "mps"):
            try:
                torch.mps.synchronize()
            except Exception:
                pass
    except Exception:
        return


def measure_latency(
    predict_fn: Callable[[Sequence[str]], Iterable[int]],
    inputs: Sequence[str],
    *,
    batch_size: int = 256,
    repeats: int = 3,
    warmup: int = 1,
    device: Optional[str] = None,
) -> LatencyMetrics:
    if not inputs:
        return LatencyMetrics(samples=0, duration_s=0.0, ms_per_sample=0.0, samples_per_s=0.0)

    # Warmup
    for _ in range(max(1, warmup)):
        _maybe_sync(device)
        _ = predict_fn(inputs[:batch_size])
        _maybe_sync(device)

    durations = []
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        _maybe_sync(device)
        _ = predict_fn(inputs)
        _maybe_sync(device)
        durations.append(time.perf_counter() - start)

    duration = float(np.median(durations))
    samples = len(inputs)
    ms_per_sample = (duration / max(samples, 1)) * 1000.0
    samples_per_s = samples / max(duration, 1e-9)

    return LatencyMetrics(
        samples=samples,
        duration_s=duration,
        ms_per_sample=ms_per_sample,
        samples_per_s=samples_per_s,
    )
