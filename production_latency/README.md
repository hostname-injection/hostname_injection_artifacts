# Production-Latency Aggregate Accounting

This directory contains release-safe aggregate values for the paper's production
latency and throughput claims in Figure 5 and Table 5.

Run:

```bash
python scripts/recompute_production_latency_metrics.py
```

The script validates full-path latency quantile ordering, single-host
throughput, scoring-kernel latency ordering, Table 5 CCD tail-latency alignment,
and the baseline-context constraints used for the production alert budget. The
local latency smoke benchmark remains hardware-dependent and is not expected to
reproduce these production numbers on evaluator hardware.

Raw production serving traces, deployment topology, and closed control internals
are intentionally outside the public artifact boundary.
