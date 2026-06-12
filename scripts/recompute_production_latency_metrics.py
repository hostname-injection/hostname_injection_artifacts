#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_number(value: object, *, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"{path} must be a finite number")
    return float(value)


def require_int(value: object, *, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{path} must be an integer")
    if value < 0:
        raise ValueError(f"{path} must be non-negative")
    return value


def require_bool(value: object, *, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be true or false")
    return value


def require_mapping(value: object, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def validate_full_path(latency: Mapping[str, Any]) -> dict[str, Any]:
    median = require_number(latency.get("median_ms"), path="full_path.median_ms")
    p95 = require_number(latency.get("p95_ms"), path="full_path.p95_ms")
    p99 = require_number(latency.get("p99_ms"), path="full_path.p99_ms")
    p999 = require_number(latency.get("p999_ms"), path="full_path.p999_ms")
    throughput = require_number(latency.get("single_host_throughput_per_s"), path="full_path.single_host_throughput_per_s")
    concurrency = require_int(latency.get("replay_concurrency"), path="full_path.replay_concurrency")
    if not (0.0 < median <= p95 <= p99 <= p999):
        raise ValueError("full-path latency quantiles must satisfy 0 < median <= p95 <= p99 <= p999")
    if throughput <= 0.0:
        raise ValueError("single-host throughput must be positive")
    if concurrency <= 0:
        raise ValueError("replay concurrency must be positive")
    return {
        "median_ms": median,
        "p95_ms": p95,
        "p99_ms": p99,
        "p999_ms": p999,
        "tail_spread_p999_over_median": p999 / median,
        "single_host_throughput_per_s": throughput,
        "single_host_throughput_k_per_s": throughput / 1000.0,
        "replay_concurrency": concurrency,
    }


def validate_scoring_kernel(scoring: Mapping[str, Any]) -> dict[str, Any]:
    p50 = require_number(scoring.get("p50_ms"), path="scoring_kernel.p50_ms")
    p99 = require_number(scoring.get("p99_ms"), path="scoring_kernel.p99_ms")
    if not (0.0 < p50 <= p99):
        raise ValueError("scoring-kernel latency quantiles must satisfy 0 < p50 <= p99")
    return {
        "p50_ms": p50,
        "p99_ms": p99,
        "p99_over_p50": p99 / p50,
    }


def validate_table5_alignment(table5: Mapping[str, Any], full_path: Mapping[str, Any]) -> dict[str, Any]:
    ccd = require_mapping(table5.get("ccd"), path="table5_alignment.ccd")
    table_p99 = require_number(ccd.get("p99_ms"), path="table5_alignment.ccd.p99_ms")
    table_p999 = require_number(ccd.get("p999_ms"), path="table5_alignment.ccd.p999_ms")
    if not math.isclose(table_p99, require_number(full_path.get("p99_ms"), path="full_path.p99_ms"), abs_tol=1e-9):
        raise ValueError("Table 5 CCD p99 does not match production full-path p99")
    if not math.isclose(table_p999, require_number(full_path.get("p999_ms"), path="full_path.p999_ms"), abs_tol=1e-9):
        raise ValueError("Table 5 CCD p99.9 does not match production full-path p99.9")
    return {
        "ccd_table5_p99_ms": table_p99,
        "ccd_table5_p999_ms": table_p999,
        "aligned_with_full_path_tails": True,
    }


def validate_baseline_context(context: Mapping[str, Any]) -> dict[str, Any]:
    min_slowdown = require_number(context.get("llm_slowdown_min_x"), path="baseline_context.llm_slowdown_min_x")
    max_slowdown = require_number(context.get("llm_slowdown_max_x"), path="baseline_context.llm_slowdown_max_x")
    if not (1.0 < min_slowdown <= max_slowdown):
        raise ValueError("LLM slowdown range must satisfy 1 < min <= max")
    candidate_count = require_int(context.get("predeployment_baseline_count"), path="baseline_context.predeployment_baseline_count")
    if candidate_count <= 0:
        raise ValueError("predeployment baseline count must be positive")
    for key in ("learned_and_llm_candidates_not_inserted_in_shared_alert_queue", "matched_splits", "validation_only_model_selection", "benign_split_conformal_thresholds"):
        if require_bool(context.get(key), path=f"baseline_context.{key}") is not True:
            raise ValueError(f"baseline_context.{key} must be true")
    return {
        "llm_slowdown_range_x": [min_slowdown, max_slowdown],
        "predeployment_baseline_count": candidate_count,
        "alert_queue_limited_to_deployable_pair": True,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    counts_path = Path(args.counts)
    data = load_json(counts_path)
    full_path = require_mapping(data.get("full_path_latency"), path="full_path_latency")
    scoring = require_mapping(data.get("scoring_kernel_latency"), path="scoring_kernel_latency")
    table5 = require_mapping(data.get("table5_alignment"), path="table5_alignment")
    baseline_context = require_mapping(data.get("baseline_context"), path="baseline_context")
    local_smoke = require_mapping(data.get("local_smoke_boundary"), path="local_smoke_boundary")
    if require_bool(local_smoke.get("hardware_dependent"), path="local_smoke_boundary.hardware_dependent") is not True:
        raise ValueError("local_smoke_boundary.hardware_dependent must be true")
    if require_bool(local_smoke.get("expected_to_reproduce_production_latency"), path="local_smoke_boundary.expected_to_reproduce_production_latency") is not False:
        raise ValueError("local smoke must not be expected to reproduce production latency")

    return {
        "status": "pass",
        "counts_file": str(counts_path),
        "source": data.get("source", "published_aggregate_production_latency"),
        "paper_sections": data.get("paper_sections", []),
        "full_path_latency": full_path,
        "scoring_kernel_latency": scoring,
        "table5_alignment": table5,
        "baseline_context": baseline_context,
        "local_smoke_boundary": local_smoke,
        "derived": {
            "full_path_latency": validate_full_path(full_path),
            "scoring_kernel_latency": validate_scoring_kernel(scoring),
            "table5_alignment": validate_table5_alignment(table5, full_path),
            "baseline_context": validate_baseline_context(baseline_context),
        },
        "private_by_design": data.get("private_by_design", []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute release-safe aggregate production-latency accounting.")
    parser.add_argument(
        "--counts",
        default="production_latency/paper_production_latency_counts.json",
        help="Release-safe production-latency aggregate counts.",
    )
    parser.add_argument("--out", default=None, help="Optional path for the JSON report.")
    args = parser.parse_args()

    report = build_report(args)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
