#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_int(value: object, *, path: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{path} must be an integer")
    if value < 0:
        raise ValueError(f"{path} must be non-negative")
    return value


def require_number(value: object, *, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"{path} must be a finite number")
    return float(value)


def require_bool(value: object, *, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be true or false")
    return value


def require_mapping(value: object, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def sum_ints(mapping: Mapping[str, Any], *, path: str) -> int:
    return sum(require_int(value, path=f"{path}.{key}") for key, value in mapping.items())


def percent(part: int, total: int) -> float:
    if total == 0:
        raise ValueError("cannot compute percentage with denominator 0")
    return 100.0 * part / total


def assert_close(observed: float, expected: float, *, path: str, abs_tol: float = 0.02) -> None:
    if not math.isclose(observed, expected, abs_tol=abs_tol):
        raise ValueError(f"{path} expected {expected}, observed {observed}")


def validate_replay_profile(replay: Mapping[str, Any]) -> dict[str, Any]:
    n_rows = require_int(replay.get("n_rows"), path="replay.n_rows")
    n_tenants = require_int(replay.get("n_tenants"), path="replay.n_tenants")
    window_months = require_int(replay.get("window_months"), path="replay.window_months")
    source_rows = require_mapping(replay.get("source_rows"), path="replay.source_rows")
    resolved_by_source = require_mapping(replay.get("resolved_rows_by_source"), path="replay.resolved_rows_by_source")
    unresolved_by_source = require_mapping(replay.get("unresolved_rows_by_source"), path="replay.unresolved_rows_by_source")
    labels = require_mapping(replay.get("label_counts"), path="replay.label_counts")
    event_counts = require_mapping(replay.get("event_counts"), path="replay.event_counts")
    tenant_source_counts = require_mapping(replay.get("tenant_source_counts"), path="replay.tenant_source_counts")
    repairs = require_mapping(replay.get("quality_repairs"), path="replay.quality_repairs")
    agreement = require_mapping(replay.get("agreement"), path="replay.agreement")
    partition_policy = require_mapping(replay.get("partition_policy"), path="replay.partition_policy")

    source_total = sum_ints(source_rows, path="replay.source_rows")
    if source_total != n_rows:
        raise ValueError(f"source rows sum to {source_total}, expected replay.n_rows {n_rows}")

    resolved_total = sum_ints(resolved_by_source, path="replay.resolved_rows_by_source")
    unresolved_total = sum_ints(unresolved_by_source, path="replay.unresolved_rows_by_source")
    for source, value in source_rows.items():
        source_count = require_int(value, path=f"replay.source_rows.{source}")
        resolved = require_int(resolved_by_source.get(source), path=f"replay.resolved_rows_by_source.{source}")
        unresolved = require_int(unresolved_by_source.get(source), path=f"replay.unresolved_rows_by_source.{source}")
        if resolved + unresolved != source_count:
            raise ValueError(f"{source} resolved + unresolved does not equal source total")

    benign = require_int(labels.get("resolved_benign"), path="replay.label_counts.resolved_benign")
    positive = require_int(labels.get("verified_executable_semantics"), path="replay.label_counts.verified_executable_semantics")
    unresolved = require_int(labels.get("unresolved"), path="replay.label_counts.unresolved")
    if benign + positive + unresolved != n_rows:
        raise ValueError("label counts do not sum to replay.n_rows")
    if benign + positive != resolved_total:
        raise ValueError("resolved label counts do not sum to resolved rows")
    if unresolved != unresolved_total:
        raise ValueError("unresolved label count does not match source unresolved rows")

    user_login_events = require_int(event_counts.get("successful_logins"), path="replay.event_counts.successful_logins") + require_int(
        event_counts.get("failed_logins"), path="replay.event_counts.failed_logins"
    )
    dns_events = require_int(
        event_counts.get("successful_dns_resolutions"), path="replay.event_counts.successful_dns_resolutions"
    ) + require_int(event_counts.get("failed_dns_resolutions"), path="replay.event_counts.failed_dns_resolutions")
    user_login_residual = require_int(source_rows.get("user_login_hostnames"), path="replay.source_rows.user_login_hostnames") - user_login_events
    dns_residual = require_int(source_rows.get("dns_resolution_hostnames"), path="replay.source_rows.dns_resolution_hostnames") - dns_events
    max_event_residual = require_int(replay.get("max_event_source_residual_rows", 0), path="replay.max_event_source_residual_rows")
    if abs(user_login_residual) > max_event_residual or abs(dns_residual) > max_event_residual:
        raise ValueError(
            "event count residual exceeds allowed table-level residual: "
            f"user_login={user_login_residual}, dns={dns_residual}, allowed={max_event_residual}"
        )

    for key in ("user_login", "dns", "both"):
        require_int(tenant_source_counts.get(key), path=f"replay.tenant_source_counts.{key}")
    if require_int(tenant_source_counts["user_login"], path="tenant_source_counts.user_login") > n_tenants:
        raise ValueError("user-login tenant count exceeds total tenant count")
    if require_int(tenant_source_counts["dns"], path="tenant_source_counts.dns") > n_tenants:
        raise ValueError("DNS tenant count exceeds total tenant count")
    if require_int(tenant_source_counts["both"], path="tenant_source_counts.both") > min(
        require_int(tenant_source_counts["user_login"], path="tenant_source_counts.user_login"),
        require_int(tenant_source_counts["dns"], path="tenant_source_counts.dns"),
    ):
        raise ValueError("tenant count in both sources exceeds a source-specific tenant count")

    for key in ("tenant_disjoint", "time_forward", "calibration_from_benign_only", "test_not_used_for_threshold_selection"):
        if require_bool(partition_policy.get(key), path=f"replay.partition_policy.{key}") is not True:
            raise ValueError(f"partition policy is not satisfied: {key}")

    derived = {
        "source_total_rows": source_total,
        "resolved_rows": resolved_total,
        "unresolved_rows": unresolved_total,
        "resolved_replay_denominator": benign + positive,
        "positive_prevalence_percent": percent(positive, benign + positive),
        "unresolved_rate_percent": percent(unresolved, n_rows),
        "source_percentages": {source: percent(require_int(value, path=f"source_rows.{source}"), n_rows) for source, value in source_rows.items()},
        "event_count_residuals": {
            "user_login_hostnames": user_login_residual,
            "dns_resolution_hostnames": dns_residual,
        },
        "quality_repair_rows": sum_ints(repairs, path="replay.quality_repairs"),
        "quality_repair_rate_percent": percent(sum_ints(repairs, path="replay.quality_repairs"), n_rows),
        "observed_binary_agreement_percent": 100.0
        * require_number(
            agreement.get("observed_binary_agreement_excluding_unresolved"),
            path="replay.agreement.observed_binary_agreement_excluding_unresolved",
        ),
        "cohen_kappa_overall": require_number(agreement.get("cohen_kappa_overall"), path="replay.agreement.cohen_kappa_overall"),
        "cohen_kappa_dns": require_number(agreement.get("cohen_kappa_dns"), path="replay.agreement.cohen_kappa_dns"),
        "window_months": window_months,
        "n_tenants": n_tenants,
    }

    expected_source_percentages = require_mapping(replay.get("reported_source_percentages"), path="replay.reported_source_percentages")
    for source, expected in expected_source_percentages.items():
        assert_close(
            round(derived["source_percentages"][source], 2),
            require_number(expected, path=f"replay.reported_source_percentages.{source}"),
            path=f"replay.source_percentages.{source}",
            abs_tol=0.01,
        )

    return derived


def validate_verified_positive_profile(profile: Mapping[str, Any], positive_denominator: int) -> dict[str, Any]:
    denominator = require_int(profile.get("denominator"), path="verified_positive_profile.denominator")
    if denominator != positive_denominator:
        raise ValueError("verified-positive profile denominator does not match replay positive labels")

    families = profile.get("attack_family_distribution")
    if not isinstance(families, list) or not families:
        raise ValueError("verified_positive_profile.attack_family_distribution must be a non-empty list")
    clean_families: list[dict[str, Any]] = []
    for idx, row in enumerate(families, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"attack family row {idx} is not an object")
        family = str(row.get("family", "")).strip()
        if not family:
            raise ValueError(f"attack family row {idx} missing family")
        count = require_int(row.get("count"), path=f"attack_family_distribution.{family}.count")
        reported_share = require_number(row.get("share_percent"), path=f"attack_family_distribution.{family}.share_percent")
        computed_share = percent(count, denominator)
        assert_close(round(computed_share, 1), reported_share, path=f"attack_family_distribution.{family}.share_percent", abs_tol=0.1)
        clean_families.append({"family": family, "count": count, "share_percent": reported_share})
    family_total = sum(row["count"] for row in clean_families)
    if family_total != denominator:
        raise ValueError(f"attack-family counts sum to {family_total}, expected {denominator}")

    obfuscation = require_mapping(profile.get("obfuscation_distribution"), path="verified_positive_profile.obfuscation_distribution")
    obfuscation_total = sum_ints(obfuscation, path="verified_positive_profile.obfuscation_distribution")
    if obfuscation_total != denominator:
        raise ValueError(f"obfuscation counts sum to {obfuscation_total}, expected {denominator}")
    none_count = require_int(obfuscation.get("none"), path="verified_positive_profile.obfuscation_distribution.none")
    non_none = require_int(profile.get("non_none_obfuscation_count"), path="verified_positive_profile.non_none_obfuscation_count")
    if denominator - none_count != non_none:
        raise ValueError("non-none obfuscation count does not match denominator minus none count")

    largest = max(clean_families, key=lambda row: row["count"])
    second = sorted(clean_families, key=lambda row: row["count"], reverse=True)[1]
    return {
        "attack_family_count": len(clean_families),
        "attack_family_total": family_total,
        "largest_attack_family": largest["family"],
        "largest_attack_family_share_percent": largest["share_percent"],
        "second_attack_family": second["family"],
        "second_attack_family_share_percent": second["share_percent"],
        "no_family_exceeds_percent": largest["share_percent"],
        "obfuscation_total": obfuscation_total,
        "non_none_obfuscation_rate_percent": percent(non_none, denominator),
        "none_obfuscation_rate_percent": percent(none_count, denominator),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    counts_path = Path(args.counts)
    data = load_json(counts_path)
    replay = require_mapping(data.get("replay"), path="replay")
    profile = require_mapping(data.get("verified_positive_profile"), path="verified_positive_profile")
    replay_derived = validate_replay_profile(replay)
    positive_denominator = require_int(
        require_mapping(replay.get("label_counts"), path="replay.label_counts").get("verified_executable_semantics"),
        path="replay.label_counts.verified_executable_semantics",
    )
    profile_derived = validate_verified_positive_profile(profile, positive_denominator)

    return {
        "status": "pass",
        "counts_file": str(counts_path),
        "source": data.get("source", "published_aggregate_hib_profile"),
        "paper_sections": data.get("paper_sections", []),
        "replay": replay,
        "verified_positive_profile": profile,
        "derived": {
            "replay": replay_derived,
            "verified_positive_profile": profile_derived,
        },
        "public_reproduction_boundary": data.get("public_reproduction_boundary", {}),
        "private_by_design": data.get("private_by_design", []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute release-safe aggregate accounting for the HIB dataset profile.")
    parser.add_argument(
        "--counts",
        default="hib_profile/paper_hib_profile_counts.json",
        help="Release-safe aggregate HIB profile counts.",
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
