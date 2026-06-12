#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from validate_public_bundle import validate_bundle


PUBLIC_SCHEMA_FIELDS = [
    "public_row_id",
    "released_artifact",
    "released_canonical_artifact",
    "source_family",
    "time_bucket",
    "split",
    "label",
    "evidence_tier",
    "sink_family",
    "obfuscation_family",
    "released_length_bucket",
    "character_class_mask",
    "detector_outputs",
    "row_integrity_hash",
]

ALLOWED_NONLINKABILITY_RELEASE_FLAGS = {
    "raw_hostname_group_counts_released",
    "raw_hostname_group_existence_released",
    "raw_hostname_group_sizes_released",
    "raw_hostname_multiplicity_released",
    "private_raw_hostname_group_results_released",
    "stable_hostname_identifier_fields_released",
}

ALLOWED_PUBLIC_DUPLICATE_KEYS = {
    "n_duplicate_public_row_ids",
    "n_duplicate_released_artifact_values",
    "n_duplicate_released_canonical_values",
}


def find_private_duplicate_audit_key_leaks(value: Any, prefix: tuple[str, ...] = ()) -> list[str]:
    """Return audit keys that would disclose private raw-hostname recurrence.

    Public audits may say that private grouping/multiplicity facts were not
    released, and they may count duplicate values in the public release itself.
    They must not include counts, existence booleans, or group sizes for private
    raw/original hostnames.
    """
    leaks: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = (*prefix, str(key))
            key_lower = str(key).lower()
            if key_lower in ALLOWED_NONLINKABILITY_RELEASE_FLAGS or key_lower in ALLOWED_PUBLIC_DUPLICATE_KEYS:
                leaks.extend(find_private_duplicate_audit_key_leaks(child, path))
                continue
            duplicate_token = "duplicate" in key_lower or "dedup" in key_lower
            private_group_token = (
                ("raw_hostname" in key_lower or "private" in key_lower or "original" in key_lower)
                and any(token in key_lower for token in ("group", "multiplicity", "occurrence", "frequency", "recurrence"))
            )
            if duplicate_token or private_group_token:
                leaks.append(".".join(path))
            leaks.extend(find_private_duplicate_audit_key_leaks(child, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            leaks.extend(find_private_duplicate_audit_key_leaks(item, (*prefix, str(index))))
    return leaks


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def validate_sha_sidecar(target: Path, failures: list[str]) -> None:
    sidecar = target.with_suffix(target.suffix + ".sha256")
    require(sidecar.exists(), f"missing sha256 sidecar for {target}", failures)
    if not sidecar.exists():
        return
    parts = sidecar.read_text(encoding="utf-8").strip().split()
    require(len(parts) == 2, f"sha256 sidecar is not sha256sum-compatible: {sidecar}", failures)
    if len(parts) != 2:
        return
    digest, filename = parts
    require(filename == target.name, f"sha256 sidecar filename mismatch: {sidecar}", failures)
    require(digest == file_sha256(target), f"sha256 sidecar digest mismatch: {sidecar}", failures)


def validate_schema(schema_path: Path, failures: list[str]) -> None:
    require(schema_path.exists(), f"missing schema file: {schema_path}", failures)
    if not schema_path.exists():
        return
    schema = load_json(schema_path)
    require(schema.get("fields") == PUBLIC_SCHEMA_FIELDS, f"public schema fields do not match required schema: {schema_path}", failures)


def validate_audits(audit_dir: Path, release_rows: int | None, failures: list[str]) -> None:
    anon_path = audit_dir / "anonymization_audit_report.json"
    nonlink_path = audit_dir / "nonlinkability_audit_report.json"
    shortcut_path = audit_dir / "anonymization_shortcut_audit_report.json"
    metrics_path = audit_dir / "recomputed_public_metrics.json"
    data_card = audit_dir / "release_data_card.md"
    for path in (anon_path, nonlink_path, shortcut_path, data_card):
        require(path.exists(), f"missing required audit artifact: {path}", failures)
    if not anon_path.exists() or not nonlink_path.exists():
        return

    anon = load_json(anon_path)
    nonlink = load_json(nonlink_path)
    shortcut = load_json(shortcut_path) if shortcut_path.exists() else {}

    require(anon.get("status") == "pass", "anonymization audit status is not pass", failures)
    require(nonlink.get("status") == "pass", "nonlinkability audit status is not pass", failures)
    require(shortcut.get("status") in {"pass", None}, "shortcut audit status is not pass", failures)
    for leaked_key in find_private_duplicate_audit_key_leaks(nonlink):
        failures.append(f"nonlinkability audit leaks private raw-hostname duplicate/group key: {leaked_key}")

    privacy = anon.get("privacy_safety_checks", {})
    zero_privacy_keys = [
        "n_forbidden_public_fields",
        "emails_usernames_user_ids_device_ids_blockers",
        "ip_addresses_outside_documentation_ranges",
        "secrets_api_keys_jwts_tokens_signed_urls",
        "raw_internal_suffixes_private_tlds",
        "live_callback_domains",
        "exact_raw_hostname_strings",
        "exact_raw_canonical_hostname_strings",
        "unsafe_executable_payloads_after_inerting",
        "intent_signaling_generated_hostnames",
        "manual_privacy_review_blockers",
    ]
    for key in zero_privacy_keys:
        require(privacy.get(key) == 0, f"privacy safety check is nonzero: {key}={privacy.get(key)}", failures)

    expected = anon.get("public_expected_row_checks", {})
    for key in (
        "n_missing_public_rows",
        "n_unmatched_public_rows",
        "n_integrity_failures",
        "n_label_mismatches",
        "n_released_artifact_mismatches",
        "n_released_canonical_artifact_mismatches",
    ):
        if key in expected:
            require(expected.get(key) == 0, f"public expected-row check is nonzero: {key}={expected.get(key)}", failures)
    if expected:
        require(expected.get("status") == "pass", "public expected-row check status is not pass", failures)

    llm = anon.get("llm_label_reason_handling", {})
    require(llm.get("labels_preserved") is True, "LLM/resolved labels are not marked preserved", failures)
    require(llm.get("n_label_mismatches") == 0, "LLM/resolved label mismatch count is nonzero", failures)
    require(llm.get("public_llm_reasons_released") is False, "raw LLM reasons are marked as public", failures)

    utility = anon.get("utility_reproducibility_checks", {})
    require(utility.get("label_preservation_rate") == 1.0, "label preservation rate is not 1.0", failures)

    private_linkage = nonlink.get("private_origin_linkage_checks", {})
    public_uniqueness = nonlink.get("public_uniqueness_checks", {})
    access = nonlink.get("access_pattern_checks", {})
    fingerprint = nonlink.get("structural_fingerprint_checks", {})
    website = nonlink.get("website_access_pattern_audit", {})
    for section_name, section in (
        ("private_origin_linkage_checks", private_linkage),
        ("public_uniqueness_checks", public_uniqueness),
        ("access_pattern_checks", access),
        ("structural_fingerprint_checks", fingerprint),
        ("website_access_pattern_audit", website),
    ):
        require(section.get("status") == "pass", f"nonlinkability section status is not pass: {section_name}", failures)

    for key in (
        "raw_hostname_group_counts_released",
        "raw_hostname_group_existence_released",
        "raw_hostname_multiplicity_released",
        "stable_hostname_identifier_fields_released",
    ):
        if key in private_linkage:
            require(private_linkage.get(key) is False, f"private linkage field is marked released: {key}", failures)
    for key in (
        "n_duplicate_public_row_ids",
        "n_duplicate_released_artifact_values",
        "n_duplicate_released_canonical_values",
        "n_forbidden_stable_hostname_ids",
        "n_forbidden_stable_hostname_hashes",
    ):
        if key in public_uniqueness:
            require(public_uniqueness.get(key) == 0, f"public uniqueness check is nonzero: {key}={public_uniqueness.get(key)}", failures)
    require(access.get("n_sparse_public_combinations") == 0, "sparse public combinations remain", failures)
    for key in (
        "private_raw_hostname_group_results_released",
        "raw_hostname_group_counts_released",
        "raw_hostname_group_existence_released",
        "raw_hostname_group_sizes_released",
        "stable_hostname_identifier_fields_released",
    ):
        if key in fingerprint:
            require(fingerprint.get(key) is False, f"structural fingerprint private result is marked released: {key}", failures)
        if key in website:
            require(website.get(key) is False, f"website access-pattern private result is marked released: {key}", failures)

    if release_rows is not None:
        require(anon.get("n_public_rows") == release_rows, "anonymization audit row count does not match release", failures)
        require(nonlink.get("n_public_rows") == release_rows, "nonlinkability audit row count does not match release", failures)
        if metrics_path.exists():
            metrics = load_json(metrics_path)
            require(metrics.get("n_rows") == release_rows, "recomputed metrics row count does not match release", failures)
            require(isinstance(metrics.get("label_accounting"), dict), "recomputed metrics missing label accounting", failures)
            require(isinstance(metrics.get("calibration"), dict), "recomputed metrics missing calibration accounting", failures)
            require(isinstance(metrics.get("fixed_fpr_replay"), dict), "recomputed metrics missing fixed-FPR replay block", failures)
            require(isinstance(metrics.get("detector_overlap"), dict), "recomputed metrics missing detector-overlap block", failures)
            label_accounting = metrics.get("label_accounting", {})
            fixed_fpr = metrics.get("fixed_fpr_replay", {})
            if isinstance(label_accounting, dict):
                require(
                    label_accounting.get("unresolved_excluded_from_tpr_fpr") is True,
                    "recomputed metrics do not mark unresolved rows as excluded from TPR/FPR",
                    failures,
                )
            if isinstance(fixed_fpr, dict):
                require(fixed_fpr.get("requested_fpr") is not None, "fixed-FPR replay block missing requested_fpr", failures)
                require(fixed_fpr.get("status") in {"available", "not_available"}, "fixed-FPR replay status is invalid", failures)


def count_jsonl_rows(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            count += chunk.count(b"\n")
    return count


def validate_release_gate(release: Path, audit_dir: Path, bundle: Path | None = None, *, count_rows: bool = False) -> dict[str, Any]:
    failures: list[str] = []
    require(release.exists(), f"missing public release: {release}", failures)
    if release.exists():
        validate_sha_sidecar(release, failures)
        validate_schema(release.with_suffix(".schema.json"), failures)
    release_rows = count_jsonl_rows(release) if count_rows and release.exists() else None
    validate_audits(audit_dir, release_rows, failures)
    if bundle is not None:
        try:
            validate_bundle(bundle)
        except Exception as exc:  # noqa: BLE001 - CLI validation should report all failures uniformly.
            failures.append(f"public bundle validation failed: {exc}")
    return {
        "release": str(release),
        "audit_dir": str(audit_dir),
        "bundle": str(bundle) if bundle else None,
        "row_count_checked": release_rows,
        "failures": failures,
        "status": "pass" if not failures else "fail",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate final HIB release gate artifacts after full verification.")
    parser.add_argument("--public-release", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, default=None)
    parser.add_argument("--count-rows", action="store_true", help="Count JSONL rows and compare them with audit/metric row counts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_release_gate(args.public_release, args.audit_dir, args.bundle, count_rows=args.count_rows)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
