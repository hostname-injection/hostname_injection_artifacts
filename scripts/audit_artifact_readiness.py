#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import sys
import tomllib
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEID_SCRIPTS = ROOT / "deidentification_release" / "scripts"
FORBIDDEN_PUBLIC_PATTERNS = tuple(
    "".join(parts)
    for parts in (
        ("duplicate", "_private", "_hostname", "_groups"),
        ("public", "_duplicate", "_checks"),
        ("n", "_private", "_duplicate"),
        ("n", "_groups", "_size", "_gt", "_1"),
        ("n", "_rows", "_in", "_groups", "_size", "_gt", "_1"),
        ("max", "_private", "_group", "_size"),
        ("n", "_groups", "_publicly"),
        ("private", "_duplicate"),
        ("Duplicate", " private", " hostnames"),
        ("duplicate", " private", " hostname"),
        ("Repeated", " private", " hostnames"),
        ("deduplicated", "-hostname", " and tenant", " mappings"),
    )
)
FORBIDDEN_PORTABILITY_PATTERNS = tuple(
    "".join(parts)
    for parts in (
        ("/media", "/sameer/"),
        ("/Users", "/sameer"),
        ("/home", "/sameer"),
    )
)
FORBIDDEN_TRACKING_PATTERNS = tuple(
    "".join(parts)
    for parts in (
        ("google", "-analytics.com"),
        ("googletag", "manager.com"),
        ("analytics", ".js"),
        ("gtag", "("),
        ("ga(", "'create"),
        ('ga(', '"create'),
        ("mixpanel", ".init"),
        ("cdn.segment", ".com"),
        ("segment", ".io"),
        ("amplitude", ".getinstance"),
        ("post", "hog.init"),
        ("plausible", ".io/js"),
        ("static.hot", "jar.com"),
        ("full", "story.com"),
        ("clarity", ".ms/tag"),
        ("sentry", ".io"),
        ("new", "relic"),
        ("datadog", "-rum"),
    )
)
PLACEHOLDER_PATTERNS = ("example.org", "REPLACE_WITH")
MODEL_ARTIFACT_SUFFIXES = {".ckpt", ".pt", ".pth", ".safetensors"}
MODEL_ARTIFACT_DIR_SUFFIX = "_model_checkpoint"
MODEL_ARTIFACT_EXCLUDED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "ccd.egg-info",
    "dist",
    "htmlcov",
    "out",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_text_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_file() and child.suffix.lower() not in {".gz", ".zip", ".pt", ".safetensors", ".npz", ".pyc"}:
                yield child


def iter_package_visible_paths(root: Path) -> Iterable[Path]:
    for child in sorted(root.rglob("*")):
        rel = child.relative_to(root)
        if any(part in MODEL_ARTIFACT_EXCLUDED_DIRS for part in rel.parts):
            continue
        yield child


def check_no_pretrained_model_artifacts(manifest: dict[str, Any]) -> list[str]:
    del manifest
    failures: list[str] = []
    for path in iter_package_visible_paths(ROOT):
        rel = path.relative_to(ROOT).as_posix()
        if path.is_dir():
            if path.name.endswith(MODEL_ARTIFACT_DIR_SUFFIX):
                failures.append(f"pretrained/checkpoint directory is package-visible: {rel}")
            continue
        if path.suffix in MODEL_ARTIFACT_SUFFIXES:
            failures.append(f"pretrained/checkpoint weight file is package-visible: {rel}")
    return failures


def check_required_files(manifest: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for rel in manifest.get("required_files", []):
        path = ROOT / rel
        if not path.exists():
            failures.append(f"missing required artifact path: {rel}")
    return failures


def check_claims(manifest: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    claims = manifest.get("claims")
    if not isinstance(claims, list) or not claims:
        return ["manifest has no claims"]
    for idx, claim in enumerate(claims, start=1):
        if not isinstance(claim, dict):
            failures.append(f"claim {idx} is not an object")
            continue
        for key in ("claim", "script", "expected"):
            if not str(claim.get(key, "")).strip():
                failures.append(f"claim {idx} missing {key}")
    return failures


def iter_claim_path_candidates(token: str) -> Iterable[str]:
    if token.startswith(("http://", "https://", "-")):
        return
    if "=" in token:
        key, value = token.split("=", 1)
        if key.isidentifier() and key.upper() in {"PYTHONPATH", "PATH"}:
            for part in value.split(":"):
                yield from iter_claim_path_candidates(part)
            return
        if key.isidentifier():
            token = value
    if token.startswith(("http://", "https://", "-", "/tmp/", "/path/to/")):
        return
    if token.startswith("/"):
        return
    token = token.split("#", 1)[0]
    if not token or any(char in token for char in "*?[]{}"):
        return
    if "/" in token or token.endswith((".py", ".md", ".json", ".jsonl", ".gz", ".toml", ".yml", ".yaml", ".npz")):
        yield token


def check_claim_script_targets(manifest: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for idx, claim in enumerate(manifest.get("claims", []), start=1):
        if not isinstance(claim, dict):
            continue
        script = str(claim.get("script", "")).strip()
        if script.startswith("See "):
            script = script.removeprefix("See ").strip()
        try:
            tokens = shlex.split(script)
        except ValueError as exc:
            failures.append(f"claim {idx} script is not shell-parseable: {exc}")
            continue
        for token in tokens:
            for rel in iter_claim_path_candidates(token):
                if not (ROOT / rel).exists():
                    failures.append(f"claim {idx} references missing path: {rel}")
    return failures


def check_metadata_template(manifest: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    rel = manifest.get("metadata_template")
    if not rel:
        return ["manifest missing metadata_template"]
    path = ROOT / str(rel)
    if not path.exists():
        return [f"metadata template missing: {rel}"]
    try:
        metadata = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        return [f"metadata template is not valid TOML: {exc}"]

    required = [
        "badge",
        "artifact_url",
        "cd",
        "citation",
        "license_url",
        "install_script",
        "use",
        "destructive",
        "hw",
        "sw",
        "api",
        "gui",
        "provenance",
        "ethics",
        "readme",
    ]
    for key in required:
        if not str(metadata.get(key, "")).strip():
            failures.append(f"metadata template missing {key}")
    claim_indices = sorted(
        int(key.removeprefix("claim"))
        for key in metadata
        if key.startswith("claim") and key.removeprefix("claim").isdigit()
    )
    if len(claim_indices) < 6:
        failures.append("metadata template should include at least six claim blocks")
    manifest_claim_count = len(manifest.get("claims", [])) if isinstance(manifest.get("claims"), list) else 0
    if manifest_claim_count and len(claim_indices) < manifest_claim_count:
        failures.append(
            f"metadata template has {len(claim_indices)} claim block(s), fewer than manifest claims {manifest_claim_count}"
        )
    for idx in claim_indices:
        for key in (f"claim{idx}", f"script{idx}", f"expected{idx}"):
            if not str(metadata.get(key, "")).strip():
                failures.append(f"metadata template missing {key}")
    if metadata.get("badge") != "r":
        failures.append("metadata template should request reproduced badge with badge='r'")
    if metadata.get("cd") != "b":
        failures.append("metadata template should identify artifact as both code and dataset with cd='b'")
    return failures


def check_badge_basics(manifest: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    target_badges = set(manifest.get("target_badges", []))
    for badge in ("Available", "Functional", "Reproduced"):
        if badge not in target_badges:
            failures.append(f"target_badges missing {badge}")

    for key in ("primary_readme", "badge_readiness", "evaluation_guide", "resource_notes", "provenance_and_ethics"):
        rel = manifest.get(key)
        if not rel:
            failures.append(f"manifest missing {key}")
        elif not (ROOT / str(rel)).exists():
            failures.append(f"{key} path does not exist: {rel}")

    kick = manifest.get("kick_the_tires", {})
    if not isinstance(kick, dict):
        return failures + ["kick_the_tires is not an object"]
    if "run_artifact_smoke.py" not in str(kick.get("command", "")):
        failures.append("kick_the_tires command should run scripts/run_artifact_smoke.py")
    if "pass" not in str(kick.get("expected", "")).lower():
        failures.append("kick_the_tires expected result should name a pass condition")
    if kick.get("network_required") is not False:
        failures.append("kick_the_tires.network_required should be false")
    if kick.get("gpu_required") is not False:
        failures.append("kick_the_tires.gpu_required should be false")

    full_tests = manifest.get("full_tests", {})
    if not isinstance(full_tests, dict):
        failures.append("full_tests should be an object")
    else:
        if "pytest" not in str(full_tests.get("command", "")):
            failures.append("full_tests command should run pytest")
        observed = str(full_tests.get("last_observed", "")).strip().lower()
        if not observed:
            failures.append("full_tests.last_observed is missing")
        elif "passed" not in observed:
            failures.append("full_tests.last_observed should record a passing pytest result")
    return failures


def check_evidence_paths(value: Any, failures: list[str], context: str) -> None:
    if not isinstance(value, list) or not value:
        failures.append(f"{context} should list at least one evidence path")
        return
    for item in value:
        rel = str(item).split("#", 1)[0]
        if not rel or not (ROOT / rel).exists():
            failures.append(f"{context} evidence path missing or does not exist: {item}")


def check_ieee_sp_requirements(manifest: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    requirements = manifest.get("ieee_sp_artifact_requirements")
    if not isinstance(requirements, dict):
        return ["manifest missing ieee_sp_artifact_requirements"]

    if "sp2027.ieee-security.org/artifact_instructions.html" not in str(requirements.get("instruction_url", "")):
        failures.append("IEEE S&P instruction URL is missing or unexpected")

    criteria = requirements.get("badge_criteria")
    if not isinstance(criteria, dict):
        failures.append("IEEE S&P badge criteria matrix is missing")
    else:
        for badge in ("Available", "Functional", "Reproduced"):
            if badge not in criteria:
                failures.append(f"IEEE S&P badge criteria missing {badge}")

        available = criteria.get("Available", {})
        if isinstance(available, dict):
            if available.get("permanent_public_retrieval_required") is not True:
                failures.append("Available criterion should require permanent public retrieval")
            if available.get("doi_required") is not True:
                failures.append("Available criterion should require DOI-backed storage")
            if "external" not in str(available.get("current_status", "")).lower():
                failures.append("Available current status should record the external publication boundary")
            check_evidence_paths(available.get("evidence"), failures, "Available criterion")

        functional = criteria.get("Functional", {})
        if isinstance(functional, dict):
            for aspect in ("documentation", "completeness", "exercisability", "other_machine_portability"):
                aspect_block = functional.get(aspect)
                if not isinstance(aspect_block, dict):
                    failures.append(f"Functional criterion missing aspect: {aspect}")
                    continue
                if aspect_block.get("supported") is not True:
                    failures.append(f"Functional {aspect} support should be marked true")
                check_evidence_paths(aspect_block.get("evidence"), failures, f"Functional {aspect}")

        reproduced = criteria.get("Reproduced", {})
        if isinstance(reproduced, dict):
            for key in (
                "main_results_supported",
                "allowed_tolerance_documented",
                "scaled_down_for_lengthy_experiments",
                "external_full_data_required",
            ):
                if reproduced.get(key) is not True:
                    failures.append(f"Reproduced criterion should mark {key}=true")
            repeat_rel = str(reproduced.get("independent_repeat_path", "")).split("#", 1)[0]
            if not repeat_rel or not (ROOT / repeat_rel).exists():
                failures.append(
                    "Reproduced independent repeat path missing or does not exist: "
                    f"{reproduced.get('independent_repeat_path')}"
                )
            check_evidence_paths(reproduced.get("evidence"), failures, "Reproduced criterion")

    if requirements.get("aec_contact_required") is not True:
        failures.append("AEC contact requirement should be acknowledged")
    if not str(requirements.get("aec_contact_note", "")).strip():
        failures.append("AEC contact note is missing")

    public_infra = requirements.get("public_research_infrastructure", {})
    if not isinstance(public_infra, dict) or public_infra.get("supported") is not True:
        failures.append("public research infrastructure support should be marked true")
    else:
        notes = str(public_infra.get("notes", "")).lower()
        for term in ("public research infrastructure", "no ssh", "no", "gpu"):
            if term not in notes:
                failures.append(f"public infrastructure notes should mention {term!r}")
        metadata_field = str(public_infra.get("metadata_field", ""))
        metadata_rel = manifest.get("metadata_template")
        if metadata_field and metadata_rel and (ROOT / str(metadata_rel)).exists():
            metadata = tomllib.loads((ROOT / str(metadata_rel)).read_text(encoding="utf-8"))
            if not str(metadata.get(metadata_field, "")).strip():
                failures.append(f"metadata template missing public infrastructure field: {metadata_field}")

    runtime = requirements.get("runtime", {})
    if not isinstance(runtime, dict):
        failures.append("runtime requirements must be an object")
    else:
        if int(runtime.get("aec_limit_hours", 0)) != 24:
            failures.append("AEC runtime limit should be recorded as 24 hours")
        for key in ("kick_the_tires_runtime", "full_tests_runtime", "scaled_down_justification"):
            if not str(runtime.get(key, "")).strip():
                failures.append(f"runtime requirement missing {key}")
        if runtime.get("scaled_down_experiments_justified") is not True:
            failures.append("scaled-down experiment justification should be marked true")

    packaging = requirements.get("packaging", {})
    if not isinstance(packaging, dict):
        failures.append("packaging requirements must be an object")
    else:
        if packaging.get("source_package") is not True:
            failures.append("source-package preference should be marked true")
        if packaging.get("container_required") is not False:
            failures.append("container_required should be false for this source artifact")
        if packaging.get("hotcrp_metadata_required") is not True:
            failures.append("HotCRP metadata requirement should be marked true")
        if packaging.get("all_relevant_information_in_packaging") is not True:
            failures.append("all relevant information should be marked as packaged")
        for key in ("metadata_template", "archive_builder"):
            rel = packaging.get(key)
            if not rel or not (ROOT / str(rel)).exists():
                failures.append(f"packaging path missing or does not exist: {key}={rel}")

    claims = requirements.get("claims", {})
    if not isinstance(claims, dict):
        failures.append("claim requirements must be an object")
    else:
        if claims.get("concrete_claims_required") is not True:
            failures.append("concrete claim requirement should be marked true")
        for key in ("claim_map", "machine_readable_manifest", "paper_claim_coverage", "paper_headline_claims"):
            rel = str(claims.get(key, "")).split("#", 1)[0]
            if not rel or not (ROOT / rel).exists():
                failures.append(f"claim requirement path missing or does not exist: {key}={claims.get(key)}")

    release = requirements.get("release_expectation", {})
    if not isinstance(release, dict):
        failures.append("release expectation requirements must be an object")
    else:
        if release.get("meaningful_public_release_expected") is not True:
            failures.append("meaningful public release expectation should be marked true")
        if release.get("full_evaluated_release_external") is not True:
            failures.append("full evaluated release external boundary should be marked true")
        for key in ("private_by_design_document",):
            rel = release.get(key)
            if not rel or not (ROOT / str(rel)).exists():
                failures.append(f"release expectation path missing or does not exist: {key}={rel}")
        if "strict-final" not in str(release.get("strict_final_gate", "")):
            failures.append("strict final gate should reference --strict-final")

    tracking = requirements.get("tracking", {})
    if not isinstance(tracking, dict):
        failures.append("tracking requirements must be an object")
    else:
        if tracking.get("web_tracking_embedded") is not False:
            failures.append("web tracking should be marked absent")
        if tracking.get("tracking_scan_required") is not True:
            failures.append("tracking scan requirement should be marked true")
        scan_key = str(tracking.get("tracking_scan_paths", ""))
        if scan_key not in manifest or not manifest.get(scan_key):
            failures.append(f"tracking scan key is missing from manifest: {scan_key}")
    return failures


def check_public_privacy_scan(manifest: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for rel in manifest.get("privacy_scan_paths", []):
        path = ROOT / rel
        if not path.exists():
            failures.append(f"privacy scan path missing: {rel}")
            continue
        for text_path in iter_text_files(path):
            try:
                text = text_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for pattern in FORBIDDEN_PUBLIC_PATTERNS:
                if pattern in text:
                    failures.append(f"forbidden public wording found in {text_path.relative_to(ROOT)}: {pattern}")
    return failures


def check_portability_scan(manifest: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for rel in manifest.get("portability_scan_paths", []):
        path = ROOT / rel
        if not path.exists():
            failures.append(f"portability scan path missing: {rel}")
            continue
        for text_path in iter_text_files(path):
            try:
                text = text_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for pattern in FORBIDDEN_PORTABILITY_PATTERNS:
                if pattern in text:
                    failures.append(f"author-local path found in {text_path.relative_to(ROOT)}: {pattern}")
    return failures


def check_tracking_scan(manifest: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    scan_paths = manifest.get("tracking_scan_paths", manifest.get("portability_scan_paths", []))
    for rel in scan_paths:
        path = ROOT / rel
        if not path.exists():
            failures.append(f"tracking scan path missing: {rel}")
            continue
        for text_path in iter_text_files(path):
            try:
                text = text_path.read_text(encoding="utf-8").lower()
            except UnicodeDecodeError:
                continue
            for pattern in FORBIDDEN_TRACKING_PATTERNS:
                if pattern in text:
                    failures.append(f"web tracking pattern found in {text_path.relative_to(ROOT)}: {pattern}")
    return failures


def check_public_release_gate(manifest: dict[str, Any]) -> list[str]:
    sys.path.insert(0, str(DEID_SCRIPTS))
    from validate_public_bundle import validate_bundle
    from validate_release_gate import validate_release_gate

    failures: list[str] = []
    bundle_info = manifest.get("public_release_bundle", {})
    bundle = ROOT / str(bundle_info.get("bundle", ""))
    release = ROOT / str(bundle_info.get("sample_release", ""))
    audit_dir = ROOT / str(bundle_info.get("audit_dir", ""))
    expected_rows = bundle_info.get("row_count")

    try:
        bundle_result = validate_bundle(bundle)
    except Exception as exc:  # noqa: BLE001 - readiness audit reports validation failures uniformly.
        failures.append(f"public bundle validation failed: {exc}")
        bundle_result = None

    try:
        gate_result = validate_release_gate(release, audit_dir, bundle, count_rows=True)
    except Exception as exc:  # noqa: BLE001
        failures.append(f"release gate raised: {exc}")
        gate_result = None

    if bundle_result is not None and bundle_result.get("status") != "pass":
        failures.append("public bundle status is not pass")
    if gate_result is not None:
        if gate_result.get("status") != "pass":
            failures.extend(str(failure) for failure in gate_result.get("failures", []))
        if expected_rows is not None and gate_result.get("row_count_checked") != expected_rows:
            failures.append(
                f"release gate row count {gate_result.get('row_count_checked')} does not match manifest row_count {expected_rows}"
            )

    metrics_expectations = bundle_info.get("metrics_expectations", {})
    if isinstance(metrics_expectations, dict) and metrics_expectations:
        metrics_path = audit_dir / "recomputed_public_metrics.json"
        if not metrics_path.exists():
            failures.append(f"missing recomputed metrics: {metrics_path.relative_to(ROOT)}")
        else:
            metrics = load_json(metrics_path)
            if expected_rows is not None and metrics.get("n_rows") != expected_rows:
                failures.append(f"recomputed metrics n_rows {metrics.get('n_rows')} does not match manifest row_count {expected_rows}")
            calibration = metrics.get("calibration", {})
            fixed_fpr = metrics.get("fixed_fpr_replay", {})
            label_accounting = metrics.get("label_accounting", {})
            expected_status = metrics_expectations.get("fixed_fpr_status")
            expected_threshold_source = metrics_expectations.get("threshold_source")
            if expected_status and fixed_fpr.get("status") != expected_status:
                failures.append(f"fixed-FPR status {fixed_fpr.get('status')} does not match expected {expected_status}")
            if expected_threshold_source and fixed_fpr.get("threshold_source") != expected_threshold_source:
                failures.append(
                    f"fixed-FPR threshold source {fixed_fpr.get('threshold_source')} does not match expected {expected_threshold_source}"
                )
            min_scored = int(metrics_expectations.get("min_scored_benign_calibration_rows", 0))
            if calibration.get("scored_benign_calibration_rows", 0) < min_scored:
                failures.append(
                    "scored benign calibration rows "
                    f"{calibration.get('scored_benign_calibration_rows')} below expected minimum {min_scored}"
                )
            min_groups = int(metrics_expectations.get("min_calibration_groups", 0))
            if calibration.get("n_calibration_groups", 0) < min_groups:
                failures.append(f"calibration groups {calibration.get('n_calibration_groups')} below expected minimum {min_groups}")
            min_pos = int(metrics_expectations.get("min_metric_positive_rows", 0))
            if label_accounting.get("metric_positive_rows", 0) < min_pos:
                failures.append(f"metric positive rows {label_accounting.get('metric_positive_rows')} below expected minimum {min_pos}")
            min_neg = int(metrics_expectations.get("min_metric_negative_rows", 0))
            if label_accounting.get("metric_negative_rows", 0) < min_neg:
                failures.append(f"metric negative rows {label_accounting.get('metric_negative_rows')} below expected minimum {min_neg}")
            for key in ("tp", "fp", "tn", "fn"):
                min_key = f"min_fixed_fpr_{key}"
                minimum = int(metrics_expectations.get(min_key, 0))
                if fixed_fpr.get(key, 0) < minimum:
                    failures.append(f"fixed-FPR {key} {fixed_fpr.get(key)} below expected minimum {minimum}")
    return failures


def check_final_publication_readiness(manifest: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    metadata_rel = manifest.get("metadata_template")
    if metadata_rel:
        metadata_path = ROOT / str(metadata_rel)
        if metadata_path.exists():
            metadata_text = metadata_path.read_text(encoding="utf-8")
            for pattern in PLACEHOLDER_PATTERNS:
                if pattern in metadata_text:
                    failures.append(f"metadata template still contains placeholder: {pattern}")
        else:
            failures.append(f"metadata template missing for final publication check: {metadata_rel}")

    external_items = [str(item).strip() for item in manifest.get("external_completion_items", []) if str(item).strip()]
    if external_items:
        failures.append(f"{len(external_items)} external completion item(s) remain before final badge readiness")
        failures.extend(f"external completion item remains: {item}" for item in external_items)
    return failures


def audit(manifest_path: Path, *, skip_gates: bool = False, strict_final: bool = False) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    checks = {
        "required_files": check_required_files(manifest),
        "badge_basics": check_badge_basics(manifest),
        "ieee_sp_requirements": check_ieee_sp_requirements(manifest),
        "claims": check_claims(manifest),
        "claim_script_targets": check_claim_script_targets(manifest),
        "metadata_template": check_metadata_template(manifest),
        "model_artifact_scan": check_no_pretrained_model_artifacts(manifest),
        "public_privacy_scan": check_public_privacy_scan(manifest),
        "portability_scan": check_portability_scan(manifest),
        "tracking_scan": check_tracking_scan(manifest),
    }
    if not skip_gates:
        checks["public_release_gate"] = check_public_release_gate(manifest)
    if strict_final:
        checks["final_publication_readiness"] = check_final_publication_readiness(manifest)
    failures = [failure for failures in checks.values() for failure in failures]
    return {
        "manifest": str(manifest_path),
        "checks": {name: "pass" if not failures else "fail" for name, failures in checks.items()},
        "failures": failures,
        "status": "pass" if not failures else "fail",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit local IEEE S&P artifact readiness metadata and public release gates.")
    parser.add_argument("--manifest", type=Path, default=ROOT / "ARTIFACT_MANIFEST.json")
    parser.add_argument("--skip-gates", action="store_true", help="Only check manifest structure, required files, and public wording.")
    parser.add_argument(
        "--strict-final",
        action="store_true",
        help="Also fail if DOI/publication placeholders or external completion items remain.",
    )
    args = parser.parse_args()

    result = audit(args.manifest, skip_gates=args.skip_gates, strict_final=args.strict_final)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
