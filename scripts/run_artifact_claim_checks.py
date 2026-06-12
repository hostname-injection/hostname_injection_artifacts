#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEID_SCRIPTS = ROOT / "deidentification_release" / "scripts"


def deid_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(DEID_SCRIPTS) if not existing else f"{DEID_SCRIPTS}{os.pathsep}{existing}"
    return env


def command_string(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def standard_checks(python: str) -> list[dict[str, Any]]:
    return [
        {
            "id": "readiness_audit",
            "description": "Manifest, IEEE badge metadata, privacy, portability, tracking, and public release gates.",
            "command": [python, "scripts/audit_artifact_readiness.py"],
        },
        {
            "id": "method_contracts",
            "description": "Table 1 and Appendix C CCD/CAHO method contracts.",
            "command": [python, "scripts/recompute_method_contracts.py"],
        },
        {
            "id": "paper_claim_coverage",
            "description": "Named contribution, figure, table, equation, and appendix coverage.",
            "command": [python, "scripts/recompute_paper_claim_coverage.py"],
        },
        {
            "id": "paper_headline_claims",
            "description": "Abstract/contribution/conclusion headline numeric anchors.",
            "command": [python, "scripts/recompute_paper_headline_claims.py"],
        },
        {
            "id": "hib_profile",
            "description": "HIB dataset profile, labels, repairs, and verified-positive profile.",
            "command": [python, "scripts/recompute_hib_profile_metrics.py"],
        },
        {
            "id": "evaluation_accounting",
            "description": "Table 2 evidence units and Appendix E/Table 11 reproducibility boundaries.",
            "command": [python, "scripts/recompute_evaluation_accounting.py"],
        },
        {
            "id": "source_reachability",
            "description": "50-repo CodeQL/Semgrep source-reachability aggregate accounting.",
            "command": [python, "scripts/recompute_source_reachability_metrics.py"],
        },
        {
            "id": "public_scope",
            "description": "Public-report taxonomy scope and public-anchor accounting.",
            "command": [python, "scripts/recompute_public_scope_metrics.py"],
        },
        {
            "id": "production_latency",
            "description": "Figure 5 and Table 5 production latency/throughput aggregates.",
            "command": [python, "scripts/recompute_production_latency_metrics.py"],
        },
        {
            "id": "live_overlap",
            "description": "Table 7 CCD-vs-Regex/WAF live-overlap accounting.",
            "command": [python, "scripts/recompute_live_overlap_metrics.py"],
        },
        {
            "id": "sink_evidence",
            "description": "Table 8 controlled metadata-to-code sink-evidence traces.",
            "command": [python, "scripts/recompute_sink_evidence_metrics.py"],
        },
        {
            "id": "paper_metric_tables",
            "description": "Tables 5, 6, 10, and 12 plus Appendix F aggregate metrics.",
            "command": [python, "scripts/recompute_paper_metric_tables.py"],
        },
        {
            "id": "stability_scope",
            "description": "Figure 6/Figure 7 stability, drift, holdout, and public-real scope aggregates.",
            "command": [python, "scripts/recompute_stability_scope_metrics.py"],
        },
    ]


def public_replay_check(python: str, metrics_out: Path) -> dict[str, Any]:
    return {
        "id": "public_sample_replay_metrics",
        "description": "Checked-in HIB public sample replay metrics and fixed-FPR accounting.",
        "command": [
            python,
            "deidentification_release/scripts/recompute_metrics.py",
            "--public-release",
            "deidentification_release/data/release/hib_release.jsonl",
            "--alpha",
            "1e-4",
            "--out",
            str(metrics_out),
        ],
        "json_output_path": metrics_out,
        "env": deid_env(),
    }


def parse_json_output(completed: subprocess.CompletedProcess[str], json_output_path: Path | None) -> tuple[dict[str, Any] | None, str | None]:
    try:
        if json_output_path is not None:
            return json.loads(json_output_path.read_text(encoding="utf-8")), None
        return json.loads(completed.stdout), None
    except (json.JSONDecodeError, OSError) as exc:
        return None, str(exc)


def validate_report(check_id: str, report: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if check_id == "public_sample_replay_metrics":
        if report.get("n_rows") != 150:
            failures.append(f"{check_id}: expected n_rows=150, observed {report.get('n_rows')}")
        fixed_fpr = report.get("fixed_fpr_replay", {})
        if not isinstance(fixed_fpr, dict) or fixed_fpr.get("status") != "available":
            failures.append(f"{check_id}: fixed_fpr_replay.status is not available")
        if fixed_fpr.get("threshold_source") != "recomputed_from_public_grouped_calibration_scores":
            failures.append(f"{check_id}: unexpected threshold source {fixed_fpr.get('threshold_source')}")
        for key in ("tp", "fp", "tn", "fn"):
            if fixed_fpr.get(key, 0) < 1:
                failures.append(f"{check_id}: fixed_fpr_replay.{key} is below 1")
        return failures

    if report.get("status") != "pass":
        failures.append(f"{check_id}: JSON status is not pass")
    return failures


def summarize_report(check_id: str, report: dict[str, Any]) -> dict[str, Any]:
    if check_id == "readiness_audit":
        return {
            "reported_status": report.get("status"),
            "checks": report.get("checks", {}),
        }
    if check_id == "public_sample_replay_metrics":
        fixed_fpr = report.get("fixed_fpr_replay", {})
        label_accounting = report.get("label_accounting", {})
        return {
            "n_rows": report.get("n_rows"),
            "fixed_fpr_status": fixed_fpr.get("status") if isinstance(fixed_fpr, dict) else None,
            "threshold_source": fixed_fpr.get("threshold_source") if isinstance(fixed_fpr, dict) else None,
            "fixed_fpr_confusion": {
                key: fixed_fpr.get(key) for key in ("tp", "fp", "tn", "fn") if isinstance(fixed_fpr, dict)
            },
            "metric_positive_rows": label_accounting.get("metric_positive_rows")
            if isinstance(label_accounting, dict)
            else None,
            "metric_negative_rows": label_accounting.get("metric_negative_rows")
            if isinstance(label_accounting, dict)
            else None,
        }
    derived = report.get("derived")
    return {
        "reported_status": report.get("status"),
        "derived_keys": sorted(derived) if isinstance(derived, dict) else [],
    }


def run_one(check: dict[str, Any], *, root: Path) -> dict[str, Any]:
    start = time.monotonic()
    command = [str(part) for part in check["command"]]
    try:
        completed = subprocess.run(
            command,
            cwd=root,
            env=check.get("env"),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        duration = time.monotonic() - start
        return {
            "id": check["id"],
            "description": check["description"],
            "command": command_string(command),
            "returncode": None,
            "duration_seconds": round(duration, 3),
            "status": "fail",
            "failures": [f"{check['id']}: could not execute command: {exc}"],
        }
    duration = time.monotonic() - start
    json_output_path = check.get("json_output_path")
    parsed, parse_error = parse_json_output(completed, json_output_path if isinstance(json_output_path, Path) else None)
    failures: list[str] = []
    if completed.returncode != 0:
        failures.append(f"{check['id']}: command exited {completed.returncode}")
    if parse_error is not None:
        failures.append(f"{check['id']}: could not parse JSON output: {parse_error}")
    if parsed is not None:
        failures.extend(validate_report(str(check["id"]), parsed))

    result: dict[str, Any] = {
        "id": check["id"],
        "description": check["description"],
        "command": command_string(command),
        "returncode": completed.returncode,
        "duration_seconds": round(duration, 3),
        "status": "pass" if not failures else "fail",
        "failures": failures,
    }
    if parsed is not None:
        result["summary"] = summarize_report(str(check["id"]), parsed)
    if failures:
        if completed.stderr:
            result["stderr_tail"] = completed.stderr[-2000:]
        if completed.stdout:
            result["stdout_tail"] = completed.stdout[-2000:]
    return result


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    checks = standard_checks(args.python)
    with tempfile.TemporaryDirectory(prefix="artifact-claim-checks-") as tmp:
        checks.append(public_replay_check(args.python, Path(tmp) / "recomputed_public_metrics.json"))
        results: list[dict[str, Any]] = []
        for check in checks:
            result = run_one(check, root=root)
            results.append(result)
            if args.fail_fast and result["status"] != "pass":
                break
    failures = [failure for result in results for failure in result.get("failures", [])]
    return {
        "status": "pass" if not failures else "fail",
        "root": str(root),
        "n_checks": len(results),
        "n_passed": sum(1 for result in results if result["status"] == "pass"),
        "n_failed": sum(1 for result in results if result["status"] != "pass"),
        "failures": failures,
        "checks": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run all release-safe paper-claim artifact checks.")
    parser.add_argument("--root", default=ROOT, help="Repository root.")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter used for child checks.")
    parser.add_argument("--out", default=None, help="Optional path for the JSON report.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failing check.")
    args = parser.parse_args()

    report = build_report(args)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
