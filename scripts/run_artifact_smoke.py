#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEID_SCRIPTS = ROOT / "deidentification_release" / "scripts"
sys.path.insert(0, str(DEID_SCRIPTS))

from validate_public_bundle import extract_validated_bundle  # noqa: E402


REQUIRED_RUNTIME_MODULES = {
    "sentence_transformers": "sentence-transformers",
    "torch": "pytorch",
    "grad_cache": "GradCache",
}


def check_runtime_deps() -> None:
    missing: list[str] = []
    for module_name, package_name in REQUIRED_RUNTIME_MODULES.items():
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            missing.append(f"{module_name} ({package_name}): {exc}")
    if not missing:
        return

    details = "\n".join(f"- {item}" for item in missing)
    raise SystemExit(
        "Artifact smoke requires the runtime dependencies from environment.yml. "
        "Create and activate the ccd conda environment with scripts/install_conda.sh "
        "before running this check.\n"
        "Missing or unusable modules:\n"
        f"{details}"
    )


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def deid_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(DEID_SCRIPTS) if not existing else f"{DEID_SCRIPTS}{os.pathsep}{existing}"
    return env


def assert_score_file(path: Path, *, grouped: bool = False) -> None:
    rows = path.read_text(encoding="utf-8").strip().splitlines()
    expected_header = "hostname,calibration_group,threshold,score,prediction" if grouped else "hostname,score,prediction"
    if len(rows) < 2 or rows[0] != expected_header:
        raise RuntimeError(f"unexpected score output in {path}")
    predictions = [row.rsplit(",", 1)[-1] for row in rows[1:]]
    if not set(predictions).issubset({"0", "1"}):
        raise RuntimeError(f"unexpected predictions in {path}: {predictions}")


def validate_extracted_public_bundle(bundle: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="hib-public-bundle-") as tmp:
        tmp_path = Path(tmp)
        extract_validated_bundle(bundle, tmp_path)

        extracted = tmp_path / "deidentification_release"
        run(
            [
                sys.executable,
                str(extracted / "scripts" / "validate_release_gate.py"),
                "--public-release",
                str(extracted / "data" / "release" / "hib_release.jsonl"),
                "--audit-dir",
                str(extracted / "data" / "audits"),
                "--bundle",
                str(bundle),
                "--count-rows",
            ]
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the IEEE S&P artifact smoke path for CCD and the HIB de-identification bundle."
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip pytest. The CCD and release-gate smoke checks still run.",
    )
    args = parser.parse_args()

    check_runtime_deps()

    run([sys.executable, "-m", "py_compile", *[str(p) for p in (ROOT / "ccd").glob("*.py")]])
    run([sys.executable, "-m", "py_compile", *[str(p) for p in (ROOT / "scripts").glob("*.py")]])
    run([sys.executable, "-m", "py_compile", *[str(p) for p in DEID_SCRIPTS.glob("*.py")]])

    if not args.skip_tests:
        run([sys.executable, "-m", "pytest", "-q"])

    run(
        [
            sys.executable,
            "deidentification_release/scripts/validate_public_bundle.py",
            "--bundle",
            "deidentification_release/data/release/hib_release_public_bundle.tar.gz",
        ],
        env=deid_env(),
    )
    run(
        [
            sys.executable,
            "deidentification_release/scripts/validate_release_gate.py",
            "--public-release",
            "deidentification_release/data/release/hib_release.jsonl",
            "--audit-dir",
            "deidentification_release/data/audits",
            "--bundle",
            "deidentification_release/data/release/hib_release_public_bundle.tar.gz",
            "--count-rows",
        ],
        env=deid_env(),
    )
    validate_extracted_public_bundle(ROOT / "deidentification_release" / "data" / "release" / "hib_release_public_bundle.tar.gz")

    with tempfile.TemporaryDirectory(prefix="hib-replay-smoke-") as tmp:
        metrics_path = Path(tmp) / "recomputed_public_metrics.json"
        run(
            [
                sys.executable,
                "deidentification_release/scripts/recompute_metrics.py",
                "--public-release",
                "deidentification_release/data/release/hib_release.jsonl",
                "--alpha",
                "1e-4",
                "--out",
                str(metrics_path),
            ],
            env=deid_env(),
        )
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("n_rows") != 150 or "fixed_fpr_replay" not in metrics or "ccd_output_counts" not in metrics:
            raise RuntimeError("unexpected recomputed public metrics output")
        fixed_fpr = metrics["fixed_fpr_replay"]
        calibration = metrics.get("calibration", {})
        label_accounting = metrics.get("label_accounting", {})
        if fixed_fpr.get("status") != "available" or fixed_fpr.get("threshold_source") != "recomputed_from_public_grouped_calibration_scores":
            raise RuntimeError("expected public grouped fixed-FPR replay to be available")
        if calibration.get("scored_benign_calibration_rows", 0) < 1 or label_accounting.get("metric_positive_rows", 0) < 1:
            raise RuntimeError("expected scored benign calibration rows and metric positives in public sample")
        if calibration.get("n_calibration_groups", 0) < 2:
            raise RuntimeError("expected at least two public calibration groups in public sample")
        if any(fixed_fpr.get(key, 0) < 1 for key in ("tp", "fp", "tn", "fn")):
            raise RuntimeError("expected public fixed-FPR sample to exercise TP/FP/TN/FN accounting")

    with tempfile.TemporaryDirectory(prefix="ccd-artifact-smoke-") as tmp:
        tmp_path = Path(tmp)
        caho_encoder = tmp_path / "caho_encoder"
        model = tmp_path / "ccd_smoke_model.npz"
        calibrated_model = tmp_path / "ccd_smoke_model.calibrated.npz"
        refreshed_model = tmp_path / "ccd_smoke_model.refreshed.npz"
        calibration = tmp_path / "calibration.json"
        refresh_report = tmp_path / "refresh.json"
        scores = tmp_path / "scores.csv"
        explanations = tmp_path / "explanations.json"
        certificates = tmp_path / "certificates.json"
        embeddings = tmp_path / "embeddings.npz"
        calibration_groups = tmp_path / "benign_calibration_groups.txt"
        query_groups = tmp_path / "query_groups.txt"
        calibration_groups.write_text(
            "\n".join(["tenant-a", "tenant-a", "tenant-b", "tenant-b", "tenant-a", "tenant-b", "tenant-a", "tenant-b"]) + "\n",
            encoding="utf-8",
        )
        query_groups.write_text(
            "\n".join(["tenant-a", "tenant-b", "tenant-a", "tenant-b", "tenant-a"]) + "\n",
            encoding="utf-8",
        )

        run(
            [
                sys.executable,
                "-m",
                "ccd.cli",
                "train-caho",
                "--benign",
                "examples/benign.txt",
                "--malicious",
                "examples/malicious.csv",
                "--out",
                str(caho_encoder),
                "--batch-size",
                "8",
                "--device",
                "cpu",
            ]
        )
        run(
            [
                sys.executable,
                "-m",
                "ccd.cli",
                "train-priors",
                "--benign",
                "examples/benign.txt",
                "--malicious",
                "examples/malicious.csv",
                "--config",
                "examples/ccd_smoke_config.json",
                "--encoder",
                str(caho_encoder),
                "--output",
                str(model),
                "--batch-size",
                "8",
            ]
        )
        run(
            [
                sys.executable,
                "-m",
                "ccd.cli",
                "calibrate",
                "--model",
                str(model),
                "--benign",
                "examples/benign_calibration.txt",
                "--groups",
                str(calibration_groups),
                "--output",
                str(calibration),
                "--save-model",
                str(calibrated_model),
                "--batch-size",
                "8",
            ]
        )
        run(
            [
                sys.executable,
                "-m",
                "ccd.cli",
                "refresh-benign",
                "--model",
                str(calibrated_model),
                "--benign",
                "examples/benign_calibration.txt",
                "--groups",
                str(calibration_groups),
                "--output",
                str(refreshed_model),
                "--report",
                str(refresh_report),
                "--batch-size",
                "8",
            ]
        )
        parsed_refresh = json.loads(refresh_report.read_text(encoding="utf-8"))
        if parsed_refresh.get("threshold_source") != "grouped_benign_refresh_scores":
            raise RuntimeError("expected grouped benign refresh threshold source")
        if parsed_refresh.get("n_calibration_groups", 0) < 2:
            raise RuntimeError("expected grouped thresholds after benign refresh")
        if parsed_refresh.get("refresh_scope", {}).get("grouped_thresholds_updated") is not True:
            raise RuntimeError("expected grouped threshold refresh scope")
        run(
            [
                sys.executable,
                "-m",
                "ccd.cli",
                "score",
                "--model",
                str(refreshed_model),
                "--input",
                "examples/queries.txt",
                "--output",
                str(scores),
                "--groups",
                str(query_groups),
                "--require-group-thresholds",
                "--batch-size",
                "8",
            ]
        )
        assert_score_file(scores, grouped=True)

        run(
            [
                sys.executable,
                "-m",
                "ccd.explain",
                "--model",
                str(refreshed_model),
                "--input",
                "examples/queries.txt",
                "--output",
                str(explanations),
                "--groups",
                str(query_groups),
                "--require-group-thresholds",
                "--top-k",
                "3",
                "--batch-size",
                "8",
            ]
        )
        parsed_explanations = json.loads(explanations.read_text(encoding="utf-8"))
        if parsed_explanations.get("count") != 5 or len(parsed_explanations.get("explanations", [])) != 5:
            raise RuntimeError("expected five explanation rows")
        if parsed_explanations.get("grouped_thresholds_used") is not True:
            raise RuntimeError("expected grouped thresholds in explanation smoke output")
        if parsed_explanations.get("grouped_thresholds_source") != "model_bundle_grouped_thresholds":
            raise RuntimeError("expected grouped threshold source in explanation smoke output")
        if parsed_explanations.get("decision_rule") != "score > threshold":
            raise RuntimeError("expected decision rule in explanation smoke output")
        if not isinstance(parsed_explanations.get("score_path"), dict):
            raise RuntimeError("expected score path in explanation smoke output")
        if any("calibration_group" not in row for row in parsed_explanations.get("explanations", [])):
            raise RuntimeError("expected calibration_group on each explanation row")
        if any("threshold_source" not in row for row in parsed_explanations.get("explanations", [])):
            raise RuntimeError("expected threshold_source on each explanation row")

        run(
            [
                sys.executable,
                "-m",
                "ccd.cli",
                "certify",
                "--model",
                str(refreshed_model),
                "--input",
                "examples/queries.txt",
                "--output",
                str(certificates),
                "--radius",
                "0",
                "--groups",
                str(query_groups),
                "--require-group-thresholds",
                "--batch-size",
                "8",
            ]
        )
        parsed_certificates = json.loads(certificates.read_text(encoding="utf-8"))
        if parsed_certificates.get("count") != 5 or len(parsed_certificates.get("certificates", [])) != 5:
            raise RuntimeError("expected five certificate rows")
        if parsed_certificates.get("grouped_thresholds_used") is not True:
            raise RuntimeError("expected grouped thresholds in certificate smoke output")
        if any("calibration_group" not in row for row in parsed_certificates.get("certificates", [])):
            raise RuntimeError("expected calibration_group on each certificate row")

        run(
            [
                sys.executable,
                "-m",
                "ccd.cli",
                "eval-caho",
                "--model",
                str(caho_encoder),
                "--input",
                "examples/queries.txt",
                "--output",
                str(embeddings),
                "--batch-size",
                "8",
                "--normalize",
                "--embed-normalize",
            ]
        )

    print("artifact smoke status: pass", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
