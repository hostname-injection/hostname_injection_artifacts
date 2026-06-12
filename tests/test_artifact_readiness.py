from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import audit_artifact_readiness as readiness  # noqa: E402
from audit_artifact_readiness import check_claim_script_targets  # noqa: E402


def test_artifact_manifest_has_claim_scripts_and_required_files() -> None:
    manifest = json.loads((ROOT / "ARTIFACT_MANIFEST.json").read_text(encoding="utf-8"))

    assert manifest["paper_title"] == "When Hostnames Become Code: Detecting Persisted Hostname Injection at Production Scale"
    assert {"Available", "Functional", "Reproduced"}.issubset(set(manifest["target_badges"]))
    requirements = manifest["ieee_sp_artifact_requirements"]
    criteria = requirements["badge_criteria"]
    assert criteria["Available"]["permanent_public_retrieval_required"] is True
    assert criteria["Available"]["doi_required"] is True
    assert "external" in criteria["Available"]["current_status"]
    assert criteria["Functional"]["documentation"]["supported"] is True
    assert criteria["Functional"]["completeness"]["supported"] is True
    assert criteria["Functional"]["exercisability"]["supported"] is True
    assert criteria["Functional"]["other_machine_portability"]["supported"] is True
    assert criteria["Reproduced"]["main_results_supported"] is True
    assert criteria["Reproduced"]["allowed_tolerance_documented"] is True
    assert criteria["Reproduced"]["scaled_down_for_lengthy_experiments"] is True
    assert criteria["Reproduced"]["external_full_data_required"] is True
    assert requirements["public_research_infrastructure"]["supported"] is True
    assert requirements["runtime"]["aec_limit_hours"] == 24
    assert requirements["runtime"]["scaled_down_experiments_justified"] is True
    assert requirements["packaging"]["source_package"] is True
    assert requirements["packaging"]["container_required"] is False
    assert requirements["tracking"]["web_tracking_embedded"] is False
    assert (ROOT / manifest["badge_readiness"]).exists()
    assert manifest["claims"]
    claim_text = " ".join(claim["claim"] for claim in manifest["claims"])
    assert "source-reachability" in claim_text
    assert "Public-scope taxonomy" in claim_text
    assert "latency smoke" in claim_text
    assert "method contracts" in claim_text
    assert "Named paper claims" in claim_text
    assert "headline numeric claims" in claim_text
    assert "one evaluator command" in claim_text
    assert "Live-overlap" in claim_text
    assert "Aggregate paper metric tables" in claim_text
    assert "HIB dataset-profile" in claim_text
    assert "Evaluation units" in claim_text
    assert "Decision-stability" in claim_text
    assert "Production latency" in claim_text
    assert "sink-evidence" in claim_text
    assert manifest["public_release_bundle"]["row_count"] == 150
    assert manifest["public_release_bundle"]["metrics_expectations"]["fixed_fpr_status"] == "available"
    assert manifest["public_release_bundle"]["metrics_expectations"]["threshold_source"] == "recomputed_from_public_grouped_calibration_scores"
    assert manifest["public_release_bundle"]["metrics_expectations"]["min_calibration_groups"] == 2
    assert manifest["public_release_bundle"]["metrics_expectations"]["min_fixed_fpr_fp"] == 1
    assert manifest["public_release_bundle"]["metrics_expectations"]["min_fixed_fpr_fn"] == 1
    assert all(claim.get("claim") and claim.get("script") and claim.get("expected") for claim in manifest["claims"])
    assert all((ROOT / rel).exists() for rel in manifest["required_files"])
    assert manifest["tracking_scan_paths"]
    assert (ROOT / manifest["metadata_template"]).exists()
    assert (ROOT / "source_reachability" / "paper_source_reachability_counts.json").exists()
    assert (ROOT / "method_contracts" / "paper_method_contracts.json").exists()
    assert (ROOT / "paper_claim_coverage" / "paper_claim_coverage.json").exists()
    assert (ROOT / "paper_headline_claims" / "paper_headline_claims.json").exists()
    assert (ROOT / "scripts" / "run_artifact_claim_checks.py").exists()
    assert (ROOT / "hib_profile" / "paper_hib_profile_counts.json").exists()
    assert (ROOT / "evaluation_accounting" / "paper_evaluation_accounting.json").exists()
    assert (ROOT / "public_scope" / "paper_public_scope_counts.json").exists()
    assert (ROOT / "live_overlap" / "paper_live_overlap_counts.json").exists()
    assert (ROOT / "paper_metric_tables" / "paper_metric_tables.json").exists()
    assert (ROOT / "stability_scope" / "paper_stability_scope_counts.json").exists()
    assert (ROOT / "production_latency" / "paper_production_latency_counts.json").exists()
    assert (ROOT / "sink_evidence" / "paper_sink_evidence_counts.json").exists()


def test_metadata_template_is_parseable_and_points_to_artifact_docs() -> None:
    metadata = tomllib.loads((ROOT / "metadata.template.toml").read_text(encoding="utf-8"))

    assert metadata["badge"] == "r"
    assert metadata["cd"] == "b"
    assert "When Hostnames Become Code" in metadata["citation"]
    assert "scripts/build_artifact_archive.py" in metadata["script6"]
    assert "scripts/recompute_source_reachability_metrics.py" in metadata["script7"]
    assert "scripts/recompute_public_scope_metrics.py" in metadata["script8"]
    assert "scripts/benchmark_artifact_latency.py" in metadata["script9"]
    assert "scripts/recompute_live_overlap_metrics.py" in metadata["script10"]
    assert "scripts/recompute_paper_metric_tables.py" in metadata["script11"]
    assert "scripts/recompute_hib_profile_metrics.py" in metadata["script12"]
    assert "scripts/recompute_stability_scope_metrics.py" in metadata["script13"]
    assert "scripts/recompute_production_latency_metrics.py" in metadata["script14"]
    assert "scripts/recompute_sink_evidence_metrics.py" in metadata["script15"]
    assert "scripts/recompute_evaluation_accounting.py" in metadata["script16"]
    assert "scripts/recompute_method_contracts.py" in metadata["script17"]
    assert "scripts/recompute_paper_claim_coverage.py" in metadata["script18"]
    assert "tests/test_certify.py" in metadata["script19"]
    assert "tests/test_benchmark_training.py" in metadata["script20"]
    assert "scripts/recompute_paper_headline_claims.py" in metadata["script21"]
    assert "scripts/run_artifact_claim_checks.py" in metadata["script22"]
    assert "ARTIFACT_PROVENANCE_AND_ETHICS.md" in metadata["provenance"]
    assert "ARTIFACT_PROVENANCE_AND_ETHICS.md" in metadata["ethics"]
    assert "scripts/install_conda.sh" in metadata["install_script"]


def test_dependency_files_cover_artifact_smoke_and_claim_map() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = set(pyproject["project"]["dependencies"])
    extras = pyproject["project"]["optional-dependencies"]
    artifact_deps = set(extras["artifact"])
    baseline_deps = set(extras["baselines"])

    assert any(dep.startswith("numpy") for dep in deps)
    assert any(dep.startswith("scipy") for dep in deps)
    assert any(dep.startswith("torch") for dep in deps)
    assert any(dep.startswith("sentence-transformers") for dep in deps)
    assert any(dep.startswith("idna") for dep in deps)
    assert any(dep.startswith("pytest") for dep in artifact_deps)
    assert any(dep.startswith("sentencepiece") for dep in artifact_deps)
    assert any(dep.startswith("scikit-learn") for dep in artifact_deps)
    assert any(dep.startswith("pandas") for dep in baseline_deps)
    assert any(dep.startswith("transformers") for dep in baseline_deps)
    assert any(dep.startswith("xgboost") for dep in baseline_deps)

    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "-e .[artifact]" in requirements

    environment = (ROOT / "environment.yml").read_text(encoding="utf-8")
    for package in (
        "python=3.11",
        "numpy",
        "scipy",
        "pytorch",
        "sentence-transformers",
        "idna",
        "pytest",
        "sentencepiece",
        "scikit-learn",
    ):
        assert package in environment

    install_script = (ROOT / "scripts/install_conda.sh").read_text(encoding="utf-8")
    assert 'INSTALL_GRADCACHE="${INSTALL_GRADCACHE:-0}"' in install_script
    assert "scikit-learn" in install_script


def test_artifact_readiness_audit_quick_path_passes() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/audit_artifact_readiness.py", "--skip-gates"],
        cwd=ROOT,
        capture_output=True,
        check=True,
        text=True,
    )

    result = json.loads(completed.stdout)
    assert result["status"] == "pass"
    assert result["checks"]["ieee_sp_requirements"] == "pass"
    assert result["checks"]["tracking_scan"] == "pass"


def test_tracking_scan_rejects_web_analytics(tmp_path, monkeypatch) -> None:
    pattern = "post" + "hog.init"
    tracked = tmp_path / "page.html"
    tracked.write_text(f"<script>{pattern}({{}})</script>", encoding="utf-8")
    monkeypatch.setattr(readiness, "ROOT", tmp_path)

    failures = readiness.check_tracking_scan({"tracking_scan_paths": ["page.html"]})

    assert failures == [f"web tracking pattern found in page.html: {pattern}"]


def test_strict_final_audit_reports_publication_blockers() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/audit_artifact_readiness.py", "--skip-gates", "--strict-final"],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )

    result = json.loads(completed.stdout)
    assert completed.returncode == 1
    assert result["checks"]["final_publication_readiness"] == "fail"
    assert any("external completion item" in failure for failure in result["failures"])


def test_artifact_readiness_audit_rejects_missing_claim_paths() -> None:
    failures = check_claim_script_targets(
        {
            "claims": [
                {
                    "claim": "missing path example",
                    "script": "python does/not/exist.py",
                    "expected": "failure",
                }
            ]
        }
    )

    assert "claim 1 references missing path: does/not/exist.py" in failures


def test_ieee_sp_requirements_reject_missing_public_infrastructure() -> None:
    manifest = json.loads((ROOT / "ARTIFACT_MANIFEST.json").read_text(encoding="utf-8"))
    manifest["ieee_sp_artifact_requirements"]["public_research_infrastructure"]["supported"] = False

    failures = readiness.check_ieee_sp_requirements(manifest)

    assert "public research infrastructure support should be marked true" in failures


def test_ieee_sp_requirements_reject_missing_functional_badge_aspect() -> None:
    manifest = json.loads((ROOT / "ARTIFACT_MANIFEST.json").read_text(encoding="utf-8"))
    del manifest["ieee_sp_artifact_requirements"]["badge_criteria"]["Functional"]["exercisability"]

    failures = readiness.check_ieee_sp_requirements(manifest)

    assert "Functional criterion missing aspect: exercisability" in failures


def test_ieee_sp_requirements_reject_missing_badge_evidence_path() -> None:
    manifest = json.loads((ROOT / "ARTIFACT_MANIFEST.json").read_text(encoding="utf-8"))
    manifest["ieee_sp_artifact_requirements"]["badge_criteria"]["Reproduced"]["evidence"] = ["does/not/exist.json"]

    failures = readiness.check_ieee_sp_requirements(manifest)

    assert "Reproduced criterion evidence path missing or does not exist: does/not/exist.json" in failures


def test_metadata_template_rejects_too_few_claim_blocks(tmp_path: Path, monkeypatch) -> None:
    metadata = (ROOT / "metadata.template.toml").read_text(encoding="utf-8")
    trimmed = metadata.split("claim20 =", 1)[0]
    metadata_path = tmp_path / "metadata.template.toml"
    metadata_path.write_text(trimmed, encoding="utf-8")
    monkeypatch.setattr(readiness, "ROOT", tmp_path)

    failures = readiness.check_metadata_template(
        {
            "metadata_template": "metadata.template.toml",
            "claims": [{} for _ in range(20)],
        }
    )

    assert any("fewer than manifest claims" in failure for failure in failures)
