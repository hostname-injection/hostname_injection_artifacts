from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_root_readme_uses_current_deidentification_cli_flags():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "--public-release deidentification_release/data/release/hib_release.jsonl" in readme
    assert "--audit-dir deidentification_release/data/audits" in readme
    assert "--release deidentification_release/data/release/hib_release.jsonl" not in readme


def test_root_readme_certification_example_includes_radius():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    start = readme.index("ccd certify \\")
    end = readme.index("```", start)
    snippet = readme[start:end]

    assert "--radius 1" in snippet


def test_root_readme_states_runtime_requirements_and_scaled_smoke_scope():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "## Runtime Requirements" in readme
    assert "Python 3.11" in readme
    assert "94 GB of VRAM" in readme
    assert "small `examples/` smoke path" in readme
    assert "different results" in readme
    assert "The reviewer-facing CAHO training pipeline uses GradCache" in readme
    assert "--loss contrastive --augmenter weighted --epochs 20" in readme


def test_root_readme_maps_reviewer_evidence_to_method_surfaces():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(readme.split())

    assert "## Reviewer Evidence" in readme
    assert "make test" in readme
    assert "make artifact-readiness" in readme
    assert "python scripts/run_artifact_smoke.py --skip-tests" in readme
    for phrase in [
        "CAHO augmentation",
        "GradCache hooks",
        "split-conformal calibration",
        "benign-only `P_B` refresh",
        "finite-edit certification",
        "CLI CAHO-first gates",
        "de-identification release gates",
    ]:
        assert phrase in normalized


def test_root_readme_requires_embedded_calibrated_threshold_for_downstream_commands():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(readme.split())

    assert "ccd calibrate --save-model" in normalized
    assert "`--save-model` is required" in normalized
    assert "The input model must already contain an embedded calibrated threshold" in normalized
    assert "do not accept ad hoc threshold or calibration-file overrides" in normalized


def test_root_readme_shows_public_release_pipeline_export():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(readme.split())

    assert "scripts/export_hib_release_pipeline_inputs.py" in readme
    assert "--public-release deidentification_release/data/release/hib_release.jsonl" in readme
    assert "preserves released row multiplicity" in normalized


def test_root_readme_links_reviewer_slice():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert (
        "https://drive.google.com/drive/folders/1KeKZyIXIqZvEJ4tZAWxE9h4gPoinZCWt?usp=drive_link"
        in readme
    )
