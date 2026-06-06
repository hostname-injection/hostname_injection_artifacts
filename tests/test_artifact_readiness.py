import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    script = ROOT / "scripts" / "check_artifact_readiness.py"
    spec = importlib.util.spec_from_file_location("_test_check_artifact_readiness", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_readiness_gate_scans_constructed_removed_terms(tmp_path: Path) -> None:
    module = _load_module()
    sample = tmp_path / "sample.txt"
    sample.write_text("contains " + "base" + "line" + " wording\n", encoding="utf-8")
    failures: list[str] = []

    module.check_review_terms([sample], failures)

    assert failures


def test_readiness_gate_enforces_reviewer_shape() -> None:
    module = _load_module()
    files = [
        module.ROOT / "README.md",
        module.ROOT / "deidentification_release/data/audits/release_data_card.md",
        module.ROOT / "scripts/benchmark_artifact_latency.py",
        module.ROOT / "scripts/check_artifact_readiness.py",
        module.ROOT / "scripts/export_hib_release_pipeline_inputs.py",
        module.ROOT / "scripts/run_artifact_smoke.py",
        module.ROOT / "scripts/train_benchmark_caho.py",
    ]
    failures: list[str] = []

    module.check_repo_shape(files, failures)

    assert failures == []

    module.check_repo_shape([*files, module.ROOT / "scripts/extra.py"], failures)

    assert any("script surface" in item for item in failures)


def test_readiness_gate_validates_packaging_metadata() -> None:
    module = _load_module()
    failures: list[str] = []

    module.check_packaging(failures)

    assert failures == []


def test_readiness_gate_accepts_current_reviewer_artifact() -> None:
    module = _load_module()

    assert module.main() == 0
