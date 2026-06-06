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


def test_readiness_gate_accepts_current_reviewer_artifact() -> None:
    module = _load_module()

    assert module.main() == 0
