import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_smoke_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_artifact_smoke.py"
    spec = importlib.util.spec_from_file_location("_test_run_artifact_smoke", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_artifact_smoke_preflight_reports_missing_runtime_dependency(monkeypatch):
    module = _load_smoke_module()

    def fake_import(name):
        if name == "sentence_transformers":
            raise ModuleNotFoundError("No module named 'sentence_transformers'")
        return SimpleNamespace()

    monkeypatch.setattr(module.importlib, "import_module", fake_import)

    with pytest.raises(SystemExit) as excinfo:
        module.check_runtime_deps()

    message = str(excinfo.value)
    assert "environment.yml" in message
    assert "scripts/install_conda.sh" in message
    assert "sentence_transformers" in message


def test_artifact_smoke_preflight_accepts_required_runtime_dependencies(monkeypatch):
    module = _load_smoke_module()
    imported = []

    def fake_import(name):
        imported.append(name)
        return SimpleNamespace()

    monkeypatch.setattr(module.importlib, "import_module", fake_import)

    module.check_runtime_deps()

    assert set(imported) == set(module.REQUIRED_RUNTIME_MODULES)
