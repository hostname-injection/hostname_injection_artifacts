import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ccd.cli import build_parser


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


def test_artifact_smoke_command_sequence_is_cli_parseable(monkeypatch):
    module = _load_smoke_module()
    commands = []

    def fake_run(cmd, *, env=None):
        del env
        commands.append(cmd)
        if any(str(part).endswith("recompute_metrics.py") for part in cmd) and "--out" in cmd:
            out = Path(cmd[cmd.index("--out") + 1])
            out.write_text(
                json.dumps(
                    {
                        "n_rows": 150,
                        "fixed_fpr_replay": {
                            "status": "available",
                            "threshold_source": "recomputed_from_public_grouped_calibration_scores",
                            "tp": 1,
                            "fp": 1,
                            "tn": 1,
                            "fn": 1,
                        },
                        "calibration": {
                            "scored_benign_calibration_rows": 1,
                            "n_calibration_groups": 2,
                        },
                        "label_accounting": {"metric_positive_rows": 1},
                        "ccd_output_counts": {},
                    }
                ),
                encoding="utf-8",
            )
        elif "refresh-benign" in cmd:
            report = Path(cmd[cmd.index("--report") + 1])
            report.write_text(
                json.dumps(
                    {
                        "threshold_source": "grouped_benign_refresh_scores",
                        "n_calibration_groups": 2,
                        "refresh_scope": {"grouped_thresholds_updated": True},
                    }
                ),
                encoding="utf-8",
            )
        elif "score" in cmd and "--output" in cmd:
            output = Path(cmd[cmd.index("--output") + 1])
            output.write_text(
                "hostname,calibration_group,threshold,score,prediction\n"
                "example.invalid,tenant-a,0.500000,0.100000,0\n",
                encoding="utf-8",
            )
        elif "ccd.explain" in cmd:
            output = Path(cmd[cmd.index("--output") + 1])
            output.write_text(
                json.dumps(
                    {
                        "count": 5,
                        "grouped_thresholds_used": True,
                        "grouped_thresholds_source": "model_bundle_grouped_thresholds",
                        "decision_rule": "score > threshold",
                        "score_path": {},
                        "explanations": [
                            {"calibration_group": "tenant-a", "threshold_source": "model_bundle_grouped_thresholds"}
                            for _ in range(5)
                        ],
                    }
                ),
                encoding="utf-8",
            )
        elif "certify" in cmd:
            output = Path(cmd[cmd.index("--output") + 1])
            output.write_text(
                json.dumps(
                    {
                        "count": 5,
                        "grouped_thresholds_used": True,
                        "certificates": [{"calibration_group": "tenant-a"} for _ in range(5)],
                    }
                ),
                encoding="utf-8",
            )

    monkeypatch.setattr(module, "check_runtime_deps", lambda: None)
    monkeypatch.setattr(module, "run", fake_run)
    monkeypatch.setattr(module, "validate_extracted_public_bundle", lambda _bundle: None)
    monkeypatch.setattr(module.sys, "argv", ["run_artifact_smoke.py", "--skip-tests"])

    assert module.main() == 0

    parser = build_parser()
    for cmd in commands:
        if "-m" not in cmd:
            continue
        module_name = cmd[cmd.index("-m") + 1]
        if module_name == "ccd.cli":
            parser.parse_args(cmd[cmd.index("ccd.cli") + 1 :])
