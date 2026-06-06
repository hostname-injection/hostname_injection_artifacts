import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_makefile_uses_configurable_python_interpreter():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert "PYTHON ?= python3" in makefile
    assert "$(PYTHON) -m pytest" in makefile
    assert "$(PYTHON) scripts/check_artifact_readiness.py" in makefile
    assert "\n\tpython " not in makefile
    assert "\n\tpytest" not in makefile


def test_pyproject_exports_reviewer_console_scripts_and_packages():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject["project"]["scripts"]
    packages = set(pyproject["tool"]["setuptools"]["packages"])

    assert pyproject["project"]["requires-python"] == ">=3.11"
    assert "Programming Language :: Python :: 3.11" in pyproject["project"]["classifiers"]
    assert "license" not in pyproject["project"]
    assert all("License ::" not in classifier for classifier in pyproject["project"]["classifiers"])
    assert "optional-dependencies" not in pyproject["project"]
    assert {"ccd", "ccd-diagnose", "ccd-explain", "ccd-score"}.issubset(scripts)
    assert packages == {"ccd"}


def test_reviewer_scripts_are_canonical_entry_points_only():
    scripts = {path.name for path in (ROOT / "scripts").glob("*.py")}

    assert scripts == {
        "benchmark_artifact_latency.py",
        "check_artifact_readiness.py",
        "export_hib_release_pipeline_inputs.py",
        "run_artifact_smoke.py",
        "train_benchmark_caho.py",
    }


def test_requirements_install_package_and_pytest_without_extras():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")

    assert "-e ." in requirements
    assert "pytest>=7.4" in requirements
    assert ".[" not in requirements


def test_reviewer_repo_has_no_standalone_license_file():
    assert not (ROOT / "LICENSE").exists()


def test_reviewer_repo_omits_redundant_and_private_example_docs():
    assert not (ROOT / "examples" / "README.md").exists()
    assert not (ROOT / "deidentification_release" / "README.md").exists()
    assert not (ROOT / "deidentification_release" / "configs" / "anonymization_policy.private.example.yaml").exists()


def test_reviewer_repo_keeps_only_required_markdown_files():
    markdown_files = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*.md")
        if ".git" not in path.parts and ".pytest_cache" not in path.parts
    }

    assert markdown_files == {
        "README.md",
        "deidentification_release/data/audits/release_data_card.md",
    }
