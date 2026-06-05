import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.calibrate as calibrate_script
import scripts.train_caho_corpus as train_caho_corpus


ROOT = Path(__file__).resolve().parents[1]


def test_makefile_uses_configurable_python_interpreter():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert "PYTHON ?= python3" in makefile
    assert "$(PYTHON) -m pytest" in makefile
    assert "\n\tpython " not in makefile
    assert "\n\tpytest" not in makefile


def test_train_caho_corpus_wrapper_uses_current_interpreter():
    args = SimpleNamespace(
        benign_dir="benign",
        malicious_jsonl_dir="jsonl",
        malicious_txt_dir="txt",
        jsonl_key="hostname",
        csv_hostname_col="Hostname",
        min_length=5,
        no_dedup=False,
        malicious_family="corpus",
        model="sentence-transformers/all-MiniLM-L6-v2",
        out="out",
        epochs=20,
        batch_size=None,
        lr=1e-4,
        weight_decay=1e-2,
        temperature=0.07,
        loss="supcon",
        augmenter="edit",
        weighted_num_augs=2,
        weighted_max_attempts=3,
        weighted_no_retry=False,
        max_grad_norm=1.0,
        scheduler="cosine",
        min_lr=1e-5,
        grad_cache=False,
        grad_cache_chunk_size=8192,
        contrastive_loss="fixed",
        contrastive_max_scale=100.0,
        contrastive_min_scale=1.0,
        optimize_contrastive_scale=False,
        num_workers=0,
        empty_cache=False,
        device="auto",
        resume=False,
        save_best=False,
        no_save_final=False,
        no_normalize=False,
        seed=13,
    )

    cmd = train_caho_corpus._build_command(args)

    assert cmd[:3] == [sys.executable, "-m", "ccd.cli"]
    assert "train-caho-corpus" in cmd


def test_standalone_calibrate_script_requires_saved_model_bundle():
    parser = calibrate_script.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--model",
                "ccd_model.npz",
                "--benign",
                "benign.txt",
                "--output",
                "calibration.json",
            ]
        )


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
