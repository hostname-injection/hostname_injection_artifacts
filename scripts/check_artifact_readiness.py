#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REVIEWER_SLICE_URL = "https://drive.google.com/drive/folders/1KeKZyIXIqZvEJ4tZAWxE9h4gPoinZCWt?usp=drive_link"

SKIP_DIRS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
}
SKIP_SUFFIXES = {
    ".gz",
    ".pyc",
    ".pyo",
}
MODEL_ARTIFACT_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
}


def review_term_fragments() -> tuple[str, ...]:
    return (
        "base" + "line",
        "base" + "lines",
        "comparison " + "model",
        "comparison " + "detector",
        "competing " + "model",
        "competing " + "detector",
        "traditional " + "detector",
        "detector" + "_output" + "s",
        "detector" + "_output",
        "W" + "AF",
        "Mod" + "Security",
        "Sn" + "ort",
        "Suri" + "cata",
        "--no-" + "dedup",
        "--max-" + "rows",
        "--max-" + "steps",
        "--" + "edits",
        "--skip-" + "encoder",
        "train-" + "user-" + "logins",
    )


def iter_repo_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        rel_parts = path.relative_to(ROOT).parts
        if any(part in SKIP_DIRS for part in rel_parts):
            continue
        if path.is_file():
            files.append(path)
    return sorted(files)


def text_files(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for path in paths:
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        try:
            path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        out.append(path)
    return out


def fail(message: str, failures: list[str]) -> None:
    failures.append(message)


def check_repo_shape(files: list[Path], failures: list[str]) -> None:
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        lower_name = path.name.lower()
        if "license" in lower_name:
            fail(f"license file is not part of the reviewer artifact: {rel}", failures)
        if path.suffix.lower() in {".pyc", ".pyo"}:
            fail(f"compiled Python artifact is checked in: {rel}", failures)
        if path.suffix.lower() in MODEL_ARTIFACT_SUFFIXES:
            fail(f"pretrained/model artifact is checked in: {rel}", failures)


def check_readme(failures: list[str]) -> None:
    readme = ROOT / "README.md"
    if not readme.exists():
        fail("README.md is missing", failures)
        return
    text = readme.read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    required = [
        REVIEWER_SLICE_URL,
        "does not include pretrained models",
        "GradCache is required",
        "data shift relative to the original evaluation set",
        "ccd calibrate --save-model",
        "There is no supported path for training, scoring, calibrating, refreshing, or certifying CCD without a trained CAHO checkpoint",
    ]
    for needle in required:
        if needle not in normalized:
            fail(f"README.md is missing required reviewer-facing text: {needle}", failures)


def check_review_terms(paths: list[Path], failures: list[str]) -> None:
    terms = tuple(term.lower() for term in review_term_fragments())
    for path in paths:
        try:
            rel = path.relative_to(ROOT).as_posix()
        except ValueError:
            rel = path.as_posix()
        text = path.read_text(encoding="utf-8").lower()
        for term in terms:
            if term in text:
                fail(f"reviewer artifact contains removed term {term!r} in {rel}", failures)


def run_release_validators(failures: list[str]) -> None:
    env = os.environ.copy()
    scripts_dir = ROOT / "deidentification_release" / "scripts"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(scripts_dir) + os.pathsep + env.get("PYTHONPATH", "")
    commands = [
        [
            sys.executable,
            "deidentification_release/scripts/validate_public_bundle.py",
            "--bundle",
            "deidentification_release/data/release/hib_release_public_bundle.tar.gz",
        ],
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
    ]
    for command in commands:
        completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        if completed.returncode != 0:
            fail(
                "release validator failed: "
                + " ".join(command)
                + "\n"
                + completed.stdout
                + completed.stderr,
                failures,
            )


def main() -> int:
    failures: list[str] = []
    files = iter_repo_files()
    check_repo_shape(files, failures)
    check_readme(failures)
    check_review_terms(text_files(files), failures)
    run_release_validators(failures)

    if failures:
        for item in failures:
            print(f"FAIL: {item}", file=sys.stderr)
        return 1
    print("artifact readiness status: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
