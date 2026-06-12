from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_artifact_archive as archive  # noqa: E402


def test_archive_excludes_intermediate_training_checkpoints(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("artifact\n", encoding="utf-8")
    (tmp_path / "ARTIFACT_MANIFEST.json").write_text(
        json.dumps({"required_files": ["README.md"]}),
        encoding="utf-8",
    )
    checkpoint_dir = tmp_path / "checkpoints" / "step-00005000"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "model.safetensors").write_text("intermediate", encoding="utf-8")

    result = archive.run(
        tmp_path,
        tmp_path / "ARTIFACT_MANIFEST.json",
        tmp_path / "dist" / "artifact.tar.gz",
        prefix="artifact",
        dry_run=True,
    )

    assert "checkpoints" in result["excluded_dirs"]
    assert "checkpoints" not in result["top_level_bytes"]
    assert "model.safetensors" not in result["top_level_bytes"]


def test_archive_excludes_internal_codex_brief(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("artifact\n", encoding="utf-8")
    (tmp_path / "ARTIFACT_MANIFEST.json").write_text(
        json.dumps({"required_files": ["README.md"]}),
        encoding="utf-8",
    )
    (tmp_path / "CODEX_DEIDENTIFICATION_IMPLEMENTATION_BRIEF.md").write_text(
        "internal implementation notes\n",
        encoding="utf-8",
    )

    result = archive.run(
        tmp_path,
        tmp_path / "ARTIFACT_MANIFEST.json",
        tmp_path / "dist" / "artifact.tar.gz",
        prefix="artifact",
        dry_run=True,
    )

    assert "CODEX_DEIDENTIFICATION_IMPLEMENTATION_BRIEF.md" in result["excluded_files"]
    assert "CODEX_DEIDENTIFICATION_IMPLEMENTATION_BRIEF.md" not in result["top_level_bytes"]
