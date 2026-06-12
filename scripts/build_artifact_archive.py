#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXCLUDED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "ccd.egg-info",
    "checkpoints",
    "dist",
}
DEFAULT_EXCLUDED_FILES = {
    ".DS_Store",
    "CODEX_DEIDENTIFICATION_IMPLEMENTATION_BRIEF.md",
    "uv.lock",
}
DEFAULT_EXCLUDED_SUFFIXES = {
    ".pyc",
    ".pyo",
}


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def should_exclude(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    if any(part in DEFAULT_EXCLUDED_DIRS for part in rel.parts):
        return True
    if path.name in DEFAULT_EXCLUDED_FILES:
        return True
    return path.suffix in DEFAULT_EXCLUDED_SUFFIXES


def iter_artifact_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        if should_exclude(path, root):
            continue
        if path.is_symlink():
            raise ValueError(f"refusing to archive symlink: {path.relative_to(root)}")
        yield path


def check_required_paths(root: Path, manifest: dict[str, Any]) -> list[str]:
    missing = []
    for rel in manifest.get("required_files", []):
        if not (root / rel).exists():
            missing.append(str(rel))
    return missing


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_sha256_sidecar(path: Path, target: Path) -> None:
    path.write_text(f"{sha256(target)}  {target.name}\n", encoding="utf-8")


def build_archive(root: Path, output: Path, files: list[Path], *, prefix: str) -> dict[str, str]:
    output.parent.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    with tarfile.open(output, "w:gz") as tar:
        for path in files:
            rel = path.relative_to(root).as_posix()
            arcname = f"{prefix.rstrip('/')}/{rel}" if prefix else rel
            tar.add(path, arcname=arcname)
            hashes[rel] = sha256(path)
    write_sha256_sidecar(output.with_suffix(output.suffix + ".sha256"), output)
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps({"archive": output.name, "sha256": sha256(output), "prefix": prefix, "files": hashes}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return hashes


def summarize_files(root: Path, files: list[Path]) -> dict[str, Any]:
    total_bytes = 0
    by_top_level: dict[str, int] = {}
    for path in files:
        size = path.stat().st_size
        total_bytes += size
        rel = path.relative_to(root)
        key = rel.parts[0] if rel.parts else "."
        by_top_level[key] = by_top_level.get(key, 0) + size
    return {
        "n_files": len(files),
        "total_bytes": total_bytes,
        "top_level_bytes": dict(sorted(by_top_level.items())),
    }


def run(root: Path, manifest_path: Path, output: Path, *, prefix: str, dry_run: bool) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    missing = check_required_paths(root, manifest)
    if missing:
        raise FileNotFoundError(f"required artifact paths are missing: {missing}")
    files = list(iter_artifact_files(root))
    summary = summarize_files(root, files)
    result: dict[str, Any] = {
        "root": str(root),
        "manifest": str(manifest_path),
        "output": str(output),
        "prefix": prefix,
        "dry_run": dry_run,
        **summary,
        "excluded_dirs": sorted(DEFAULT_EXCLUDED_DIRS),
        "excluded_files": sorted(DEFAULT_EXCLUDED_FILES),
        "excluded_suffixes": sorted(DEFAULT_EXCLUDED_SUFFIXES),
        "status": "pass",
    }
    if dry_run:
        return result
    hashes = build_archive(root, output, files, prefix=prefix)
    result["archive_manifest"] = str(output.with_suffix(".manifest.json"))
    result["sha256_sidecar"] = str(output.with_suffix(output.suffix + ".sha256"))
    result["n_hashed_files"] = len(hashes)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a clean DOI-ready source artifact archive.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, default=ROOT / "ARTIFACT_MANIFEST.json")
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "ccd-hostname-command-injection-artifact.tar.gz")
    parser.add_argument("--prefix", default="ccd-hostname-command-injection-artifact")
    parser.add_argument("--dry-run", action="store_true", help="List archive summary without writing the tarball.")
    args = parser.parse_args()

    result = run(args.root.resolve(), args.manifest.resolve(), args.output.resolve(), prefix=args.prefix, dry_run=args.dry_run)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
