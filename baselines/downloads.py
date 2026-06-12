from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass(frozen=True)
class RepoSpec:
    key: str
    url: str
    description: str


REPO_SOURCES: Dict[str, RepoSpec] = {
    "urlnet": RepoSpec(
        key="urlnet",
        url="https://github.com/Antimalweb/URLNet",
        description="URLNet: Learning a URL Representation with Deep Learning for Malicious URL Detection",
    ),
    "urlbert": RepoSpec(
        key="urlbert",
        url="https://github.com/Davidup1/URLBERT",
        description="URLBERT: Continuous Multi-Task Pre-training for Malicious URL Detection",
    ),
    "csi": RepoSpec(
        key="csi",
        url="https://github.com/alinlab/CSI",
        description="CSI: Novelty Detection via Contrastive Learning on Distributionally Shifted Instances",
    ),
    "deep-sad": RepoSpec(
        key="deep-sad",
        url="https://github.com/lukasruff/Deep-SAD-PyTorch",
        description="Deep SAD: Deep Semi-Supervised Anomaly Detection",
    ),
    "deep-svdd": RepoSpec(
        key="deep-svdd",
        url="https://github.com/lukasruff/Deep-SVDD-PyTorch",
        description="Deep SVDD / Deep One-Class Classification",
    ),
    "drocc": RepoSpec(
        key="drocc",
        url="https://github.com/microsoft/EdgeML",
        description="DROCC: Deep Robust One-Class Classification (EdgeML repo)",
    ),
}

REPO_CHECKS: Dict[str, List[str]] = {
    "urlnet": ["README.md"],
    "urlbert": ["README.md"],
    "csi": ["README.md"],
    "deep-sad": ["README.md"],
    "deep-svdd": ["README.md"],
    "drocc": ["EdgeML", "README.md"],
}


BASELINE_TO_REPO: Dict[str, str] = {
    "urlnet": "urlnet",
    "urlbert": "urlbert",
    "csi": "csi",
    "deep-sad": "deep-sad",
    "deep-svdd": "deep-svdd",
    "deep-one-class": "deep-svdd",
    "drocc": "drocc",
}


def repo_path(root: Path, key: str) -> Path:
    return root / key


def ensure_repo(key: str, root: Path, allow_downloads: bool) -> Path:
    spec = REPO_SOURCES.get(key)
    if spec is None:
        raise KeyError(f"Unknown repo key: {key}")
    path = repo_path(root, key)
    if path.exists():
        _verify_repo(key, path)
        return path
    if not allow_downloads:
        raise RuntimeError(
            f"Repo '{key}' not found at {path}. Pass --download-repos to fetch it."
        )
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", spec.url, str(path)],
        check=True,
    )
    _verify_repo(key, path)
    return path


def _verify_repo(key: str, path: Path) -> None:
    checks = REPO_CHECKS.get(key, [])
    missing = []
    for rel in checks:
        if not (path / rel).exists():
            missing.append(rel)
    if missing:
        raise RuntimeError(f"Repo '{key}' missing expected paths: {missing}")


def ensure_repos_for_baselines(
    baselines: Iterable[str],
    root: Path,
    allow_downloads: bool,
) -> Dict[str, Path]:
    resolved: Dict[str, Path] = {}
    for name in baselines:
        repo_key = BASELINE_TO_REPO.get(name)
        if not repo_key:
            continue
        resolved[repo_key] = ensure_repo(repo_key, root, allow_downloads)
    return resolved


def list_repos() -> List[RepoSpec]:
    return [REPO_SOURCES[k] for k in sorted(REPO_SOURCES.keys())]


def main() -> int:
    parser = argparse.ArgumentParser(description="Download baseline repos.")
    parser.add_argument("--root", type=Path, default=Path("baselines/downloads"))
    parser.add_argument("--all", action="store_true", help="Download all baseline repos.")
    parser.add_argument("--repos", type=str, default="", help="Comma-separated repo keys.")

    args = parser.parse_args()

    keys: List[str] = []
    if args.all:
        keys = list(REPO_SOURCES.keys())
    elif args.repos:
        keys = [k.strip() for k in args.repos.split(",") if k.strip()]

    if not keys:
        print("Available repos:")
        for spec in list_repos():
            print(f"{spec.key}: {spec.url}")
        return 0

    for key in keys:
        spec = REPO_SOURCES.get(key)
        if not spec:
            print(f"Unknown repo key: {key}")
            continue
        path = ensure_repo(key, args.root, allow_downloads=True)
        print(f"Downloaded {key} -> {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
