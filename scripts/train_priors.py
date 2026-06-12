#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from ccd.config import CCDConfig
from ccd.cone import ConePartition
from ccd.encoder import CahoEncoder
from ccd.io import ModelBundle, save_model
from ccd.priors import build_benign_prior, build_malicious_priors
from ccd.preprocess import normalize_hostname


def read_lines(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(errors="ignore").splitlines() if line.strip()]


def read_malicious_csv(path: Path) -> Dict[str, List[str]]:
    # CSV with columns: hostname,family
    out: Dict[str, List[str]] = {}
    for line in path.read_text(errors="ignore").splitlines():
        if not line.strip() or line.lower().startswith("hostname"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        host, family = parts[0], parts[1]
        out.setdefault(family, []).append(host)
    return out


def apply_normalization(hosts: List[str]) -> List[str]:
    return [normalize_hostname(h) for h in hosts]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benign", required=True, type=Path)
    parser.add_argument("--malicious", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--encoder", type=str, default=None, help="Override encoder model path/name")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-normalize", action="store_true", help="Skip hostname normalization")
    args = parser.parse_args()

    config = CCDConfig()
    if args.config and args.config.exists():
        config_dict = json.loads(args.config.read_text())
        config = CCDConfig.from_dict(config_dict)

    if args.encoder:
        config.encoder.model_name = args.encoder

    encoder = CahoEncoder(config.encoder)
    benign = read_lines(args.benign)
    malicious = read_malicious_csv(args.malicious)

    if not args.no_normalize:
        benign = apply_normalization(benign)
        malicious = {fam: apply_normalization(hosts) for fam, hosts in malicious.items()}

    benign_emb = encoder.encode(benign, batch_size=args.batch_size)
    mal_emb = {fam: encoder.encode(hosts, batch_size=args.batch_size) for fam, hosts in malicious.items()}

    cones = ConePartition.build(config.cone)
    benign_prior = build_benign_prior(benign_emb, cones, config.prior)
    mal_priors = build_malicious_priors(mal_emb, cones, config.prior)

    bundle = ModelBundle(
        axes=cones.axes,
        benign_prior=benign_prior,
        malicious_priors=mal_priors,
        config=config,
    )
    save_model(args.output, bundle)
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
