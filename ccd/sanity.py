"""Run a CCD sanity check on user_login usernames with a CAHO checkpoint."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .config import CCDConfig
from .encoder import CahoEncoder
from .model import CCDModel
from .preprocess import normalize_hostname


def _sample_usernames(
    data_dir: Path,
    per_class: int,
    seed: int,
) -> Tuple[List[str], List[str], int, int]:
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {data_dir}")

    rng = random.Random(seed)
    rng.shuffle(files)

    benign: List[str] = []
    malicious: List[str] = []
    seen = set()
    files_scanned = 0

    for path in files:
        files_scanned += 1
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                username = (row.get("USERNAME") or "").strip()
                if not username:
                    continue
                if username in seen:
                    continue
                sonnet = (row.get("GPT_5_5_IS_DNS_CMD_INJECTION") or "").strip()
                opus = (row.get("CLAUDE_OPUS_4_8_IS_DNS_CMD_INJECTION") or "").strip()
                is_malicious = (sonnet == "M") or (opus == "M")
                if is_malicious:
                    if len(malicious) < per_class:
                        malicious.append(username)
                        seen.add(username)
                else:
                    if len(benign) < per_class:
                        benign.append(username)
                        seen.add(username)
                if len(benign) >= per_class and len(malicious) >= per_class:
                    break
        if len(benign) >= per_class and len(malicious) >= per_class:
            break

    return benign, malicious, files_scanned, len(files)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CCD sanity check on user_login usernames.")
    parser.add_argument(
        "--user-logins-dir",
        type=Path,
        default=Path("hostname_injection_benchmark/user_logins"),
        help="Directory containing user_login CSV files.",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=50,
        help="Number of benign and malicious usernames to sample.",
    )
    parser.add_argument("--seed", type=int, default=786, help="Random seed for file order.")
    parser.add_argument(
        "--checkpoint",
        default="caho_model_checkpoint",
        help="CAHO checkpoint path or SentenceTransformer name.",
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for encoding.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto|cpu|cuda|mps",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable hostname normalization prior to encoding.",
    )
    parser.add_argument(
        "--allow-imperfect",
        action="store_true",
        help="Return success even if accuracy is not 1.0.",
    )
    parser.add_argument(
        "--approximate",
        action="store_true",
        help="Use fast approximate scoring (hard-cone).",
    )
    parser.add_argument(
        "--approximate-k",
        type=int,
        default=None,
        help="Top-k cones to use for approximate scoring (implies --approximate).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.per_class <= 0:
        raise ValueError("--per-class must be positive")

    benign, malicious, scanned, total = _sample_usernames(
        args.user_logins_dir,
        args.per_class,
        args.seed,
    )
    print(f"Sampled {len(benign)} benign and {len(malicious)} malicious usernames.")
    print(f"Files scanned: {scanned} of {total} (stopped early: {scanned < total})")

    if not benign or not malicious:
        raise RuntimeError("Insufficient samples for one or both classes.")

    if args.no_normalize:
        benign_norm = benign
        mal_norm = malicious
    else:
        benign_norm = [normalize_hostname(u) for u in benign]
        mal_norm = [normalize_hostname(u) for u in malicious]

    config = CCDConfig()
    config.encoder.model_name = args.checkpoint
    config.encoder.device = args.device
    encoder = CahoEncoder(config.encoder)

    benign_emb = encoder.encode(benign_norm, batch_size=args.batch_size, normalize=True)
    mal_emb = encoder.encode(mal_norm, batch_size=args.batch_size, normalize=True)
    model = CCDModel.from_embeddings(benign_emb, {"user_login": mal_emb}, config=config)

    hosts = benign_norm + mal_norm
    labels = np.array([0] * len(benign_norm) + [1] * len(mal_norm), dtype=int)
    embeddings = encoder.encode(hosts, batch_size=args.batch_size, normalize=True)
    scores = model.score_embeddings(
        embeddings,
        approximate=args.approximate,
        approximate_k=args.approximate_k,
    )
    preds = (scores > 0.0).astype(int)

    acc = float((preds == labels).mean())
    print(f"Accuracy: {acc:.4f}")

    if acc != 1.0:
        mismatch_idx = np.where(preds != labels)[0]
        print(f"Mismatches: {len(mismatch_idx)}")
        for i in mismatch_idx[:10]:
            print(
                f"  idx={i} username={hosts[i]!r} label={labels[i]} "
                f"pred={preds[i]} score={scores[i]:.4f}"
            )
        if not args.allow_imperfect:
            return 1

    print("Sanity check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
