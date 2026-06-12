#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys


def _build_command(args) -> list[str]:
    cmd = [
        "ccd",
        "train-caho-corpus",
        "--benign-dir",
        args.benign_dir,
        "--out",
        args.out,
    ]
    if args.malicious_jsonl_dir:
        cmd += ["--malicious-jsonl-dir", args.malicious_jsonl_dir]
    if args.malicious_txt_dir:
        cmd += ["--malicious-txt-dir", args.malicious_txt_dir]
    if args.jsonl_key:
        cmd += ["--jsonl-key", args.jsonl_key]
    if args.csv_hostname_col:
        cmd += ["--csv-hostname-col", args.csv_hostname_col]
    if args.min_length is not None:
        cmd += ["--min-length", str(args.min_length)]
    if args.no_dedup:
        cmd += ["--no-dedup"]
    if args.malicious_family:
        cmd += ["--malicious-family", args.malicious_family]
    if args.model:
        cmd += ["--model", args.model]
    if args.epochs is not None:
        cmd += ["--epochs", str(args.epochs)]
    if args.batch_size is not None:
        cmd += ["--batch-size", str(args.batch_size)]
    if args.lr is not None:
        cmd += ["--lr", str(args.lr)]
    if args.temperature is not None:
        cmd += ["--temperature", str(args.temperature)]
    if args.loss:
        cmd += ["--loss", args.loss]
    if args.augmenter:
        cmd += ["--augmenter", args.augmenter]
    if args.weighted_num_augs is not None:
        cmd += ["--weighted-num-augs", str(args.weighted_num_augs)]
    if args.weighted_max_attempts is not None:
        cmd += ["--weighted-max-attempts", str(args.weighted_max_attempts)]
    if args.weighted_no_retry:
        cmd += ["--weighted-no-retry"]
    if args.max_grad_norm is not None:
        cmd += ["--max-grad-norm", str(args.max_grad_norm)]
    if args.scheduler:
        cmd += ["--scheduler", args.scheduler]
    if args.min_lr is not None:
        cmd += ["--min-lr", str(args.min_lr)]
    if args.grad_cache:
        cmd += ["--grad-cache"]
    if args.grad_cache_chunk_size is not None:
        cmd += ["--grad-cache-chunk-size", str(args.grad_cache_chunk_size)]
    if args.contrastive_loss:
        cmd += ["--contrastive-loss", args.contrastive_loss]
    if args.contrastive_max_scale is not None:
        cmd += ["--contrastive-max-scale", str(args.contrastive_max_scale)]
    if args.contrastive_min_scale is not None:
        cmd += ["--contrastive-min-scale", str(args.contrastive_min_scale)]
    if args.optimize_contrastive_scale:
        cmd += ["--optimize-contrastive-scale"]
    if args.num_workers is not None:
        cmd += ["--num-workers", str(args.num_workers)]
    if args.empty_cache:
        cmd += ["--empty-cache"]
    if args.device:
        cmd += ["--device", args.device]
    if args.resume:
        cmd += ["--resume"]
    if args.save_best:
        cmd += ["--save-best"]
    if args.no_save_final:
        cmd += ["--no-save-final"]
    if args.no_normalize:
        cmd += ["--no-normalize"]
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benign-dir", required=True)
    parser.add_argument("--malicious-jsonl-dir", default=None)
    parser.add_argument("--malicious-txt-dir", default=None)
    parser.add_argument("--jsonl-key", default="hostname")
    parser.add_argument("--csv-hostname-col", default="Hostname")
    parser.add_argument("--min-length", type=int, default=5)
    parser.add_argument("--no-dedup", action="store_true")
    parser.add_argument("--malicious-family", default="corpus")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--loss", choices=["supcon", "contrastive"], default="supcon")
    parser.add_argument("--augmenter", choices=["edit", "weighted", "hybrid"], default="edit")
    parser.add_argument("--weighted-num-augs", type=int, default=2)
    parser.add_argument("--weighted-max-attempts", type=int, default=3)
    parser.add_argument("--weighted-no-retry", action="store_true")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--scheduler", choices=["cosine", "none"], default="cosine")
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--grad-cache", action="store_true")
    parser.add_argument("--grad-cache-chunk-size", type=int, default=128)
    parser.add_argument("--contrastive-loss", choices=["fixed", "learnable"], default="fixed")
    parser.add_argument("--contrastive-max-scale", type=float, default=100.0)
    parser.add_argument("--contrastive-min-scale", type=float, default=1.0)
    parser.add_argument("--optimize-contrastive-scale", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--empty-cache", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-best", action="store_true")
    parser.add_argument("--no-save-final", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")
    args = parser.parse_args()

    cmd = _build_command(args)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
