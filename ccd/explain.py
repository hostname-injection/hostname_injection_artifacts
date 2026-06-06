"""Explain CCD scores by exposing top cone contributions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from .calibration import SPLIT_CONFORMAL_DECISION_RULE, require_calibrated_threshold
from .encoder import require_model_uses_trained_caho_checkpoint
from .io import load_model
from .line_io import read_nonempty_lines, read_parallel_lines
from .preprocess import normalize_hostname


def read_lines(path: Path) -> List[str]:
    return read_nonempty_lines(path)


def add_explain_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--model", required=True, type=Path, help="CCD .npz model bundle")
    parser.add_argument("--input", required=True, type=Path, help="Input file of hostnames (one per line)")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    parser.add_argument("--groups", type=Path, default=None, help="Optional one-calibration-group-per-input-row file.")
    parser.add_argument(
        "--require-group-thresholds",
        action="store_true",
        help="Fail if --groups contains a group missing from grouped calibration output.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of top cones to show")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-normalize", action="store_true", help="Skip hostname normalization")
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Explain CCD predictions.")
    return add_explain_arguments(parser)


def run(args: argparse.Namespace) -> int:
    model = load_model(args.model)
    require_model_uses_trained_caho_checkpoint(model, purpose="ccd explain")
    hostnames = read_lines(args.input)
    if not args.no_normalize:
        hostnames = [normalize_hostname(h) for h in hostnames]
    groups = read_parallel_lines(args.groups, len(hostnames), "groups") if args.groups else None
    threshold = require_calibrated_threshold(model, purpose="ccd explain")
    threshold_source = "model_bundle_threshold"
    grouped_thresholds = getattr(model, "grouped_thresholds", None)
    grouped_thresholds_source = "model_bundle_grouped_thresholds" if grouped_thresholds else "none"

    explanations = model.explain(
        hostnames,
        batch_size=args.batch_size,
        normalize=False,
        top_k=args.top_k,
        calibration_groups=groups,
        missing_group_threshold="error" if args.require_group_thresholds else "default",
    )
    for index, row in enumerate(explanations):
        row["decision_rule"] = SPLIT_CONFORMAL_DECISION_RULE
        row_threshold_source = threshold_source
        if groups is not None:
            group_name = str(groups[index]).strip()
            if grouped_thresholds and group_name in grouped_thresholds:
                row_threshold_source = grouped_thresholds_source
            elif not args.require_group_thresholds:
                row_threshold_source = f"{threshold_source}_fallback"
        row["threshold_source"] = row_threshold_source

    payload = {
        "model": str(args.model),
        "count": len(explanations),
        "top_k": args.top_k,
        "threshold": threshold,
        "threshold_source": threshold_source,
        "grouped_thresholds_source": grouped_thresholds_source,
        "grouped_thresholds_used": groups is not None,
        "decision_rule": SPLIT_CONFORMAL_DECISION_RULE,
        "score_path": {
            "exact_all_cones": True,
            "score_statistic": "deployed_top_r_cone_sketch",
            "normalized_inputs": not args.no_normalize,
        },
        "normalizer": {
            "enabled": not args.no_normalize,
            "function": "ccd.preprocess.normalize_hostname" if not args.no_normalize else None,
            "unicode_form": "NFKC" if not args.no_normalize else None,
            "decode_percent": True if not args.no_normalize else None,
            "decode_utf8_percent_runs": True if not args.no_normalize else None,
            "idna_roundtrip": True if not args.no_normalize else None,
        },
        "explanations": explanations,
    }

    if args.output:
        args.output.write_text(json.dumps(payload, indent=2))
        print(f"Wrote explanations to {args.output}")
        return 0

    print(json.dumps(payload, indent=2))
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
