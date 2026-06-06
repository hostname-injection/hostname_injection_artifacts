#!/usr/bin/env python3
from __future__ import annotations

import sys
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    from ccd.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["calibrate", *(sys.argv[1:] if argv is None else argv)])
    args.func(args)


if __name__ == "__main__":
    main()
