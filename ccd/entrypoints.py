"""Entry points for convenience commands."""

from __future__ import annotations

import sys


def _dispatch(module: str) -> int:
    mod = __import__(module, fromlist=["main"])
    return int(mod.main())


def diagnose_main() -> None:
    sys.exit(_dispatch("ccd.diagnostics"))


def explain_main() -> None:
    sys.exit(_dispatch("ccd.explain"))


def score_main() -> None:
    sys.exit(_dispatch("ccd.score_cli"))
