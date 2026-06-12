from __future__ import annotations

from pathlib import Path
from typing import List


def read_nonempty_lines(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(errors="ignore").splitlines() if line.strip()]


def read_parallel_lines(path: Path, expected_len: int, field_name: str) -> List[str]:
    values = [line.strip() for line in path.read_text(errors="ignore").splitlines()]
    empty_lines = [index + 1 for index, value in enumerate(values) if not value]
    if empty_lines:
        shown = ", ".join(str(line) for line in empty_lines[:5])
        suffix = "" if len(empty_lines) <= 5 else f", ... ({len(empty_lines)} total)"
        raise ValueError(f"{field_name} file contains empty values at line(s) {shown}{suffix}")
    if len(values) != expected_len:
        raise ValueError(f"{field_name} file has {len(values)} rows; expected {expected_len}")
    return values
