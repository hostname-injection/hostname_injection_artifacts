from __future__ import annotations

import re
import unicodedata
from urllib.parse import urlsplit


_PERCENT_RE = re.compile(r"%([0-9A-Fa-f]{2})")


def _percent_decode(s: str) -> str:
    def repl(m):
        return chr(int(m.group(1), 16))
    return _PERCENT_RE.sub(repl, s)


def _strip_scheme_and_path(host: str) -> str:
    if "://" in host:
        split = urlsplit(host)
        host = split.netloc or split.path
    # remove userinfo
    if "@" in host:
        host = host.split("@", 1)[1]
    # remove port
    if ":" in host and host.count(":") == 1:
        host = host.split(":", 1)[0]
    # IPv6 in brackets
    host = host.strip("[]")
    return host


def normalize_hostname(
    hostname: str,
    *,
    unicode_form: str = "NFKC",
    decode_percent: bool = True,
    idna_roundtrip: bool = True,
) -> str:
    """Normalize hostname input.

    The paper describes normalization via IDNA2008 ToASCII/ToUnicode,
    case-folding, and NFC/NFKC. We approximate with available stdlib
    facilities and optional idna package if present.
    """
    if hostname is None:
        return ""
    host = hostname.strip()
    host = _strip_scheme_and_path(host)
    host = host.strip().strip(".")

    if decode_percent:
        try:
            host = _percent_decode(host)
        except Exception:
            pass

    try:
        host = unicodedata.normalize(unicode_form, host)
    except Exception:
        pass

    host = host.casefold()

    if idna_roundtrip:
        # Prefer the external idna package (IDNA2008) if installed.
        try:
            import idna  # type: ignore

            ascii_host = idna.encode(host, uts46=True).decode("ascii")
            host = idna.decode(ascii_host, uts46=True)
        except Exception:
            # Fallback to stdlib codec
            try:
                ascii_host = host.encode("idna").decode("ascii")
                host = ascii_host.encode("ascii").decode("idna")
            except Exception:
                pass

    return host


def is_valid_hostname(hostname: str) -> bool:
    """Basic DNS length checks (ASCII/IDNA normalized)."""
    if not hostname or len(hostname) > 253:
        return False
    labels = hostname.split(".")
    for label in labels:
        if not label or len(label) > 63:
            return False
    return True
