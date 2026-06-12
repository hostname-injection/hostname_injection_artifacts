from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlsplit


_PERCENT_RUN_RE = re.compile(r"(?:%[0-9A-Fa-f]{2})+")


@dataclass(frozen=True)
class ArtifactSegmentation:
    scheme: str | None
    authority_present: bool
    userinfo_present: bool
    port_present: bool
    path_present: bool
    query_present: bool
    fragment_present: bool
    bracketed_ipv6: bool


def _percent_decode(s: str) -> str:
    def repl(m):
        token = m.group(0)
        raw = bytes(int(part, 16) for part in token.split("%") if part)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return "".join(chr(byte) for byte in raw)

    return _PERCENT_RUN_RE.sub(repl, s)


def _split_recorded_artifact(value: str) -> tuple[str, ArtifactSegmentation]:
    text = value.strip()
    scheme = None
    authority_present = False
    path_present = False
    query_present = False
    fragment_present = False
    host = text
    if "://" in text:
        split = urlsplit(text)
        scheme = split.scheme or None
        authority_present = bool(split.netloc)
        path_present = bool(split.path and split.path not in {"", "/"})
        query_present = bool(split.query)
        fragment_present = bool(split.fragment)
        host = split.netloc or split.path
    else:
        fragment_present = "#" in host
        host = host.split("#", 1)[0]
        query_present = "?" in host
        host = host.split("?", 1)[0]
        path_present = "/" in host
        host = host.split("/", 1)[0]

    # remove userinfo
    userinfo_present = "@" in host
    if "@" in host:
        host = host.rsplit("@", 1)[1]

    # remove port
    port_present = False
    bracketed_ipv6 = host.startswith("[") and "]" in host
    if bracketed_ipv6:
        end = host.find("]")
        suffix = host[end + 1:]
        port_present = suffix.startswith(":") and len(suffix) > 1
        host = host[1:end]
    elif ":" in host and host.count(":") == 1:
        port_present = True
        host = host.rsplit(":", 1)[0]

    return host.strip("[]"), ArtifactSegmentation(
        scheme=scheme,
        authority_present=authority_present,
        userinfo_present=userinfo_present,
        port_present=port_present,
        path_present=path_present,
        query_present=query_present,
        fragment_present=fragment_present,
        bracketed_ipv6=bracketed_ipv6,
    )


def normalization_trace(
    hostname: str,
    *,
    unicode_form: str = "NFKC",
    decode_percent: bool = True,
    idna_roundtrip: bool = True,
) -> dict[str, Any]:
    """Return the deployed normalizer output plus release-safe trace metadata."""
    raw_input = "" if hostname is None else str(hostname)
    stripped_input = raw_input.strip()
    host, segmentation = _split_recorded_artifact(stripped_input)
    host = host.strip().strip(".")
    before_percent_decode = host

    percent_decode_error = False
    if decode_percent:
        try:
            host = _percent_decode(host)
        except Exception:
            percent_decode_error = True

    after_percent_decode = host
    unicode_error = False
    try:
        host = unicodedata.normalize(unicode_form, host)
    except Exception:
        unicode_error = True

    after_unicode = host
    host = host.casefold()

    idna_backend = None
    idna_error = False
    if idna_roundtrip:
        try:
            import idna  # type: ignore

            ascii_host = idna.encode(host, uts46=True).decode("ascii")
            host = idna.decode(ascii_host, uts46=True)
            idna_backend = "idna"
        except Exception:
            try:
                ascii_host = host.encode("idna").decode("ascii")
                host = ascii_host.encode("ascii").decode("idna")
                idna_backend = "stdlib"
            except Exception:
                idna_error = True

    return {
        "raw_input": raw_input,
        "stripped_input": stripped_input,
        "segmentation": asdict(segmentation),
        "host_before_percent_decode": before_percent_decode,
        "host_after_percent_decode": after_percent_decode,
        "percent_decode_changed": before_percent_decode != after_percent_decode,
        "percent_decode_error": percent_decode_error,
        "unicode_form": unicode_form,
        "host_after_unicode_normalization": after_unicode,
        "unicode_normalization_error": unicode_error,
        "casefolded": True,
        "idna_roundtrip": idna_roundtrip,
        "idna_backend": idna_backend,
        "idna_error": idna_error,
        "normalized_hostname": host,
    }


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
    return str(
        normalization_trace(
            hostname,
            unicode_form=unicode_form,
            decode_percent=decode_percent,
            idna_roundtrip=idna_roundtrip,
        )["normalized_hostname"]
    )


def is_valid_hostname(hostname: str) -> bool:
    """Basic DNS length checks (ASCII/IDNA normalized)."""
    if not hostname or len(hostname) > 253:
        return False
    labels = hostname.split(".")
    for label in labels:
        if not label or len(label) > 63:
            return False
    return True
