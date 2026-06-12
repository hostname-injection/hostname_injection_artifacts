from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence


_PERCENT_RE = re.compile(r"%[0-9A-Fa-f]{2}")
_PERCENT_RUN_RE = re.compile(r"(?:%[0-9A-Fa-f]{2})+")

# Small confusable map (extensible)
HOMOGLYPHS = {
    "a": ["а", "ɑ"],  # Cyrillic/Latin
    "e": ["е"],
    "i": ["і", "ı"],
    "o": ["о", "օ"],
    "c": ["с"],
    "p": ["р"],
    "y": ["у"],
    "A": ["Α"],
    "B": ["Β"],
    "E": ["Ε"],
    "H": ["Н"],
    "K": ["Κ"],
    "M": ["Μ"],
    "O": ["Ο"],
    "P": ["Ρ"],
    "T": ["Τ"],
    "X": ["Χ"],
}

TLD_CONFUSABLES = {
    "com": ["co"],
    "net": ["ne"],
    "org": ["or"],
    "io": ["lo", "ia"],
}

QUOTE_COMMENT_FRAGMENTS = ["'", '"', "`", "${", "}", "--", "/*", "*/", "#"]
EDIT_MANIFEST_VERSION = "Eraw-public-v1"


@dataclass
class EditOp:
    name: str
    func: Callable[[str, random.Random], str]


def _random_char_index(s: str, rng: random.Random) -> Optional[int]:
    if not s:
        return None
    return rng.randrange(0, len(s))


def edit_percent_encoding(host: str, rng: random.Random) -> str:
    # decode one %HH or encode one char
    if _PERCENT_RE.search(host) and rng.random() < 0.5:
        # decode first occurrence
        match = _PERCENT_RE.search(host)
        if match:
            byte = int(match.group(0)[1:], 16)
            ch = chr(byte)
            return host[:match.start()] + ch + host[match.end():]
    # encode a random ASCII char
    idx = _random_char_index(host, rng)
    if idx is None:
        return host
    ch = host[idx]
    return host[:idx] + f"%{ord(ch):02X}" + host[idx + 1:]


def edit_homoglyph(host: str, rng: random.Random) -> str:
    candidates = [i for i, ch in enumerate(host) if ch in HOMOGLYPHS]
    if not candidates:
        return host
    i = rng.choice(candidates)
    repl = rng.choice(HOMOGLYPHS[host[i]])
    return host[:i] + repl + host[i + 1:]


def edit_delimiter(host: str, rng: random.Random) -> str:
    if not host:
        return host
    if rng.random() < 0.5 and ('.' in host or '-' in host):
        # toggle delimiter
        idxs = [i for i, ch in enumerate(host) if ch in '.-']
        i = rng.choice(idxs)
        new = '-' if host[i] == '.' else '.'
        return host[:i] + new + host[i + 1:]
    # insert delimiter
    i = rng.randrange(1, len(host))
    delim = rng.choice(['.', '-'])
    return host[:i] + delim + host[i:]


def edit_label_split_merge(host: str, rng: random.Random) -> str:
    labels = host.split('.')
    if len(labels) >= 2 and rng.random() < 0.5:
        # merge two adjacent labels
        i = rng.randrange(0, len(labels) - 1)
        merged = labels[i] + labels[i + 1]
        new_labels = labels[:i] + [merged] + labels[i + 2:]
        return '.'.join(new_labels)
    # split a label
    i = rng.randrange(0, len(labels))
    label = labels[i]
    if len(label) < 2:
        return host
    j = rng.randrange(1, len(label))
    new_labels = labels[:i] + [label[:j], label[j:]] + labels[i + 1:]
    return '.'.join(new_labels)


def edit_case(host: str, rng: random.Random) -> str:
    idxs = [i for i, ch in enumerate(host) if ch.isalpha()]
    if not idxs:
        return host
    i = rng.choice(idxs)
    ch = host[i]
    flipped = ch.upper() if ch.islower() else ch.lower()
    return host[:i] + flipped + host[i + 1:]


def edit_zero_pad(host: str, rng: random.Random) -> str:
    labels = host.split('.')
    numeric_idxs = [i for i, l in enumerate(labels) if l.isdigit()]
    if not numeric_idxs:
        return host
    i = rng.choice(numeric_idxs)
    label = labels[i]
    if label.startswith('0') and len(label) > 1 and rng.random() < 0.5:
        labels[i] = label.lstrip('0') or '0'
    else:
        labels[i] = '0' + label
    return '.'.join(labels)


def edit_utf8_percent(host: str, rng: random.Random) -> str:
    # Encode non-ASCII chars to percent-encoded UTF-8 or decode existing
    decodable_runs = []
    for match in _PERCENT_RUN_RE.finditer(host):
        try:
            raw = bytes(int(part, 16) for part in match.group(0).split("%") if part)
            decodable_runs.append((match.start(), match.end(), raw.decode("utf-8")))
        except UnicodeDecodeError:
            continue
    if decodable_runs and rng.random() < 0.5:
        start, end, decoded = rng.choice(decodable_runs)
        return host[:start] + decoded + host[end:]
    if _PERCENT_RE.search(host) and rng.random() < 0.5:
        # decode one byte as a mixed-decoding residue
        match = _PERCENT_RE.search(host)
        if match:
            byte = int(match.group(0)[1:], 16)
            ch = bytes([byte]).decode('latin1')
            return host[:match.start()] + ch + host[match.end():]
    # encode a non-ascii char
    idxs = [i for i, ch in enumerate(host) if ord(ch) > 127]
    if not idxs:
        return host
    i = rng.choice(idxs)
    ch = host[i]
    encoded = ''.join(f"%{b:02X}" for b in ch.encode('utf-8'))
    return host[:i] + encoded + host[i + 1:]


def edit_unicode_normalize(host: str, rng: random.Random) -> str:
    form = rng.choice(['NFC', 'NFKC'])
    try:
        return unicodedata.normalize(form, host)
    except Exception:
        return host


def edit_punycode(host: str, rng: random.Random) -> str:
    labels = host.split('.')
    idxs = [i for i, l in enumerate(labels) if any(ord(ch) > 127 for ch in l) or l.startswith('xn--')]
    if not idxs:
        return host
    i = rng.choice(idxs)
    label = labels[i]
    try:
        import idna  # type: ignore
        if label.startswith('xn--'):
            labels[i] = idna.decode(label, uts46=True)
        else:
            labels[i] = idna.encode(label, uts46=True).decode('ascii')
    except Exception:
        try:
            if label.startswith('xn--'):
                labels[i] = label.encode('ascii').decode('idna')
            else:
                labels[i] = label.encode('idna').decode('ascii')
        except Exception:
            return host
    return '.'.join(labels)


def edit_label_transpose(host: str, rng: random.Random) -> str:
    labels = host.split('.')
    if len(labels) < 3:
        return host
    # avoid swapping TLD (last label)
    i = rng.randrange(0, len(labels) - 2)
    labels[i], labels[i + 1] = labels[i + 1], labels[i]
    return '.'.join(labels)


def edit_tld_swap(host: str, rng: random.Random) -> str:
    labels = host.split('.')
    if len(labels) < 2:
        return host
    tld = labels[-1]
    if tld not in TLD_CONFUSABLES:
        return host
    labels[-1] = rng.choice(TLD_CONFUSABLES[tld])
    return '.'.join(labels)


def edit_quote_comment_fragment(host: str, rng: random.Random) -> str:
    fragment = rng.choice(QUOTE_COMMENT_FRAGMENTS)
    idx = rng.randrange(0, len(host) + 1) if host else 0
    return host[:idx] + fragment + host[idx:]


DEFAULT_EDITS = {
    "E1_percent": EditOp("E1_percent", edit_percent_encoding),
    "E2_homoglyph": EditOp("E2_homoglyph", edit_homoglyph),
    "E3_delimiter": EditOp("E3_delimiter", edit_delimiter),
    "E4_label_split": EditOp("E4_label_split", edit_label_split_merge),
    "E5_case": EditOp("E5_case", edit_case),
    "E5_zero_pad": EditOp("E5_zero_pad", edit_zero_pad),
    "E6_utf8_percent": EditOp("E6_utf8_percent", edit_utf8_percent),
    "E7_unicode_norm": EditOp("E7_unicode_norm", edit_unicode_normalize),
    "E8_punycode": EditOp("E8_punycode", edit_punycode),
    "E9_label_transpose": EditOp("E9_label_transpose", edit_label_transpose),
    "E10_tld_swap": EditOp("E10_tld_swap", edit_tld_swap),
    "E11_quote_comment": EditOp("E11_quote_comment", edit_quote_comment_fragment),
}


class EditModel:
    """Apply hostname-artifact edit operations from the deployed manifest."""
    def __init__(self, edits: Optional[Sequence[str]] = None, version: str = EDIT_MANIFEST_VERSION):
        self.version = version
        self.edits = [DEFAULT_EDITS[e] for e in (edits or DEFAULT_EDITS.keys())]

    def apply_random(self, host: str, rng: Optional[random.Random] = None) -> str:
        rng = rng or random.Random()
        op = rng.choice(self.edits)
        return op.func(host, rng)

    def apply_k(self, host: str, k: int, rng: Optional[random.Random] = None) -> str:
        rng = rng or random.Random()
        out = host
        for _ in range(k):
            out = self.apply_random(out, rng=rng)
        return out
