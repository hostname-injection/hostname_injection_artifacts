from __future__ import annotations

import base64
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

from .edit_model import EditModel
from .preprocess import normalize_hostname


@dataclass
class AugmentConfig:
    benign_min_edits: int = 0
    benign_max_edits: int = 1
    malicious_min_edits: int = 1
    malicious_max_edits: int = 3
    include_nested_encoding: bool = True
    normalize_input: bool = True
    use_edit_model: bool = True
    use_weighted_augs: bool = False
    weighted: "WeightedAugmentConfig" = field(default_factory=lambda: WeightedAugmentConfig())


@dataclass
class WeightedAugmentConfig:
    num_augs: int = 2
    benign_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTED_BENIGN_WEIGHTS))
    malicious_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTED_MALICIOUS_WEIGHTS))
    retry_on_no_change: bool = True
    max_attempts: int = 3


def _urlencode_layer(s: str) -> str:
    out = []
    for ch in s:
        o = ord(ch)
        if 0x30 <= o <= 0x39 or 0x41 <= o <= 0x5A or 0x61 <= o <= 0x7A or ch in '-._~':
            out.append(ch)
        else:
            out.append(f"%{o:02X}")
    return ''.join(out)


def _base64_layer(s: str) -> str:
    return base64.urlsafe_b64encode(s.encode('utf-8')).decode('ascii').strip('=')


def _base32_layer(s: str) -> str:
    return base64.b32encode(s.encode('utf-8')).decode('ascii').strip('=')


def _nested_escape(s: str) -> str:
    # simple nested escape: replace backslash with double escape
    return s.replace('\\', '\\\\').replace('%', '%25')


SYNONYM_MAP: Dict[str, list[str]] = {
    "application": ["app"],
    "administrator": ["admin"],
    "mobile": ["m"],
    "dashboard": ["dash"],
    "server": ["srv"],
    "portal": ["dashboard", "hub", "console", "interface"],
    "auth": ["login", "signin", "access", "authenticate"],
    "mail": ["email", "inbox", "webmail", "msg"],
    "cdn": ["assets", "static", "resources", "media"],
    "user": ["account", "member", "profile", "client"],
    "verify": ["confirm", "check", "validate", "review"],
    "support": ["help", "desk", "service", "assistance"],
}

AUG_HOMOGLYPHS: Dict[str, list[str]] = {
    "a": ["@", "4", "à", "á", "â", "ä"],
    "b": ["8"],
    "e": ["3", "€", "ë"],
    "g": ["9", "q"],
    "i": ["1", "!", "|", "í", "ï"],
    "l": ["1", "|", "!", "£"],
    "o": ["0", "ø", "ó", "ö"],
    "s": ["5", "$", "z"],
    "t": ["7", "+"],
    "z": ["2"],
}

DEFAULT_WEIGHTED_BENIGN_WEIGHTS: Dict[str, float] = {
    "random_case_variation": 0.1,
    "shuffle_subdomains": 0.1,
    "truncate_subdomain": 0.2,
    "dropout_random_char": 0.3,
    "synonym_swap": 0.2,
    "typo_swap": 0.2,
    "punctuation_replace": 0.25,
    "letter_swap_typo": 0.3,
}

DEFAULT_WEIGHTED_MALICIOUS_WEIGHTS: Dict[str, float] = {
    **DEFAULT_WEIGHTED_BENIGN_WEIGHTS,
    "toggle_protocol": 0.3,
    "base64_encode_parts": 0.3,
    "hex_encode_parts": 0.3,
    "url_encode_parts": 0.3,
    "random_homoglyph_substitution": 0.15,
    "quote_comment_fragment": 0.2,
}


def _random_case_variation(hostname: str, rng: random.Random) -> str:
    return "".join(rng.choice([c.lower(), c.upper()]) for c in hostname)


def _shuffle_subdomains(hostname: str, rng: random.Random) -> str:
    parts = hostname.split(".")
    if len(parts) <= 2:
        return hostname
    subdomains = parts[:-2]
    rng.shuffle(subdomains)
    return ".".join(subdomains + parts[-2:])


def _truncate_subdomain(hostname: str, rng: random.Random) -> str:
    del rng
    parts = hostname.split(".")
    if len(parts) > 2:
        return ".".join(parts[1:])
    return hostname


def _dropout_random_char(hostname: str, rng: random.Random, dropout_prob: float = 0.1) -> str:
    return "".join(c for c in hostname if rng.random() > dropout_prob or c == ".")


def _synonym_swap(hostname: str, rng: random.Random) -> str:
    parts = hostname.split(".")
    new_parts = []
    for part in parts:
        if any(ch.isdigit() for ch in part) or len(part) <= 4:
            new_parts.append(part)
            continue
        swapped = False
        lower = part.lower()
        for key, synonyms in SYNONYM_MAP.items():
            if key in lower:
                new_part = lower.replace(key, rng.choice(synonyms))
                if part.istitle():
                    new_part = new_part.title()
                elif part.isupper():
                    new_part = new_part.upper()
                new_parts.append(new_part)
                swapped = True
                break
        if not swapped:
            new_parts.append(part)
    return ".".join(new_parts)


def _typo_swap(hostname: str, rng: random.Random) -> str:
    if len(hostname) < 3:
        return hostname
    idx = rng.randint(0, len(hostname) - 2)
    if hostname[idx].isalnum() and hostname[idx + 1].isalnum():
        chars = list(hostname)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        return "".join(chars)
    return hostname


def _toggle_protocol(hostname: str, rng: random.Random) -> str:
    protocol = ""
    if hostname.startswith("http://"):
        protocol = "http://"
        hostname = hostname[7:]
    elif hostname.startswith("https://"):
        protocol = "https://"
        hostname = hostname[8:]
    else:
        protocol = rng.choice(["", "http://", "https://"])

    if hostname.startswith("www."):
        hostname = hostname[4:]
    else:
        hostname = "www." + hostname
    return protocol + hostname


def _punctuation_replace(hostname: str, rng: random.Random) -> str:
    del rng
    if "-" in hostname:
        return hostname.replace("-", "", 1)
    if "_" in hostname:
        return hostname.replace("_", "-", 1)
    return hostname.replace(".", "-", 1)


def _letter_swap_typo(hostname: str, rng: random.Random) -> str:
    indices = [i for i in range(len(hostname) - 1) if hostname[i].isalnum() and hostname[i + 1].isalnum()]
    if not indices:
        return hostname
    idx = rng.choice(indices)
    swapped = list(hostname)
    swapped[idx], swapped[idx + 1] = swapped[idx + 1], swapped[idx]
    return "".join(swapped)


def _base64_encode_parts(hostname: str, rng: random.Random, encode_ratio: float = 0.2) -> str:
    parts = hostname.split(".")
    for i, part in enumerate(parts):
        if rng.random() < encode_ratio:
            parts[i] = base64.urlsafe_b64encode(part.encode("utf-8")).decode("ascii").strip("=")
    return ".".join(parts)


def _hex_encode_parts(hostname: str, rng: random.Random, encode_ratio: float = 0.2) -> str:
    parts = hostname.split(".")
    for i, part in enumerate(parts):
        if rng.random() < encode_ratio:
            parts[i] = part.encode("utf-8").hex()
    return ".".join(parts)


def _url_encode_parts(hostname: str, rng: random.Random, encode_ratio: float = 0.5) -> str:
    parts = hostname.split(".")
    for i, part in enumerate(parts):
        if rng.random() < encode_ratio:
            parts[i] = "".join(f"%{hex(ord(c))[2:]}" for c in part)
    return ".".join(parts)


def _random_homoglyph_substitution(hostname: str, rng: random.Random) -> str:
    out = []
    for ch in hostname:
        if ch in AUG_HOMOGLYPHS and rng.random() < 0.3:
            out.append(rng.choice(AUG_HOMOGLYPHS[ch]))
        else:
            out.append(ch)
    return "".join(out)


def _quote_comment_fragment(hostname: str, rng: random.Random) -> str:
    fragments = ["'", '"', "`", "${", "}", "--", "/*", "*/", "#"]
    idx = rng.randrange(0, len(hostname) + 1) if hostname else 0
    return hostname[:idx] + rng.choice(fragments) + hostname[idx:]


AUG_FUNCTIONS: Dict[str, Callable[[str, random.Random], str]] = {
    "random_case_variation": _random_case_variation,
    "shuffle_subdomains": _shuffle_subdomains,
    "truncate_subdomain": _truncate_subdomain,
    "dropout_random_char": _dropout_random_char,
    "synonym_swap": _synonym_swap,
    "typo_swap": _typo_swap,
    "toggle_protocol": _toggle_protocol,
    "punctuation_replace": _punctuation_replace,
    "letter_swap_typo": _letter_swap_typo,
    "base64_encode_parts": _base64_encode_parts,
    "hex_encode_parts": _hex_encode_parts,
    "url_encode_parts": _url_encode_parts,
    "random_homoglyph_substitution": _random_homoglyph_substitution,
    "quote_comment_fragment": _quote_comment_fragment,
}


def _apply_weighted_augmentations(
    hostname: str,
    *,
    weights: Dict[str, float],
    num_augs: int,
    rng: random.Random,
    retry_on_no_change: bool,
    max_attempts: int,
) -> str:
    if not weights or num_augs <= 0:
        return hostname
    names = list(weights.keys())
    probs = list(weights.values())
    for _ in range(max(1, max_attempts)):
        selected = rng.choices(names, weights=probs, k=num_augs)
        out = hostname
        for name in selected:
            func = AUG_FUNCTIONS.get(name)
            if func is None:
                continue
            out = func(out, rng)
        if not retry_on_no_change or out != hostname:
            return out
    return hostname


class CAHOAugmenter:
    """Class-aware augmentation pipeline for CAHO."""
    def __init__(
        self,
        config: AugmentConfig = AugmentConfig(),
        benign_edits: Optional[EditModel] = None,
        malicious_edits: Optional[EditModel] = None,
    ) -> None:
        self.config = config
        self.benign_edits = benign_edits or EditModel(edits=[
            "E3_delimiter",
            "E4_label_split",
            "E5_case",
            "E5_zero_pad",
            "E7_unicode_norm",
            "E8_punycode",
        ])
        self.malicious_edits = malicious_edits or EditModel()  # full set

    def augment(self, host: str, *, is_malicious: bool, rng: Optional[random.Random] = None) -> str:
        rng = rng or random.Random()
        if self.config.normalize_input:
            host = normalize_hostname(host)
        out = host
        if is_malicious:
            if self.config.use_edit_model:
                k = rng.randint(self.config.malicious_min_edits, self.config.malicious_max_edits)
                out = self.malicious_edits.apply_k(out, k, rng=rng)
                # extra obfuscation layers
                if rng.random() < 0.5:
                    out = _urlencode_layer(out)
                if rng.random() < 0.3:
                    out = _base64_layer(out)
                if rng.random() < 0.2:
                    out = _base32_layer(out)
                if self.config.include_nested_encoding and rng.random() < 0.2:
                    out = _nested_escape(out)
            if self.config.use_weighted_augs:
                out = _apply_weighted_augmentations(
                    out,
                    weights=self.config.weighted.malicious_weights,
                    num_augs=self.config.weighted.num_augs,
                    rng=rng,
                    retry_on_no_change=self.config.weighted.retry_on_no_change,
                    max_attempts=self.config.weighted.max_attempts,
                )
            return out
        # benign
        if self.config.use_edit_model:
            k = rng.randint(self.config.benign_min_edits, self.config.benign_max_edits)
            out = self.benign_edits.apply_k(out, k, rng=rng) if k > 0 else out
        if self.config.use_weighted_augs:
            out = _apply_weighted_augmentations(
                out,
                weights=self.config.weighted.benign_weights,
                num_augs=self.config.weighted.num_augs,
                rng=rng,
                retry_on_no_change=self.config.weighted.retry_on_no_change,
                max_attempts=self.config.weighted.max_attempts,
            )
        return out
