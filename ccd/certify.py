from __future__ import annotations

import base64
import binascii
import math
import random
import re
import unicodedata
from dataclasses import dataclass
from statistics import NormalDist
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from .edit_model import DEFAULT_EDITS, HOMOGLYPHS, QUOTE_COMMENT_FRAGMENTS, TLD_CONFUSABLES, EditModel


_PERCENT_RUN_RE = re.compile(r"(?:%[0-9A-Fa-f]{2})+")
_HEX_LABEL_RE = re.compile(r"[0-9A-Fa-f]{2,}")
_BASE64URL_LABEL_RE = re.compile(r"[A-Za-z0-9_-]{2,}")


def cone_margin(prototypes: np.ndarray, u: np.ndarray) -> Tuple[int, float]:
    """Return predicted class index and margin between top-1 and top-2.

    prototypes: (C, d) unit vectors
    u: (d,) unit vector
    """
    if prototypes.ndim != 2 or prototypes.shape[0] < 2:
        raise ValueError("cone_margin requires at least two prototype vectors")
    sims = prototypes @ u
    top2 = np.argpartition(sims, -2)[-2:]
    top2 = top2[np.argsort(sims[top2])[::-1]]
    c_hat = int(top2[0])
    margin = float(sims[top2[0]] - sims[top2[1]])
    return c_hat, margin


def cone_margin_radius(margin: float, b_max: float) -> int:
    if b_max <= 0:
        return 0
    return int(math.floor(margin / (2.0 * b_max)))


@dataclass(frozen=True)
class DecisionCertificate:
    certified: bool
    prediction: bool
    method: str
    radius: int
    threshold: float
    base_score: float
    margin: float
    checked: int
    max_score_movement: float
    counterexample: Optional[str] = None
    decision_rule: str = "score > threshold"


def _validate_certificate_radius(radius: int) -> None:
    if radius < 0:
        raise ValueError("radius must be non-negative")


def _require_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _selected_edit_names(edit_model: Optional[EditModel]) -> List[str]:
    if edit_model is None:
        return list(DEFAULT_EDITS.keys())
    return [op.name for op in edit_model.edits]


def _percent_spans(host: str) -> Iterable[Tuple[int, int, str]]:
    i = 0
    hexdigits = set("0123456789abcdefABCDEF")
    while i <= len(host) - 3:
        token = host[i:i + 3]
        if token[0] == "%" and token[1] in hexdigits and token[2] in hexdigits:
            yield i, i + 3, chr(int(token[1:], 16))
            i += 3
        else:
            i += 1


def _utf8_percent_runs(host: str) -> Iterable[Tuple[int, int, str]]:
    for match in _PERCENT_RUN_RE.finditer(host):
        try:
            raw = bytes(int(part, 16) for part in match.group(0).split("%") if part)
            yield match.start(), match.end(), raw.decode("utf-8")
        except UnicodeDecodeError:
            continue


def _punycode_variants(label: str) -> Set[str]:
    out: Set[str] = set()
    try:
        import idna  # type: ignore

        if label.startswith("xn--"):
            out.add(idna.decode(label, uts46=True))
        elif any(ord(ch) > 127 for ch in label):
            out.add(idna.encode(label, uts46=True).decode("ascii"))
    except Exception:
        try:
            if label.startswith("xn--"):
                out.add(label.encode("ascii").decode("idna"))
            elif any(ord(ch) > 127 for ch in label):
                out.add(label.encode("idna").decode("ascii"))
        except Exception:
            pass
    out.discard(label)
    return out


def _safe_decoded_label(value: str) -> bool:
    return bool(value) and all(ch.isprintable() and ch not in "\r\n\t." for ch in value)


def _hex_base_variants(label: str) -> Set[str]:
    out: Set[str] = set()
    if label:
        out.add(label.encode("utf-8").hex())
        out.add(base64.urlsafe_b64encode(label.encode("utf-8")).decode("ascii").rstrip("="))
    if len(label) % 2 == 0 and _HEX_LABEL_RE.fullmatch(label):
        try:
            decoded = bytes.fromhex(label).decode("utf-8")
            if _safe_decoded_label(decoded):
                out.add(decoded)
        except (UnicodeDecodeError, ValueError):
            pass
    if _BASE64URL_LABEL_RE.fullmatch(label):
        padded = label + "=" * (-len(label) % 4)
        try:
            decoded = base64.b64decode(padded.encode("ascii"), altchars=b"-_", validate=True).decode("utf-8")
            if _safe_decoded_label(decoded):
                out.add(decoded)
        except (binascii.Error, UnicodeDecodeError, ValueError):
            pass
    out.discard(label)
    return out


def deterministic_single_edit_neighbors(host: str, edit_model: Optional[EditModel] = None) -> List[str]:
    """Enumerate one-step neighbors for the fixed edit manifest.

    This is the deterministic closure used for emitted stability certificates.
    Random edit sampling can still be used to prioritize hard cases, but the
    certificate itself should be based on this full finite neighborhood.
    """
    names = set(_selected_edit_names(edit_model))
    out: Set[str] = set()

    if "E1_percent" in names:
        for start, end, decoded in _percent_spans(host):
            out.add(host[:start] + decoded + host[end:])
        for i, ch in enumerate(host):
            if ord(ch) <= 0x7F:
                out.add(host[:i] + f"%{ord(ch):02X}" + host[i + 1:])

    if "E2_homoglyph" in names:
        for i, ch in enumerate(host):
            for repl in HOMOGLYPHS.get(ch, []):
                out.add(host[:i] + repl + host[i + 1:])

    if "E3_delimiter" in names:
        for i, ch in enumerate(host):
            if ch in ".-":
                out.add(host[:i] + ("-" if ch == "." else ".") + host[i + 1:])
        for i in range(1, len(host)):
            out.add(host[:i] + "." + host[i:])
            out.add(host[:i] + "-" + host[i:])

    labels = host.split(".")
    if "E4_label_split" in names:
        if len(labels) >= 2:
            for i in range(len(labels) - 1):
                merged = labels[i] + labels[i + 1]
                out.add(".".join(labels[:i] + [merged] + labels[i + 2:]))
        for i, label in enumerate(labels):
            for j in range(1, len(label)):
                out.add(".".join(labels[:i] + [label[:j], label[j:]] + labels[i + 1:]))

    if "E5_case" in names:
        for i, ch in enumerate(host):
            if ch.isalpha():
                out.add(host[:i] + (ch.upper() if ch.islower() else ch.lower()) + host[i + 1:])

    if "E5_zero_pad" in names:
        for i, label in enumerate(labels):
            if label.isdigit():
                padded = list(labels)
                if label.startswith("0") and len(label) > 1:
                    padded[i] = label.lstrip("0") or "0"
                else:
                    padded[i] = "0" + label
                out.add(".".join(padded))

    if "E6_utf8_percent" in names:
        for start, end, decoded in _utf8_percent_runs(host):
            out.add(host[:start] + decoded + host[end:])
        for start, end, decoded in _percent_spans(host):
            out.add(host[:start] + decoded + host[end:])
        for i, ch in enumerate(host):
            if ord(ch) > 127:
                encoded = "".join(f"%{b:02X}" for b in ch.encode("utf-8"))
                out.add(host[:i] + encoded + host[i + 1:])

    if "E7_unicode_norm" in names:
        for form in ("NFC", "NFKC"):
            try:
                out.add(unicodedata.normalize(form, host))
            except Exception:
                pass

    if "E8_punycode" in names:
        for i, label in enumerate(labels):
            for variant in _punycode_variants(label):
                changed = list(labels)
                changed[i] = variant
                out.add(".".join(changed))

    if "E9_label_transpose" in names and len(labels) >= 3:
        for i in range(0, len(labels) - 2):
            swapped = list(labels)
            swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
            out.add(".".join(swapped))

    if "E10_tld_swap" in names and len(labels) >= 2:
        for repl in TLD_CONFUSABLES.get(labels[-1], []):
            swapped = list(labels)
            swapped[-1] = repl
            out.add(".".join(swapped))

    if "E11_quote_comment" in names:
        for i in range(0, len(host) + 1):
            for fragment in QUOTE_COMMENT_FRAGMENTS:
                out.add(host[:i] + fragment + host[i:])

    if "E12_hex_base" in names:
        for i, label in enumerate(labels):
            for variant in _hex_base_variants(label):
                changed = list(labels)
                changed[i] = variant
                out.add(".".join(changed))

    out.discard(host)
    return sorted(out)


def enumerate_edit_ball(
    host: str,
    radius: int,
    *,
    edit_model: Optional[EditModel] = None,
    max_nodes: int = 10000,
) -> List[str]:
    """Deterministically enumerate the manifest edit ball up to ``radius``."""
    _validate_certificate_radius(radius)
    if max_nodes <= 0:
        raise ValueError("max_nodes must be positive")

    seen = {host}
    frontier = [host]
    for _depth in range(radius):
        next_frontier: List[str] = []
        for item in frontier:
            for neighbor in deterministic_single_edit_neighbors(item, edit_model):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                next_frontier.append(neighbor)
                if len(seen) > max_nodes:
                    raise ValueError("edit ball exceeded max_nodes")
        frontier = next_frontier
        if not frontier:
            break
    return sorted(seen)


def certify_by_margin_bound(
    score: float,
    threshold: float,
    delta_bound: float,
    *,
    radius: int,
) -> DecisionCertificate:
    """Certify a decision from an externally supplied score-movement bound."""
    _validate_certificate_radius(radius)
    score = _require_finite(score, "score")
    threshold = _require_finite(threshold, "threshold")
    delta_bound = _require_finite(delta_bound, "delta_bound")
    if delta_bound < 0.0:
        raise ValueError("delta_bound must be non-negative")
    prediction = score > threshold
    margin = score - threshold
    certified = margin > delta_bound if prediction else -margin >= delta_bound
    return DecisionCertificate(
        certified=certified,
        prediction=prediction,
        method="margin_bound",
        radius=radius,
        threshold=float(threshold),
        base_score=float(score),
        margin=float(margin),
        checked=0,
        max_score_movement=float(delta_bound),
    )


def log_ratio_envelope(
    benign_prior: np.ndarray,
    malicious_priors: Dict[str, np.ndarray],
    *,
    eps: float = 1e-12,
) -> float:
    """Compute the smoothed log-ratio envelope A_epsilon."""
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if not malicious_priors:
        raise ValueError("malicious_priors is empty")
    log_benign = np.log(np.maximum(np.asarray(benign_prior, dtype=np.float64), eps))
    envelope = 0.0
    for prior in malicious_priors.values():
        log_malicious = np.log(np.maximum(np.asarray(prior, dtype=np.float64), eps))
        envelope = max(envelope, float(np.max(np.abs(log_malicious - log_benign))))
    return envelope


def calibrated_margin_delta(
    *,
    effective_count: float,
    log_ratio_bound: float,
    sketch_lipschitz: float,
    embedding_rotation_bound: float,
) -> float:
    """Compute Delta_K = n0 * A_epsilon * L_Q * B_K(r)."""
    values = (effective_count, log_ratio_bound, sketch_lipschitz, embedding_rotation_bound)
    if any(value < 0.0 or not math.isfinite(value) for value in values):
        raise ValueError("margin-bound inputs must be finite and non-negative")
    return float(effective_count * log_ratio_bound * sketch_lipschitz * embedding_rotation_bound)


def certify_by_calibrated_margin(
    score: float,
    threshold: float,
    *,
    radius: int,
    effective_count: float,
    benign_prior: np.ndarray,
    malicious_priors: Dict[str, np.ndarray],
    sketch_lipschitz: float,
    embedding_rotation_bound: float,
    eps: float = 1e-12,
) -> DecisionCertificate:
    """Certify with the CMC bound from Appendix C."""
    _validate_certificate_radius(radius)
    delta = calibrated_margin_delta(
        effective_count=effective_count,
        log_ratio_bound=log_ratio_envelope(benign_prior, malicious_priors, eps=eps),
        sketch_lipschitz=sketch_lipschitz,
        embedding_rotation_bound=embedding_rotation_bound,
    )
    cert = certify_by_margin_bound(score, threshold, delta, radius=radius)
    return DecisionCertificate(
        certified=cert.certified,
        prediction=cert.prediction,
        method="calibrated_margin",
        radius=cert.radius,
        threshold=cert.threshold,
        base_score=cert.base_score,
        margin=cert.margin,
        checked=cert.checked,
        max_score_movement=cert.max_score_movement,
        counterexample=cert.counterexample,
    )


def certify_by_enumeration(
    host: str,
    score_fn: Callable[[str], float],
    threshold: float,
    *,
    radius: int,
    edit_model: Optional[EditModel] = None,
    normalizer: Optional[Callable[[str], str]] = None,
    max_nodes: int = 10000,
) -> DecisionCertificate:
    """Certify stability by exact deterministic edit-ball closure."""
    _validate_certificate_radius(radius)
    threshold = _require_finite(threshold, "threshold")
    normalize = normalizer or (lambda s: s)
    base_score = _require_finite(score_fn(normalize(host)), "base_score")
    prediction = base_score > threshold
    margin = base_score - threshold
    max_movement = 0.0
    checked = 0

    for candidate in enumerate_edit_ball(host, radius, edit_model=edit_model, max_nodes=max_nodes):
        candidate_score = _require_finite(score_fn(normalize(candidate)), "candidate_score")
        checked += 1
        max_movement = max(max_movement, abs(candidate_score - base_score))
        candidate_prediction = candidate_score > threshold
        if candidate_prediction != prediction:
            return DecisionCertificate(
                certified=False,
                prediction=prediction,
                method="enumeration",
                radius=radius,
                threshold=float(threshold),
                base_score=base_score,
                margin=float(margin),
                checked=checked,
                max_score_movement=float(max_movement),
                counterexample=candidate,
            )

    return DecisionCertificate(
        certified=True,
        prediction=prediction,
        method="enumeration",
        radius=radius,
        threshold=float(threshold),
        base_score=base_score,
        margin=float(margin),
        checked=checked,
        max_score_movement=float(max_movement),
    )


def clopper_pearson_interval(k: int, n: int, alpha: float) -> Tuple[float, float]:
    """Compute Clopper-Pearson interval for Bernoulli proportion.

    Uses a normal approximation if scipy is unavailable.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if n == 0:
        return 0.0, 1.0
    try:
        from scipy.stats import beta  # type: ignore

        lower = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
        upper = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
        return float(lower), float(upper)
    except Exception:
        # Wilson score interval approximation
        z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
        phat = k / n
        denom = 1 + z**2 / n
        center = (phat + z**2 / (2 * n)) / denom
        half = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n) / denom
        return max(0.0, center - half), min(1.0, center + half)


def randomized_smoothing_certificate(
    classifier: Callable[[str], int],
    x: str,
    edit_sampler: Callable[[str, random.Random], str],
    num_samples: int,
    alpha: float,
    rng: Optional[random.Random] = None,
) -> Tuple[bool, int, Tuple[float, float]]:
    """Exploratory randomized smoothing over an edit channel.

    Returns (certified, predicted_class, (p_lower, p_upper)).
    CCD's emitted finite-edit certificates should still use deterministic
    closure over the fixed manifest before being treated as deployment proof.
    """
    rng = rng or random.Random()
    votes: Dict[int, int] = {}
    for _ in range(num_samples):
        x_edit = edit_sampler(x, rng)
        pred = classifier(x_edit)
        votes[pred] = votes.get(pred, 0) + 1
    # top-2
    sorted_votes = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
    if len(sorted_votes) == 0:
        return False, -1, (0.0, 1.0)
    top_class, top_count = sorted_votes[0]
    second_count = sorted_votes[1][1] if len(sorted_votes) > 1 else 0
    pL, pU = clopper_pearson_interval(top_count, num_samples, alpha)
    qL, qU = clopper_pearson_interval(second_count, num_samples, alpha)
    certified = pL > qU
    return certified, int(top_class), (pL, pU)
