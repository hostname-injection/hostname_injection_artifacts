import random

import numpy as np

from ccd.certify import (
    calibrated_margin_delta,
    certify_by_calibrated_margin,
    certify_by_enumeration,
    certify_by_margin_bound,
    clopper_pearson_interval,
    cone_margin,
    cone_margin_radius,
    deterministic_single_edit_neighbors,
    enumerate_edit_ball,
    log_ratio_envelope,
    randomized_smoothing_certificate,
)
from ccd.edit_model import EditModel


def test_cone_margin_and_radius():
    prototypes = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    u = np.array([1.0, 0.0], dtype=np.float32)

    predicted, margin = cone_margin(prototypes, u)

    assert predicted == 0
    assert margin == 1.0
    assert cone_margin_radius(margin, b_max=0.2) == 2


def test_cone_margin_requires_two_prototypes():
    prototypes = np.array([[1.0, 0.0]], dtype=np.float32)
    u = np.array([1.0, 0.0], dtype=np.float32)

    try:
        cone_margin(prototypes, u)
    except ValueError as exc:
        assert "at least two" in str(exc)
        return

    assert False, "Expected ValueError for fewer than two prototypes"


def test_clopper_pearson_interval_bounds_probability():
    lower, upper = clopper_pearson_interval(k=8, n=10, alpha=0.05)

    assert 0.0 <= lower <= 0.8 <= upper <= 1.0


def test_randomized_smoothing_certifies_stable_classifier():
    def classifier(_text: str) -> int:
        return 1

    def sampler(text: str, _rng: random.Random) -> str:
        return text

    certified, predicted, interval = randomized_smoothing_certificate(
        classifier,
        "example.com",
        sampler,
        num_samples=40,
        alpha=0.05,
        rng=random.Random(0),
    )

    assert certified is True
    assert predicted == 1
    assert interval[0] > 0.0


def test_randomized_smoothing_rejects_tied_votes():
    state = {"i": 0}

    def classifier(text: str) -> int:
        return int(text.endswith("1"))

    def sampler(_text: str, _rng: random.Random) -> str:
        state["i"] += 1
        return f"sample-{state['i'] % 2}"

    certified, predicted, interval = randomized_smoothing_certificate(
        classifier,
        "example.com",
        sampler,
        num_samples=40,
        alpha=0.05,
        rng=random.Random(0),
    )

    assert certified is False
    assert predicted in {0, 1}
    assert 0.0 <= interval[0] <= interval[1] <= 1.0


def test_deterministic_single_edit_neighbors_cover_manifest_ops():
    neighbors = deterministic_single_edit_neighbors(
        "a.com",
        EditModel(edits=["E3_delimiter", "E10_tld_swap", "E11_quote_comment"]),
    )

    assert "a-com" in neighbors
    assert "a.co" in neighbors
    assert "a.com--" in neighbors


def test_utf8_percent_neighbors_decode_complete_runs():
    neighbors = deterministic_single_edit_neighbors(
        "caf%C3%A9.example",
        EditModel(edits=["E6_utf8_percent"]),
    )

    assert "café.example" in neighbors


def test_hex_base_neighbors_encode_and_decode_labels():
    encoded = deterministic_single_edit_neighbors(
        "evil.example",
        EditModel(edits=["E12_hex_base"]),
    )
    decoded_hex = deterministic_single_edit_neighbors(
        "6576696c.example",
        EditModel(edits=["E12_hex_base"]),
    )
    decoded_base64 = deterministic_single_edit_neighbors(
        "ZXZpbA.example",
        EditModel(edits=["E12_hex_base"]),
    )

    assert "6576696c.example" in encoded
    assert "ZXZpbA.example" in encoded
    assert "evil.example" in decoded_hex
    assert "evil.example" in decoded_base64


def test_enumerate_edit_ball_includes_origin_and_radius_neighbors():
    edit_model = EditModel(edits=["E5_case"])
    ball = enumerate_edit_ball("ab.com", radius=1, edit_model=edit_model)

    assert "ab.com" in ball
    assert "Ab.com" in ball
    assert "aB.com" in ball


def test_certify_by_margin_bound_positive_and_benign():
    positive = certify_by_margin_bound(1.5, 1.0, 0.4, radius=2)
    benign = certify_by_margin_bound(0.2, 1.0, 0.8, radius=2)

    assert positive.certified is True
    assert positive.prediction is True
    assert benign.certified is True
    assert benign.prediction is False


def test_calibrated_margin_delta_and_certificate():
    benign_prior = np.array([0.8, 0.2], dtype=np.float32)
    malicious_priors = {"m": np.array([0.2, 0.8], dtype=np.float32)}

    envelope = log_ratio_envelope(benign_prior, malicious_priors)
    delta = calibrated_margin_delta(
        effective_count=2.0,
        log_ratio_bound=envelope,
        sketch_lipschitz=0.1,
        embedding_rotation_bound=0.2,
    )
    cert = certify_by_calibrated_margin(
        score=1.0,
        threshold=0.0,
        radius=1,
        effective_count=2.0,
        benign_prior=benign_prior,
        malicious_priors=malicious_priors,
        sketch_lipschitz=0.1,
        embedding_rotation_bound=0.2,
    )

    assert envelope > 0.0
    assert delta > 0.0
    assert cert.certified is True
    assert cert.method == "calibrated_margin"


def test_certify_by_enumeration_accepts_stable_decision():
    edit_model = EditModel(edits=["E5_case"])

    def score_fn(text: str) -> float:
        return 1.0

    cert = certify_by_enumeration(
        "ab.com",
        score_fn,
        threshold=0.0,
        radius=1,
        edit_model=edit_model,
    )

    assert cert.certified is True
    assert cert.prediction is True
    assert cert.counterexample is None


def test_certify_by_enumeration_rejects_flip():
    edit_model = EditModel(edits=["E5_case"])

    def score_fn(text: str) -> float:
        return -1.0 if text.startswith("A") else 1.0

    cert = certify_by_enumeration(
        "ab.com",
        score_fn,
        threshold=0.0,
        radius=1,
        edit_model=edit_model,
    )

    assert cert.certified is False
    assert cert.counterexample == "Ab.com"
