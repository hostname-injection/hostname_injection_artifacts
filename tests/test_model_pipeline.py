import json

import numpy as np

from ccd.calibration import calibrate_threshold, conformal_p_value
from ccd.cone import ConePartition
from ccd.config import CCDConfig, ConeConfig, EncoderConfig
from ccd.encoder import CahoEncoder
from ccd.edit_model import EditModel
from ccd.io import ModelBundle, load_model, save_model
from ccd.model import CCDModel
from ccd.priors import build_benign_prior, build_malicious_priors
from ccd.scoring import ccd_scores, soft_mixture_score


def _identity_cones():
    axes = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    config = ConeConfig(dim=2, num_cones=2, active_cones=1, temperature=1.0, use_lsh=False)
    return ConePartition.build(config, axes=axes)


def test_ccd_scores_signs():
    cones = _identity_cones()
    benign_prior = np.array([0.9, 0.1], dtype=np.float32)
    malicious_priors = {"m": np.array([0.1, 0.9], dtype=np.float32)}
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    scores = ccd_scores(embeddings, cones, benign_prior, malicious_priors)
    assert scores[0] < 0.0
    assert scores[1] > 0.0


def test_soft_mixture_score_basic():
    cones = _identity_cones()
    benign_prior = np.array([0.9, 0.1], dtype=np.float32)
    malicious_priors = {"m1": np.array([0.1, 0.9], dtype=np.float32)}
    score = soft_mixture_score(np.array([1.0, 0.0], dtype=np.float32), cones, benign_prior, malicious_priors)
    assert score < 0.0


def test_build_priors_prefers_cone():
    cones = _identity_cones()
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    benign_prior = build_benign_prior(embeddings, cones)
    malicious_priors = build_malicious_priors({"fam": embeddings}, cones)
    assert benign_prior[0] > benign_prior[1]
    assert malicious_priors["fam"][0] > malicious_priors["fam"][1]


def test_save_load_roundtrip(tmp_path):
    axes = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    benign_prior = np.array([0.7, 0.3], dtype=np.float32)
    malicious_priors = {"fam": np.array([0.2, 0.8], dtype=np.float32)}
    config = CCDConfig(cone=ConeConfig(dim=2, num_cones=2, active_cones=1, use_lsh=False))

    bundle = ModelBundle(
        axes=axes,
        benign_prior=benign_prior,
        malicious_priors=malicious_priors,
        config=config,
        threshold=0.42,
        grouped_thresholds={"tenant-a": {"threshold": 0.73, "num_samples": 2}},
    )
    path = tmp_path / "model.npz"
    save_model(path, bundle)

    model = load_model(path)
    assert np.allclose(model.cones.axes, axes)
    assert np.allclose(model.benign_prior, benign_prior)
    assert np.allclose(model.malicious_priors["fam"], malicious_priors["fam"])
    assert model.config.to_dict() == config.to_dict()
    assert model.threshold == 0.42
    assert model.grouped_thresholds == {"tenant-a": {"threshold": 0.73, "num_samples": 2}}


def _write_raw_model_bundle(
    path,
    *,
    axes=None,
    benign_prior=None,
    malicious_priors=None,
    malicious_names=None,
    threshold=None,
    grouped_thresholds=None,
):
    config = CCDConfig(cone=ConeConfig(dim=2, num_cones=2, active_cones=1, use_lsh=False))
    payload = {
        "axes": np.eye(2, dtype=np.float32) if axes is None else axes,
        "benign_prior": np.array([0.7, 0.3], dtype=np.float32)
        if benign_prior is None
        else benign_prior,
        "malicious_priors": np.array([[0.2, 0.8]], dtype=np.float32)
        if malicious_priors is None
        else malicious_priors,
        "malicious_names": np.array(["fam"]) if malicious_names is None else malicious_names,
        "config": np.array([json.dumps(config.to_dict())]),
        "format_version": np.array(["1.2"]),
    }
    if threshold is not None:
        payload["threshold"] = np.array([threshold], dtype=np.float64)
    if grouped_thresholds is not None:
        payload["grouped_thresholds"] = np.array([json.dumps(grouped_thresholds)])
    np.savez(path, **payload)


def test_load_model_rejects_non_finite_threshold(tmp_path):
    path = tmp_path / "bad_threshold.npz"
    _write_raw_model_bundle(path, threshold=float("nan"))

    try:
        load_model(path)
    except ValueError as exc:
        assert "threshold" in str(exc)
        assert "finite" in str(exc)
        return

    assert False, "Expected ValueError for non-finite serialized threshold"


def test_load_model_rejects_invalid_grouped_thresholds(tmp_path):
    path = tmp_path / "bad_grouped_thresholds.npz"
    _write_raw_model_bundle(path, threshold=0.5, grouped_thresholds={"tenant-a": {"num_samples": 2}})

    try:
        load_model(path)
    except ValueError as exc:
        assert "threshold" in str(exc)
        return

    assert False, "Expected ValueError for grouped threshold without threshold value"


def test_load_model_rejects_non_finite_axes(tmp_path):
    path = tmp_path / "bad_axes.npz"
    axes = np.array([[1.0, 0.0], [float("nan"), 1.0]], dtype=np.float32)
    _write_raw_model_bundle(path, axes=axes)

    try:
        load_model(path)
    except ValueError as exc:
        assert "axes" in str(exc)
        assert "finite" in str(exc)
        return

    assert False, "Expected ValueError for non-finite cone axes"


def test_load_model_rejects_axis_shape_mismatch(tmp_path):
    path = tmp_path / "bad_axis_shape.npz"
    _write_raw_model_bundle(path, axes=np.eye(3, dtype=np.float32))

    try:
        load_model(path)
    except ValueError as exc:
        assert "axes shape" in str(exc)
        assert "config" in str(exc)
        return

    assert False, "Expected ValueError for cone axes that do not match the config"


def test_load_model_rejects_non_finite_priors(tmp_path):
    path = tmp_path / "bad_prior.npz"
    _write_raw_model_bundle(path, benign_prior=np.array([0.7, float("inf")], dtype=np.float32))

    try:
        load_model(path)
    except ValueError as exc:
        assert "benign_prior" in str(exc)
        assert "finite" in str(exc)
        return

    assert False, "Expected ValueError for non-finite serialized priors"


def test_load_model_rejects_negative_or_unnormalized_priors(tmp_path):
    negative_path = tmp_path / "negative_prior.npz"
    _write_raw_model_bundle(
        negative_path,
        malicious_priors=np.array([[-0.2, 1.2]], dtype=np.float32),
    )

    try:
        load_model(negative_path)
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("Expected ValueError for negative serialized priors")

    unnormalized_path = tmp_path / "unnormalized_prior.npz"
    _write_raw_model_bundle(
        unnormalized_path,
        malicious_priors=np.array([[0.2, 0.7]], dtype=np.float32),
    )

    try:
        load_model(unnormalized_path)
    except ValueError as exc:
        assert "sum to 1.0" in str(exc)
        return

    assert False, "Expected ValueError for unnormalized serialized priors"


def test_load_model_rejects_malicious_name_prior_mismatch(tmp_path):
    path = tmp_path / "bad_malicious_shape.npz"
    _write_raw_model_bundle(
        path,
        malicious_priors=np.array([[0.2, 0.8], [0.1, 0.9]], dtype=np.float32),
        malicious_names=np.array(["fam"]),
    )

    try:
        load_model(path)
    except ValueError as exc:
        assert "malicious_priors shape" in str(exc)
        return

    assert False, "Expected ValueError for mismatched malicious names and priors"


def test_save_model_rejects_invalid_bundle_arrays(tmp_path):
    config = CCDConfig(cone=ConeConfig(dim=2, num_cones=2, active_cones=1, use_lsh=False))
    bad_bundle = ModelBundle(
        axes=np.eye(2, dtype=np.float32),
        benign_prior=np.array([0.7, 0.2], dtype=np.float32),
        malicious_priors={"fam": np.array([0.2, 0.8], dtype=np.float32)},
        config=config,
    )

    try:
        save_model(tmp_path / "bad_model.npz", bad_bundle)
    except ValueError as exc:
        assert "benign_prior" in str(exc)
        assert "sum to 1.0" in str(exc)
        return

    assert False, "Expected ValueError for invalid ModelBundle priors"


def test_save_model_rejects_empty_malicious_priors(tmp_path):
    config = CCDConfig(cone=ConeConfig(dim=2, num_cones=2, active_cones=1, use_lsh=False))
    bad_bundle = ModelBundle(
        axes=np.eye(2, dtype=np.float32),
        benign_prior=np.array([0.7, 0.3], dtype=np.float32),
        malicious_priors={},
        config=config,
    )

    try:
        save_model(tmp_path / "empty_malicious.npz", bad_bundle)
    except ValueError as exc:
        assert "malicious_priors is empty" in str(exc)
        return

    assert False, "Expected ValueError for a ModelBundle without malicious priors"


def test_calibration_and_p_value():
    scores = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    threshold = calibrate_threshold(scores, alpha=0.25)
    assert threshold in scores
    p_val = conformal_p_value(0.3, scores)
    assert 0.0 < p_val <= 1.0
    assert conformal_p_value(0.4, scores) < conformal_p_value(0.1, scores)


def test_refresh_benign_reference_updates_only_pb_and_threshold():
    cones = _identity_cones()
    model = CCDModel(
        config=CCDConfig(cone=cones.config),
        encoder=CahoEncoder(EncoderConfig(model_name="sentence-transformers/all-MiniLM-L6-v2")),
        cones=cones,
        benign_prior=np.array([0.95, 0.05], dtype=np.float32),
        malicious_priors={"m": np.array([0.1, 0.9], dtype=np.float32)},
        threshold=9.0,
    )
    old_axes = model.cones.axes.copy()
    old_malicious = model.malicious_priors["m"].copy()
    old_score = float(model.score_embeddings(np.array([[0.0, 1.0]], dtype=np.float32))[0])

    benign_window = np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    report = model.refresh_benign_reference(benign_window, alpha=0.5)
    new_score = float(model.score_embeddings(np.array([[0.0, 1.0]], dtype=np.float32))[0])

    assert report["refresh_scope"]["benign_prior_updated"] is True
    assert report["refresh_scope"]["malicious_priors_fixed"] is True
    assert report["num_samples"] == 3
    assert report["old_threshold"] == 9.0
    assert report["benign_prior_l1_delta"] > 0.0
    assert np.allclose(model.cones.axes, old_axes)
    assert np.allclose(model.malicious_priors["m"], old_malicious)
    assert model.benign_prior[1] > model.benign_prior[0]
    assert model.threshold == report["threshold"]
    assert new_score != old_score


def test_refresh_benign_reference_updates_grouped_thresholds_when_groups_supplied():
    cones = _identity_cones()
    model = CCDModel(
        config=CCDConfig(cone=cones.config),
        encoder=CahoEncoder(EncoderConfig(model_name="sentence-transformers/all-MiniLM-L6-v2")),
        cones=cones,
        benign_prior=np.array([0.95, 0.05], dtype=np.float32),
        malicious_priors={"m": np.array([0.1, 0.9], dtype=np.float32)},
        threshold=9.0,
        grouped_thresholds={"old": {"threshold": 9.0}},
    )

    benign_window = np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    report = model.refresh_benign_reference(
        benign_window,
        alpha=0.5,
        calibration_groups=["tenant-a", "tenant-a", "tenant-b", "tenant-b"],
    )

    assert report["threshold_source"] == "grouped_benign_refresh_scores"
    assert report["old_n_calibration_groups"] == 1
    assert report["n_calibration_groups"] == 2
    assert report["refresh_scope"]["grouped_thresholds_updated"] is True
    assert set(model.grouped_thresholds or {}) == {"tenant-a", "tenant-b"}
    assert model.grouped_thresholds == report["grouped_thresholds"]


def test_refresh_benign_reference_requires_groups_for_grouped_model():
    cones = _identity_cones()
    model = CCDModel(
        config=CCDConfig(cone=cones.config),
        encoder=CahoEncoder(EncoderConfig(model_name="sentence-transformers/all-MiniLM-L6-v2")),
        cones=cones,
        benign_prior=np.array([0.95, 0.05], dtype=np.float32),
        malicious_priors={"m": np.array([0.1, 0.9], dtype=np.float32)},
        threshold=9.0,
        grouped_thresholds={"tenant-a": {"threshold": 9.0}},
    )
    old_prior = model.benign_prior.copy()
    old_threshold = model.threshold
    old_grouped = model.grouped_thresholds
    benign_window = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

    try:
        model.refresh_benign_reference(benign_window, alpha=0.5)
    except ValueError as exc:
        assert "calibration_groups are required" in str(exc)
    else:
        raise AssertionError("grouped refresh without replacement groups should fail")

    assert np.allclose(model.benign_prior, old_prior)
    assert model.threshold == old_threshold
    assert model.grouped_thresholds is old_grouped


def test_refresh_benign_reference_rolls_back_group_calibration_failures():
    cones = _identity_cones()
    model = CCDModel(
        config=CCDConfig(cone=cones.config),
        encoder=CahoEncoder(EncoderConfig(model_name="sentence-transformers/all-MiniLM-L6-v2")),
        cones=cones,
        benign_prior=np.array([0.95, 0.05], dtype=np.float32),
        malicious_priors={"m": np.array([0.1, 0.9], dtype=np.float32)},
        threshold=9.0,
        grouped_thresholds={"tenant-a": {"threshold": 9.0}},
    )
    query = np.array([[0.0, 1.0]], dtype=np.float32)
    old_prior = model.benign_prior.copy()
    old_threshold = model.threshold
    old_grouped = model.grouped_thresholds
    old_score = float(model.score_embeddings(query)[0])
    benign_window = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

    try:
        model.refresh_benign_reference(
            benign_window,
            alpha=0.5,
            calibration_groups=["tenant-a"],
        )
    except ValueError as exc:
        assert "same length" in str(exc)
    else:
        raise AssertionError("grouped refresh with mismatched groups should fail")

    assert np.allclose(model.benign_prior, old_prior)
    assert model.threshold == old_threshold
    assert model.grouped_thresholds is old_grouped
    assert float(model.score_embeddings(query)[0]) == old_score


def test_refresh_benign_reference_can_explicitly_drop_grouped_thresholds():
    cones = _identity_cones()
    model = CCDModel(
        config=CCDConfig(cone=cones.config),
        encoder=CahoEncoder(EncoderConfig(model_name="sentence-transformers/all-MiniLM-L6-v2")),
        cones=cones,
        benign_prior=np.array([0.95, 0.05], dtype=np.float32),
        malicious_priors={"m": np.array([0.1, 0.9], dtype=np.float32)},
        threshold=9.0,
        grouped_thresholds={"tenant-a": {"threshold": 9.0}},
    )
    benign_window = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

    report = model.refresh_benign_reference(
        benign_window,
        alpha=0.5,
        drop_grouped_thresholds=True,
    )

    assert model.grouped_thresholds is None
    assert report["grouped_thresholds_dropped"] is True
    assert report["refresh_scope"]["grouped_thresholds_dropped"] is True
    assert report["refresh_scope"]["grouped_thresholds_updated"] is False
    assert report["n_calibration_groups"] == 0


def test_update_benign_prior_rejects_empty_embeddings():
    cones = _identity_cones()
    model = CCDModel(
        config=CCDConfig(cone=cones.config),
        encoder=CahoEncoder(EncoderConfig(model_name="sentence-transformers/all-MiniLM-L6-v2")),
        cones=cones,
        benign_prior=np.array([0.5, 0.5], dtype=np.float32),
        malicious_priors={"m": np.array([0.1, 0.9], dtype=np.float32)},
    )

    try:
        model.update_benign_prior(np.empty((0, 2), dtype=np.float32))
    except ValueError as exc:
        assert "cannot be empty" in str(exc)
        return

    assert False, "Expected ValueError for an empty benign refresh window"


def test_update_benign_prior_rejects_non_finite_embeddings_before_mutating():
    cones = _identity_cones()
    model = CCDModel(
        config=CCDConfig(cone=cones.config),
        encoder=CahoEncoder(EncoderConfig(model_name="sentence-transformers/all-MiniLM-L6-v2")),
        cones=cones,
        benign_prior=np.array([0.5, 0.5], dtype=np.float32),
        malicious_priors={"m": np.array([0.1, 0.9], dtype=np.float32)},
    )
    old_prior = model.benign_prior.copy()

    try:
        model.update_benign_prior(np.array([[1.0, 0.0], [float("nan"), 1.0]], dtype=np.float32))
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-finite benign refresh embeddings")

    assert np.allclose(model.benign_prior, old_prior)


def test_refresh_benign_reference_rejects_non_finite_embeddings_before_mutating():
    cones = _identity_cones()
    model = CCDModel(
        config=CCDConfig(cone=cones.config),
        encoder=CahoEncoder(EncoderConfig(model_name="sentence-transformers/all-MiniLM-L6-v2")),
        cones=cones,
        benign_prior=np.array([0.95, 0.05], dtype=np.float32),
        malicious_priors={"m": np.array([0.1, 0.9], dtype=np.float32)},
        threshold=9.0,
        grouped_thresholds={"tenant-a": {"threshold": 9.0}},
    )
    old_prior = model.benign_prior.copy()
    old_threshold = model.threshold
    old_grouped = model.grouped_thresholds

    try:
        model.refresh_benign_reference(
            np.array([[1.0, 0.0], [float("inf"), 1.0]], dtype=np.float32),
            alpha=0.5,
            calibration_groups=["tenant-a", "tenant-a"],
        )
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-finite benign refresh embeddings")

    assert np.allclose(model.benign_prior, old_prior)
    assert model.threshold == old_threshold
    assert model.grouped_thresholds is old_grouped


def test_model_score_normalization():
    class DummyEncoder:
        def __init__(self):
            self.seen = None

        def encode(self, texts, batch_size=32, normalize=True):
            self.seen = list(texts)
            return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)

    encoder = DummyEncoder()
    cones = _identity_cones()
    benign_prior = np.array([0.9, 0.1], dtype=np.float32)
    malicious_priors = {"m": np.array([0.1, 0.9], dtype=np.float32)}
    model = CCDModel(
        config=CCDConfig(),
        encoder=encoder,
        cones=cones,
        benign_prior=benign_prior,
        malicious_priors=malicious_priors,
    )

    host = "HTTP://WWW.Example.COM/path"
    model.score([host], normalize=True)
    assert encoder.seen == ["www.example.com"]
    model.score([host], normalize=False)
    assert encoder.seen == [host]


def test_score_raises_with_no_malicious_priors():
    cones = _identity_cones()
    benign_prior = np.array([0.5, 0.5], dtype=np.float32)
    model = CCDModel(
        config=CCDConfig(),
        encoder=CahoEncoder(EncoderConfig(model_name="sentence-transformers/all-MiniLM-L6-v2")),
        cones=cones,
        benign_prior=benign_prior,
        malicious_priors={},
    )
    try:
        _ = model.score_embeddings(np.array([[1.0, 0.0]], dtype=np.float32))
    except ValueError:
        return
    assert False, "Expected ValueError when malicious_priors is empty"


def test_ccd_model_from_embeddings_end_to_end():
    config = CCDConfig()
    config.cone = ConeConfig(dim=2, num_cones=2, active_cones=1, temperature=1.0, use_lsh=False)

    benign_emb = np.array([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32)
    malicious_emb = {"cmdi": np.array([[0.0, 1.0], [0.1, 0.9]], dtype=np.float32)}
    axes = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    model = CCDModel.from_embeddings(
        benign_embeddings=benign_emb,
        malicious_embeddings_by_family=malicious_emb,
        config=config,
        axes=axes,
    )
    test_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    scores = model.score_embeddings(test_embeddings)
    assert scores[0] < 0.0
    assert scores[1] > 0.0


def test_score_empty_inputs():
    config = CCDConfig()
    cones = ConePartition.build(config.cone)
    benign_prior = np.ones(cones.config.num_cones, dtype=np.float32) / cones.config.num_cones
    malicious_prior = np.ones(cones.config.num_cones, dtype=np.float32) / cones.config.num_cones
    model = CCDModel(
        config=config,
        encoder=CahoEncoder(config.encoder),
        cones=cones,
        benign_prior=benign_prior,
        malicious_priors={"m": malicious_prior},
    )

    scores = model.score([], batch_size=8, normalize=True)
    assert isinstance(scores, np.ndarray)
    assert scores.size == 0


def test_predict_applies_grouped_thresholds():
    class FixedEncoder:
        def encode(self, texts, batch_size=32, normalize=True):
            return np.array([[0.0, 1.0] for _ in texts], dtype=np.float32)

    cones = _identity_cones()
    model = CCDModel(
        config=CCDConfig(cone=cones.config),
        encoder=FixedEncoder(),
        cones=cones,
        benign_prior=np.array([0.9, 0.1], dtype=np.float32),
        malicious_priors={"m": np.array([0.1, 0.9], dtype=np.float32)},
        threshold=0.0,
        grouped_thresholds={
            "tenant-a": {"threshold": 3.0},
            "tenant-b": {"threshold": 0.1},
        },
    )

    preds = model.predict(
        ["same.example", "same.example"],
        calibration_groups=["tenant-a", "tenant-b"],
    )

    assert preds.tolist() == [False, True]


def test_predict_can_require_grouped_thresholds():
    class FixedEncoder:
        def encode(self, texts, batch_size=32, normalize=True):
            return np.array([[0.0, 1.0] for _ in texts], dtype=np.float32)

    cones = _identity_cones()
    model = CCDModel(
        config=CCDConfig(cone=cones.config),
        encoder=FixedEncoder(),
        cones=cones,
        benign_prior=np.array([0.9, 0.1], dtype=np.float32),
        malicious_priors={"m": np.array([0.1, 0.9], dtype=np.float32)},
        threshold=0.0,
        grouped_thresholds={"tenant-a": {"threshold": 3.0}},
    )

    try:
        model.predict(
            ["same.example"],
            calibration_groups=["tenant-missing"],
            missing_group_threshold="error",
        )
    except KeyError as exc:
        assert "tenant-missing" in str(exc)
    else:
        raise AssertionError("missing grouped threshold should fail when requested")


def test_approximate_k_validation():
    config = CCDConfig()
    cones = ConePartition.build(config.cone)
    benign_prior = np.ones(cones.config.num_cones, dtype=np.float32) / cones.config.num_cones
    malicious_prior = np.ones(cones.config.num_cones, dtype=np.float32) / cones.config.num_cones
    model = CCDModel(
        config=config,
        encoder=CahoEncoder(config.encoder),
        cones=cones,
        benign_prior=benign_prior,
        malicious_priors={"m": malicious_prior},
    )

    try:
        _ = model.score_embeddings(np.array([[1.0, 0.0]], dtype=np.float32), approximate_k=0)
    except ValueError:
        return
    assert False, "Expected ValueError for approximate_k <= 0"


def test_model_certify_uses_exact_score_path():
    class PositiveEncoder:
        def encode(self, texts, batch_size=32, normalize=True):
            return np.array([[0.0, 1.0] for _ in texts], dtype=np.float32)

    cones = _identity_cones()
    model = CCDModel(
        config=CCDConfig(cone=cones.config),
        encoder=PositiveEncoder(),
        cones=cones,
        benign_prior=np.array([0.9, 0.1], dtype=np.float32),
        malicious_priors={"m": np.array([0.1, 0.9], dtype=np.float32)},
        threshold=0.0,
    )

    cert = model.certify(
        "Example.com",
        radius=1,
        edit_model=EditModel(edits=["E5_case"]),
    )

    assert cert.certified is True
    assert cert.prediction is True
    assert cert.method == "enumeration"


def test_model_certify_supports_calibrated_margin_and_combined_fallback():
    class PositiveEncoder:
        def encode(self, texts, batch_size=32, normalize=True):
            return np.array([[0.0, 1.0] for _ in texts], dtype=np.float32)

    cones = _identity_cones()
    model = CCDModel(
        config=CCDConfig(cone=cones.config),
        encoder=PositiveEncoder(),
        cones=cones,
        benign_prior=np.array([0.9, 0.1], dtype=np.float32),
        malicious_priors={"m": np.array([0.1, 0.9], dtype=np.float32)},
        threshold=0.0,
    )

    cmc = model.certify(
        "Example.com",
        radius=1,
        method="calibrated-margin",
        sketch_lipschitz=0.1,
        embedding_rotation_bound=0.1,
    )
    fallback = model.certify(
        "Example.com",
        radius=1,
        method="combined",
        edit_model=EditModel(edits=["E5_case"]),
        sketch_lipschitz=10.0,
        embedding_rotation_bound=10.0,
    )

    assert cmc.certified is True
    assert cmc.method == "calibrated_margin"
    assert fallback.certified is True
    assert fallback.method == "enumeration"


def test_model_certify_requires_bounds_for_calibrated_margin():
    cones = _identity_cones()
    model = CCDModel(
        config=CCDConfig(cone=cones.config),
        encoder=CahoEncoder(EncoderConfig(model_name="sentence-transformers/all-MiniLM-L6-v2")),
        cones=cones,
        benign_prior=np.array([0.9, 0.1], dtype=np.float32),
        malicious_priors={"m": np.array([0.1, 0.9], dtype=np.float32)},
        threshold=0.0,
    )

    try:
        model.certify("example.com", radius=1, method="calibrated-margin")
    except ValueError as exc:
        assert "sketch_lipschitz" in str(exc)
        return

    assert False, "Expected ValueError when calibrated-margin bounds are missing"


def test_model_certify_rejects_negative_radius_before_scoring():
    class FailingEncoder:
        def encode(self, texts, batch_size=32, normalize=True):
            raise AssertionError("encoder should not be called for invalid radius")

    cones = _identity_cones()
    model = CCDModel(
        config=CCDConfig(cone=cones.config),
        encoder=FailingEncoder(),
        cones=cones,
        benign_prior=np.array([0.9, 0.1], dtype=np.float32),
        malicious_priors={"m": np.array([0.1, 0.9], dtype=np.float32)},
        threshold=0.0,
    )

    try:
        model.certify("example.com", radius=-1)
    except ValueError as exc:
        assert "radius" in str(exc)
        return

    assert False, "Expected ValueError for negative certificate radius"


def test_model_certify_rejects_invalid_inputs_before_scoring():
    class FailingEncoder:
        def encode(self, texts, batch_size=32, normalize=True):
            raise AssertionError("encoder should not be called for invalid certificate inputs")

    cones = _identity_cones()
    model = CCDModel(
        config=CCDConfig(cone=cones.config),
        encoder=FailingEncoder(),
        cones=cones,
        benign_prior=np.array([0.9, 0.1], dtype=np.float32),
        malicious_priors={"m": np.array([0.1, 0.9], dtype=np.float32)},
        threshold=0.0,
    )

    invalid_calls = [
        lambda: model.certify("example.com", radius=1, threshold=float("nan")),
        lambda: model.certify("example.com", radius=1, max_nodes=0),
        lambda: model.certify(
            "example.com",
            radius=1,
            method="calibrated-margin",
            sketch_lipschitz=-0.1,
            embedding_rotation_bound=0.1,
        ),
        lambda: model.certify(
            "example.com",
            radius=1,
            method="calibrated-margin",
            sketch_lipschitz=0.1,
            embedding_rotation_bound=float("inf"),
        ),
        lambda: model.certify(
            "example.com",
            radius=1,
            method="calibrated-margin",
            sketch_lipschitz=0.1,
            embedding_rotation_bound=0.1,
            eps=0.0,
        ),
    ]

    for call in invalid_calls:
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("Expected invalid certificate input to fail before scoring")
