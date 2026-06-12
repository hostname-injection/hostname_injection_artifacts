import numpy as np

from ccd.config import CCDConfig, ConeConfig
from ccd.cone import ConePartition
from ccd.model import CCDModel


class DummyEncoder:
    def encode(self, texts, batch_size=32, normalize=True):
        return np.zeros((len(texts), 2), dtype=np.float32)


def _simple_model():
    cone = ConeConfig(dim=2, num_cones=2, active_cones=1, temperature=1.0, use_lsh=False)
    config = CCDConfig(cone=cone)
    axes = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    cones = ConePartition.build(cone, axes=axes)
    benign_prior = np.array([0.8, 0.2], dtype=np.float32)
    malicious_priors = {"m": np.array([0.2, 0.8], dtype=np.float32)}
    return CCDModel(
        config=config,
        encoder=DummyEncoder(),
        cones=cones,
        benign_prior=benign_prior,
        malicious_priors=malicious_priors,
    )


def _multi_family_model():
    cone = ConeConfig(dim=2, num_cones=2, active_cones=1, temperature=1.0, use_lsh=False)
    config = CCDConfig(cone=cone)
    axes = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    cones = ConePartition.build(cone, axes=axes)
    benign_prior = np.array([0.8, 0.2], dtype=np.float32)
    malicious_priors = {
        "fam-low": np.array([0.2, 0.8], dtype=np.float32),
        "fam-high": np.array([0.9, 0.1], dtype=np.float32),
    }
    return CCDModel(
        config=config,
        encoder=DummyEncoder(),
        cones=cones,
        benign_prior=benign_prior,
        malicious_priors=malicious_priors,
    )


def test_explain_embeddings_top_cones():
    model = _simple_model()
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    explanations = model.explain_embeddings(
        embeddings,
        hostnames=["a.example", "b.example"],
        top_k=1,
    )

    assert len(explanations) == 2
    assert explanations[0]["top_cones"][0]["cone"] == 0
    assert explanations[1]["top_cones"][0]["cone"] == 1
    assert explanations[0]["score"] < 0.0
    assert explanations[1]["score"] > 0.0


def test_explain_embeddings_reports_family_log_ratio_evidence():
    model = _multi_family_model()
    explanations = model.explain_embeddings(
        np.array([[1.0, 0.0]], dtype=np.float32),
        hostnames=["a.example"],
        top_k=1,
    )

    cone = explanations[0]["top_cones"][0]
    assert cone["best_malicious_family"] == "fam-high"
    assert cone["min_malicious_family"] == "fam-low"
    assert cone["log_malicious_over_benign"]["fam-high"] > 0.0
    assert cone["log_malicious_over_benign"]["fam-low"] < 0.0
    assert cone["best_log_malicious_over_benign"] == cone["log_malicious_over_benign"]["fam-high"]
    assert cone["min_log_malicious_over_benign"] == cone["log_malicious_over_benign"]["fam-low"]


def test_explain_embeddings_reports_normalized_cosine_similarity():
    model = _simple_model()
    explanations = model.explain_embeddings(
        np.array([[0.0, 3.0]], dtype=np.float32),
        hostnames=["scaled.example"],
        top_k=1,
    )
    unit_explanations = model.explain_embeddings(
        np.array([[0.0, 1.0]], dtype=np.float32),
        hostnames=["unit.example"],
        top_k=1,
    )

    assert explanations[0]["top_cones"][0]["cone"] == 1
    assert np.isclose(explanations[0]["top_cones"][0]["similarity"], 1.0)
    assert np.isclose(explanations[0]["score"], unit_explanations[0]["score"])


def test_explain_uses_grouped_thresholds():
    model = _simple_model()
    model.threshold = 0.0
    model.grouped_thresholds = {
        "tenant-a": {"threshold": 2.0},
        "tenant-b": {"threshold": 0.1},
    }

    explanations = model.explain_embeddings(
        np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32),
        hostnames=["a.example", "b.example"],
        thresholds=[
            model.threshold_for_group("tenant-a", missing="error"),
            model.threshold_for_group("tenant-b", missing="error"),
        ],
        calibration_groups=["tenant-a", "tenant-b"],
        top_k=1,
    )

    assert explanations[0]["calibration_group"] == "tenant-a"
    assert explanations[0]["threshold"] == 2.0
    assert explanations[0]["prediction"] == 0
    assert explanations[1]["calibration_group"] == "tenant-b"
    assert explanations[1]["threshold"] == 0.1
    assert explanations[1]["prediction"] == 1


def test_explain_embeddings_requires_positive_top_k():
    model = _simple_model()
    try:
        _ = model.explain_embeddings(np.array([[1.0, 0.0]], dtype=np.float32), top_k=0)
    except ValueError:
        return
    assert False, "Expected ValueError for top_k <= 0"


def test_explain_empty_inputs():
    model = _simple_model()
    assert model.explain([]) == []
