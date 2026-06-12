import numpy as np

from ccd.config import ConeConfig
from ccd.cone import ConePartition
from ccd.scoring import ccd_score


def test_ccd_score_runs():
    config = ConeConfig(dim=4, num_cones=16, active_cones=4, use_lsh=False)
    cones = ConePartition.build(config)
    benign_prior = np.full(config.num_cones, 1.0 / config.num_cones, dtype=np.float32)
    malicious_prior = np.full(config.num_cones, 1.0 / config.num_cones, dtype=np.float32)
    u = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    score = ccd_score(u, cones, benign_prior, {"fam": malicious_prior})
    assert abs(score) < 1e-6


def test_ccd_score_uses_logsumexp_mixture():
    axes = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    config = ConeConfig(dim=2, num_cones=2, active_cones=1, temperature=1.0, use_lsh=False)
    cones = ConePartition.build(config, axes=axes)
    benign_prior = np.array([0.8, 0.2], dtype=np.float32)
    malicious_priors = {
        "weak": np.array([0.7, 0.3], dtype=np.float32),
        "strong": np.array([0.1, 0.9], dtype=np.float32),
    }
    u = np.array([0.0, 1.0], dtype=np.float32)

    score = ccd_score(
        u,
        cones,
        benign_prior,
        malicious_priors,
        effective_count=2.0,
        mixture_weights={"weak": 0.25, "strong": 0.75},
    )

    hb = -np.log(0.2)
    h_weak = -np.log(0.3)
    h_strong = -np.log(0.9)
    terms = np.array([
        np.log(0.25) + 2.0 * (hb - h_weak),
        np.log(0.75) + 2.0 * (hb - h_strong),
    ])
    expected = terms.max() + np.log(np.exp(terms - terms.max()).sum())
    assert np.isclose(score, expected)


def test_default_score_bypasses_lsh_candidates():
    axes = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    config = ConeConfig(dim=2, num_cones=2, active_cones=1, temperature=1.0, use_lsh=True)
    cones = ConePartition.build(config, axes=axes)

    class WrongLSH:
        def query_candidates(self, _u, probe_radius=1):
            return [0]

    cones.lsh = WrongLSH()
    benign_prior = np.array([0.9, 0.1], dtype=np.float32)
    malicious_prior = np.array([0.1, 0.9], dtype=np.float32)
    u = np.array([0.0, 1.0], dtype=np.float32)

    exact_idx, _weights = cones.cone_sketch(u)
    lsh_idx, _weights = cones.cone_sketch(u, use_lsh=True)
    score = ccd_score(u, cones, benign_prior, {"fam": malicious_prior})

    assert exact_idx.tolist() == [1]
    assert lsh_idx.tolist() == [0]
    assert score > 0.0
