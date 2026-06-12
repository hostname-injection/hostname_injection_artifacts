import numpy as np

from ccd.cone import ConePartition
from ccd.config import CCDConfig, ConeConfig, EncoderConfig
from ccd.encoder import CahoEncoder
from ccd.model import CCDModel


def test_fast_mode_matches_exact_when_k1():
    cone = ConeConfig(dim=4, num_cones=16, active_cones=1, use_lsh=False, seed=7)
    config = CCDConfig(encoder=EncoderConfig(model_name="sentence-transformers/all-MiniLM-L6-v2"), cone=cone)
    cones = ConePartition.build(cone)

    rng = np.random.default_rng(0)
    embeddings = rng.standard_normal((6, 4)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    benign_prior = rng.random(16).astype(np.float32) + 1e-6
    benign_prior /= benign_prior.sum()
    malicious_prior = rng.random(16).astype(np.float32) + 1e-6
    malicious_prior /= malicious_prior.sum()

    model = CCDModel(
        config=config,
        encoder=CahoEncoder(config.encoder),
        cones=cones,
        benign_prior=benign_prior,
        malicious_priors={"m": malicious_prior},
    )

    exact = model.score_embeddings(embeddings, approximate=False)
    fast = model.score_embeddings(embeddings, approximate_k=1)

    np.testing.assert_allclose(exact, fast, rtol=1e-5, atol=1e-5)


def test_approximate_k_matches_exact_when_active_cones():
    cone = ConeConfig(dim=4, num_cones=16, active_cones=3, use_lsh=False, seed=11)
    config = CCDConfig(encoder=EncoderConfig(model_name="sentence-transformers/all-MiniLM-L6-v2"), cone=cone)
    cones = ConePartition.build(cone)

    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((4, 4)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    benign_prior = rng.random(16).astype(np.float32) + 1e-6
    benign_prior /= benign_prior.sum()
    malicious_prior = rng.random(16).astype(np.float32) + 1e-6
    malicious_prior /= malicious_prior.sum()

    model = CCDModel(
        config=config,
        encoder=CahoEncoder(config.encoder),
        cones=cones,
        benign_prior=benign_prior,
        malicious_priors={"m": malicious_prior},
    )

    exact = model.score_embeddings(embeddings, approximate=False)
    approx = model.score_embeddings(embeddings, approximate_k=cone.active_cones)

    np.testing.assert_allclose(exact, approx, rtol=1e-5, atol=1e-5)


def test_topk_and_fast_paths_normalize_nonunit_embeddings():
    cone = ConeConfig(dim=4, num_cones=16, active_cones=3, use_lsh=False, seed=17)
    config = CCDConfig(encoder=EncoderConfig(model_name="sentence-transformers/all-MiniLM-L6-v2"), cone=cone)
    cones = ConePartition.build(cone)

    rng = np.random.default_rng(17)
    embeddings = rng.standard_normal((4, 4)).astype(np.float32)
    unit_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    scaled_embeddings = unit_embeddings * np.array([[0.25], [2.0], [4.0], [0.5]], dtype=np.float32)

    benign_prior = rng.random(16).astype(np.float32) + 1e-6
    benign_prior /= benign_prior.sum()
    malicious_prior = rng.random(16).astype(np.float32) + 1e-6
    malicious_prior /= malicious_prior.sum()

    model = CCDModel(
        config=config,
        encoder=CahoEncoder(config.encoder),
        cones=cones,
        benign_prior=benign_prior,
        malicious_priors={"m": malicious_prior},
    )

    exact_unit = model.score_embeddings(unit_embeddings, approximate=False)
    exact_scaled = model.score_embeddings(scaled_embeddings, approximate=False)
    topk_scaled = model.score_embeddings(scaled_embeddings, approximate_k=cone.active_cones)
    fast_scaled = model.score_embeddings(scaled_embeddings, approximate_k=1)
    fast_unit = model.score_embeddings(unit_embeddings, approximate_k=1)

    np.testing.assert_allclose(exact_unit, exact_scaled, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(exact_unit, topk_scaled, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(fast_unit, fast_scaled, rtol=1e-5, atol=1e-5)


def test_approximate_k_with_lsh_is_explicit_fast_path():
    cone = ConeConfig(dim=2, num_cones=3, active_cones=2, temperature=1.0, use_lsh=True, seed=5)
    config = CCDConfig(encoder=EncoderConfig(model_name="sentence-transformers/all-MiniLM-L6-v2"), cone=cone)
    axes = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    cones = ConePartition.build(cone, axes=axes)

    class WrongLSH:
        def query_candidates(self, _u, probe_radius=1):
            return [0, 2]

    cones.lsh = WrongLSH()
    embeddings = np.array([[0.0, 1.0]], dtype=np.float32)
    benign_prior = np.array([0.45, 0.1, 0.45], dtype=np.float32)
    malicious_prior = np.array([0.1, 0.8, 0.1], dtype=np.float32)

    model = CCDModel(
        config=config,
        encoder=CahoEncoder(config.encoder),
        cones=cones,
        benign_prior=benign_prior,
        malicious_priors={"m": malicious_prior},
    )

    exact = model.score_embeddings(embeddings, approximate=False)
    approx = model.score_embeddings(embeddings, approximate_k=cone.active_cones)

    assert exact.shape == approx.shape
    assert exact[0] > approx[0]
