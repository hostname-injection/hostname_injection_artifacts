import numpy as np

from ccd.config import CCDConfig, ConeConfig, EncoderConfig
from ccd.cone import ConePartition
from ccd.encoder import CahoEncoder
from ccd.io import ModelBundle
from ccd.model import CCDModel


def _bundle_with_seed(seed: int = 0) -> ModelBundle:
    cone = ConeConfig(dim=4, num_cones=16, active_cones=4, use_lsh=False, seed=seed)
    config = CCDConfig(encoder=EncoderConfig(model_name="sentence-transformers/all-MiniLM-L6-v2"), cone=cone)
    cones = ConePartition.build(cone)

    rng = np.random.default_rng(seed)
    benign_prior = rng.random(16).astype(np.float32) + 1e-6
    benign_prior /= benign_prior.sum()
    malicious_prior = rng.random(16).astype(np.float32) + 1e-6
    malicious_prior /= malicious_prior.sum()

    return ModelBundle(
        axes=cones.axes,
        benign_prior=benign_prior,
        malicious_priors={"m": malicious_prior},
        config=config,
    )


def test_model_score_with_approximate_k():
    bundle = _bundle_with_seed(1)
    model = CCDModel(
        config=bundle.config,
        encoder=CahoEncoder(bundle.config.encoder),
        cones=ConePartition.build(bundle.config.cone, axes=bundle.axes),
        benign_prior=bundle.benign_prior,
        malicious_priors=bundle.malicious_priors,
    )

    rng = np.random.default_rng(1)
    embeddings = rng.standard_normal((5, 4)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    exact = model.score_embeddings(embeddings)
    approx = model.score_embeddings(embeddings, approximate_k=2)

    assert exact.shape == approx.shape
