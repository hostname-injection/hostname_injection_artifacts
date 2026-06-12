import numpy as np

from ccd.config import CCDConfig, ConeConfig
from ccd.cone import ConePartition
from ccd.encoder import CahoEncoder
from ccd.model import CCDModel


def test_torch_scoring_matches_numpy_cpu():
    try:
        import torch
    except Exception:
        return

    cone = ConeConfig(dim=4, num_cones=16, active_cones=4, use_lsh=False, seed=9)
    config = CCDConfig(cone=cone)
    cones = ConePartition.build(cone)

    rng = np.random.default_rng(9)
    embeddings = rng.standard_normal((5, 4)).astype(np.float32)
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

    scores_np = model.score_embeddings(embeddings)
    scores_t = model.score_embeddings(torch.tensor(embeddings))

    np.testing.assert_allclose(scores_np, scores_t, rtol=1e-5, atol=1e-5)
