import numpy as np

from ccd.config import EncoderConfig
from ccd.encoder import CahoEncoder


def test_encode_torch_matches_numpy():
    try:
        import torch
    except Exception:
        return

    class DummyModel:
        def eval(self):
            return None

        def encode(
            self,
            hostnames,
            *,
            batch_size,
            convert_to_numpy=False,
            convert_to_tensor=False,
            normalize_embeddings=False,
            show_progress_bar=False,
        ):
            del batch_size, show_progress_bar
            embeddings = np.asarray(
                [[float(index + 1), float(len(hostname)), 0.25] for index, hostname in enumerate(hostnames)],
                dtype=np.float32,
            )
            if normalize_embeddings:
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            if convert_to_tensor:
                return torch.as_tensor(embeddings)
            if convert_to_numpy:
                return embeddings
            return embeddings

    config = EncoderConfig(model_name="dummy-caho", device="cpu", fp16=False)
    encoder = CahoEncoder(config)
    encoder._model = DummyModel()

    hostnames = ["example.com", "login.microsoftonline.com"]
    emb_np = encoder.encode(hostnames, batch_size=2, normalize=True)
    emb_t = encoder.encode_torch(hostnames, batch_size=2, normalize=True)

    emb_t_np = emb_t.detach().cpu().numpy()
    assert emb_np.shape == emb_t_np.shape
    norms = np.linalg.norm(emb_t_np, axis=1)
    np.testing.assert_allclose(norms, np.ones_like(norms), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(emb_np, emb_t_np, rtol=1e-4, atol=1e-4)
