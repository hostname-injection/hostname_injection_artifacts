from pathlib import Path

import numpy as np

import ccd.cli as cli_module


def test_eval_caho_checkpoint_npz(tmp_path: Path, monkeypatch):
    class DummySentenceTransformer:
        def __init__(self, model_name):
            self.model_name = model_name

        def to(self, device):
            return self

        def eval(self):
            return None

        def encode(
            self,
            hostnames,
            *,
            batch_size,
            convert_to_numpy,
            normalize_embeddings,
            show_progress_bar,
        ):
            del batch_size, convert_to_numpy, show_progress_bar
            embeddings = np.asarray(
                [[float(index + 1), float(len(hostname)), 1.0, 0.5] for index, hostname in enumerate(hostnames)],
                dtype=np.float32,
            )
            if normalize_embeddings:
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            return embeddings

    monkeypatch.setattr(cli_module, "SentenceTransformer", DummySentenceTransformer)
    input_path = tmp_path / "hosts.txt"
    input_path.write_text("example.com\nmalicious.example.com\n")
    output_path = tmp_path / "embeddings.npz"

    args = type(
        "Args",
        (),
        {
            "model": "dummy-caho",
            "input": input_path,
            "output": output_path,
            "format": "npz",
            "batch_size": 2,
            "device": "cpu",
            "normalize": False,
            "embed_normalize": True,
        },
    )()

    cli_module.cmd_eval_caho(args)
    data = np.load(output_path, allow_pickle=True)
    embeddings = data["embeddings"]
    hostnames = data["hostnames"].tolist()
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] == 4
    assert hostnames == ["example.com", "malicious.example.com"]
