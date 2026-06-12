import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ccd.benchmark_training import (
    BenchmarkBinaryContrastiveTrainer,
    BenchmarkChunkShuffleSampler,
    BenchmarkTrainingConfig,
    _orbit_labels,
)


class _TinyViewDataset:
    def __init__(self):
        self.rows = [
            ("safe.example.com", "www.safe.example.com", 0),
            ("status.example.net", "status-example.net", 0),
            ("evil.$(id).example", "evil.%24%28id%29.example", 1),
            ("probe'--.example", "probe--.example", 1),
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def _training_config(out: Path) -> BenchmarkTrainingConfig:
    return BenchmarkTrainingConfig(
        root="fixture",
        model="dummy",
        out=str(out),
        epochs=1,
        batch_size=2,
        lr=1e-2,
        temperature=0.1,
        max_grad_norm=1.0,
        scheduler="none",
        min_lr=0.0,
        grad_cache=False,
        grad_cache_chunk_size=2,
        num_workers=0,
        device="cpu",
        normalize_text=False,
        augmenter="hybrid",
        weighted_num_augs=2,
        weighted_max_attempts=3,
        weighted_retry_on_no_change=True,
        contrastive_loss="fixed",
        contrastive_max_scale=100.0,
        contrastive_min_scale=1.0,
        optimize_contrastive_scale=False,
        binary_loss_weight=0.5,
        contrastive_loss_weight=0.5,
        binary_hidden_dim=4,
        weight_decay=0.02,
        log_every=0,
        max_steps=None,
        checkpoint_every_steps=0,
        checkpoint_dir=None,
    )


def test_benchmark_binary_contrastive_trainer_fit_and_save(tmp_path):
    torch = pytest.importorskip("torch")

    class DummySentenceModel(torch.nn.Module):
        def __init__(self, dim=4):
            super().__init__()
            self.dim = dim
            self.embed = torch.nn.Embedding(256, dim)
            self.proj = torch.nn.Linear(dim, dim)

        def tokenize(self, texts):
            ids = [sum(text.encode("utf-8")) % 256 for text in texts]
            return {"input_ids": torch.tensor(ids, dtype=torch.long).unsqueeze(1)}

        def forward(self, tokenized):
            emb = self.embed(tokenized["input_ids"]).mean(dim=1)
            return {"sentence_embedding": self.proj(emb)}

        def get_sentence_embedding_dimension(self):
            return self.dim

        def save(self, out):
            path = Path(out)
            path.mkdir(parents=True, exist_ok=True)
            (path / "dummy_model.txt").write_text("saved\n", encoding="utf-8")

    model = DummySentenceModel()
    before = [param.detach().clone() for param in model.parameters()]
    trainer = BenchmarkBinaryContrastiveTrainer(
        model,
        batch_size=2,
        temperature=0.1,
        lr=1e-2,
        max_grad_norm=1.0,
        scheduler="none",
        min_lr=0.0,
        weight_decay=0.02,
        use_grad_cache=False,
        grad_cache_chunk_size=2,
        num_workers=0,
        loss_mode="fixed",
        loss_max_scale=100.0,
        loss_min_scale=1.0,
        optimize_loss=False,
        log_every=0,
        binary_loss_weight=0.5,
        contrastive_loss_weight=0.5,
        binary_hidden_dim=4,
    )

    summary = trainer.fit(_TinyViewDataset(), epochs=1)

    assert summary["steps"] == 2
    assert summary["avg_contrastive_loss"] > 0.0
    assert summary["avg_binary_loss"] > 0.0
    assert trainer.classifier is not None
    assert trainer.weight_decay == 0.02
    after = list(model.parameters())
    assert any(not torch.allclose(old, new) for old, new in zip(before, after))

    out = tmp_path / "checkpoint"
    config = _training_config(out)
    trainer.save(out, config, summary)

    assert (out / "dummy_model.txt").exists()
    assert (out / "binary_classifier.pt").exists()
    report = json.loads((out / "benchmark_training_report.json").read_text(encoding="utf-8"))
    assert report["config"]["binary_loss_weight"] == 0.5
    assert report["config"]["weight_decay"] == 0.02
    assert report["contrastive_objective"]["name"] == "supervised_orbit_contrastive_loss"
    assert report["contrastive_objective"]["binary_auxiliary_head_views"] == "both_l2_normalized_views"
    assert report["train_summary"]["steps"] == 2


def test_binary_auxiliary_head_trains_on_both_views():
    torch = pytest.importorskip("torch")

    class DummySentenceModel(torch.nn.Module):
        def __init__(self, dim=4):
            super().__init__()
            self.dim = dim
            self.embed = torch.nn.Embedding(256, dim)
            self.proj = torch.nn.Linear(dim, dim)

        def tokenize(self, texts):
            ids = [sum(text.encode("utf-8")) % 256 for text in texts]
            return {"input_ids": torch.tensor(ids, dtype=torch.long).unsqueeze(1)}

        def forward(self, tokenized):
            emb = self.embed(tokenized["input_ids"]).mean(dim=1)
            return {"sentence_embedding": self.proj(emb)}

        def get_sentence_embedding_dimension(self):
            return self.dim

    class RecordingClassifier(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.linear = torch.nn.Linear(dim, 1)
            self.batch_sizes = []

        def forward(self, embeddings):
            self.batch_sizes.append(int(embeddings.shape[0]))
            return self.linear(embeddings)

    class RecordingTrainer(BenchmarkBinaryContrastiveTrainer):
        def _build_classifier(self, embedding_dim: int):
            self.recording_classifier = RecordingClassifier(embedding_dim)
            return self.recording_classifier

    trainer = RecordingTrainer(
        DummySentenceModel(),
        batch_size=4,
        temperature=0.1,
        lr=1e-2,
        max_grad_norm=1.0,
        scheduler="none",
        min_lr=0.0,
        weight_decay=0.02,
        use_grad_cache=False,
        grad_cache_chunk_size=2,
        num_workers=0,
        loss_mode="fixed",
        loss_max_scale=100.0,
        loss_min_scale=1.0,
        optimize_loss=False,
        log_every=0,
        binary_loss_weight=0.5,
        contrastive_loss_weight=0.5,
        binary_hidden_dim=4,
    )

    summary = trainer.fit(_TinyViewDataset(), epochs=1)

    assert summary["steps"] == 1
    assert trainer.recording_classifier.batch_sizes == [8]


def test_chunk_shuffle_sampler_preserves_all_indices_and_chunk_locality():
    dataset = SimpleNamespace(
        base=SimpleNamespace(
            _selected=None,
            _chunks=[
                ("dns_hostnames", Path("chunk-a.csv"), 2),
                ("dns_hostnames", Path("chunk-b.csv"), 3),
                ("user_logins", Path("chunk-c.csv"), 1),
            ],
        ),
        __len__=lambda self: 6,
    )

    class DatasetProxy:
        base = dataset.base

        def __len__(self):
            return 6

    sampler = BenchmarkChunkShuffleSampler(DatasetProxy(), seed=0)
    indices = list(iter(sampler))

    assert sorted(indices) == list(range(6))
    assert {0, 1} in [set(indices[0:2]), set(indices[1:3]), set(indices[3:5]), set(indices[4:6])]


def test_orbit_labels_share_positive_and_keep_benign_unique():
    torch = pytest.importorskip("torch")
    labels = torch.tensor([0, 1, 0, 1, -1], dtype=torch.long)

    orbit = _orbit_labels(labels)

    assert orbit.tolist() == [-1, 1, -2, 1, -3]
