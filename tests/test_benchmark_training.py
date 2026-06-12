import importlib.util
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
        seed=17,
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
        seed=17,
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
    assert report["config"]["seed"] == 17
    assert report["contrastive_objective"]["name"] == "supervised_orbit_contrastive_loss"
    assert report["contrastive_objective"]["binary_auxiliary_head_views"] == "both_l2_normalized_views"
    assert report["contrastive_objective"]["deterministic_seed"] == 17
    assert report["train_summary"]["steps"] == 2


def test_binary_trainer_records_validation_fixed_fpr_selection():
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

    trainer = BenchmarkBinaryContrastiveTrainer(
        DummySentenceModel(),
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
        seed=17,
    )

    summary = trainer.fit(
        _TinyViewDataset(),
        epochs=1,
        validation_dataset=_TinyViewDataset(),
        validation_target_fpr=0.5,
        restore_best_validation=True,
    )

    selection = summary["validation_model_selection"]
    assert selection["metric"] == "tpr_at_target_fpr"
    assert selection["target_fpr"] == 0.5
    assert selection["selection_rule"] == "maximize_tpr_at_target_fpr;ties_keep_earliest"
    assert selection["score_source"] == "binary_auxiliary_head_sigmoid"
    assert selection["score_view"] == "canonical_view1_only"
    assert selection["best_epoch"] == 1
    assert selection["restored_best_validation_checkpoint"] is True
    assert selection["history"][0]["status"] == "pass"
    assert selection["history"][0]["threshold_source"] == "validation_benign_scores"
    assert selection["history"][0]["score_source"] == "binary_auxiliary_head_sigmoid"
    assert selection["history"][0]["score_view"] == "canonical_view1_only"
    assert selection["history"][0]["embedding_normalization"] == "l2"
    assert selection["history"][0]["alpha"] == 0.5
    assert selection["history"][0]["num_samples"] == 2
    assert selection["history"][0]["order_statistic_rank"] == 2
    assert selection["history"][0]["decision_rule"] == "score > threshold"
    assert selection["history"][0]["calibration_scores"] == "benign_only"
    assert selection["history"][0]["n_validation_benign"] == 2
    assert selection["history"][0]["n_validation_positive"] == 2


def test_binary_trainer_restores_best_validation_state():
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

    class RecordingTrainer(BenchmarkBinaryContrastiveTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.validation_states = []

        def evaluate_fixed_fpr(self, dataset, *, target_fpr=1e-4):
            del dataset
            self.validation_states.append(
                {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}
            )
            call_index = len(self.validation_states)
            return {
                "status": "pass",
                "target_fpr": float(target_fpr),
                "threshold": 0.5,
                "alpha": float(target_fpr),
                "num_samples": 2,
                "order_statistic_rank": 2,
                "decision_rule": "score > threshold",
                "calibration_scores": "benign_only",
                "threshold_source": "validation_benign_scores",
                "score_source": "binary_auxiliary_head_sigmoid",
                "score_view": "canonical_view1_only",
                "embedding_normalization": "l2",
                "n_validation_rows": 4,
                "n_validation_benign": 2,
                "n_validation_positive": 2,
                "false_positives": 0,
                "true_positives": 2 if call_index == 1 else 0,
                "observed_fpr": 0.0,
                "tpr_at_target_fpr": 1.0 if call_index == 1 else 0.0,
            }

    trainer = RecordingTrainer(
        DummySentenceModel(),
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
        seed=17,
    )

    summary = trainer.fit(
        _TinyViewDataset(),
        epochs=2,
        validation_dataset=_TinyViewDataset(),
        validation_target_fpr=0.5,
        restore_best_validation=True,
    )

    assert summary["validation_model_selection"]["best_epoch"] == 1
    assert summary["validation_model_selection"]["restored_best_validation_checkpoint"] is True
    assert len(trainer.validation_states) == 2
    assert any(
        not torch.allclose(trainer.validation_states[0][key], trainer.validation_states[1][key])
        for key in trainer.validation_states[0]
    )
    restored = trainer.model.state_dict()
    assert all(torch.allclose(restored[key].cpu(), trainer.validation_states[0][key]) for key in restored)


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


def test_binary_trainer_rejects_grad_cache_for_supervised_orbit_objective():
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

    trainer = BenchmarkBinaryContrastiveTrainer(
        DummySentenceModel(),
        batch_size=2,
        temperature=0.1,
        lr=1e-2,
        max_grad_norm=1.0,
        scheduler="none",
        min_lr=0.0,
        weight_decay=0.02,
        use_grad_cache=True,
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

    with pytest.raises(ValueError, match="GradCache is not supported"):
        trainer.fit(_TinyViewDataset(), epochs=1)


def test_benchmark_binary_script_rejects_grad_cache():
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_benchmark_caho_binary.py"
    spec = importlib.util.spec_from_file_location("_test_train_benchmark_caho_binary", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.build_parser().parse_args(["--out", "unused-output", "--grad-cache"])

    with pytest.raises(RuntimeError, match="supervised orbit labels"):
        module.validate_args(args)


def test_benchmark_binary_script_validation_selection_args():
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_benchmark_caho_binary.py"
    spec = importlib.util.spec_from_file_location("_test_train_benchmark_caho_binary_args", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.build_parser().parse_args(
        [
            "--out",
            "unused-output",
            "--validation-root",
            "validation-only",
            "--validation-target-fpr",
            "0.001",
            "--restore-best-validation",
        ]
    )

    assert module.validate_args(args) is args
    assert str(args.validation_root) == "validation-only"
    assert args.validation_target_fpr == 0.001
    assert args.restore_best_validation is True


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
