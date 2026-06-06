import sys
from types import SimpleNamespace

import random

import pytest

from ccd.augment import AugmentConfig, CAHOAugmenter, WeightedAugmentConfig
from ccd.train import (
    CAHO_94GB_GRAD_CACHE_BATCH_SIZE,
    CAHODataset,
    CAHO_DEFAULT_LR,
    CAHO_DEFAULT_WEIGHT_DECAY,
    CAHO_TRAINING_SETTING_FIELDS,
    ContrastiveTrainer,
    Sample,
    binary_labels_from_families,
    caho_training_default_deviations,
    pairwise_contrastive_loss,
    orbit_labels_from_families,
    split_input_fn,
    supervised_orbit_contrastive_loss,
    warn_if_caho_training_defaults_changed,
)


def _make_weighted_augmenter() -> CAHOAugmenter:
    weighted = WeightedAugmentConfig(
        num_augs=1,
        benign_weights={"toggle_protocol": 1.0},
        malicious_weights={"toggle_protocol": 1.0},
        retry_on_no_change=True,
        max_attempts=3,
    )
    config = AugmentConfig(
        normalize_input=False,
        use_edit_model=False,
        use_weighted_augs=True,
        weighted=weighted,
    )
    return CAHOAugmenter(config=config)


def test_split_input_fn_chunks():
    chunks = split_input_fn(["a", "b", "c", "d", "e"], chunk_size=2)
    assert chunks == [{"view": ["a", "b"]}, {"view": ["c", "d"]}, {"view": ["e"]}]


def test_weighted_augmenter_changes_hostname():
    augmenter = _make_weighted_augmenter()
    hostname = "example.com"
    out = augmenter.augment(hostname, is_malicious=False, rng=random.Random(0))
    assert out != hostname


def test_caho_dataset_include_original():
    augmenter = _make_weighted_augmenter()
    dataset = CAHODataset(
        [Sample("example.com", is_malicious=False)],
        augmenter=augmenter,
        include_original=True,
    )
    view1, view2, label = dataset[0]
    assert view1 == "example.com"
    assert view2 != "example.com"
    assert label is None


def test_caho_dataset_seed_replays_augmented_views():
    class RandomSuffixAugmenter:
        def augment(self, hostname, is_malicious=False, rng=None):
            return f"{hostname}:{rng.random():.8f}"

    samples = [
        Sample("example.com", is_malicious=False),
        Sample("evil.example", is_malicious=True, family="cmd"),
    ]
    first = CAHODataset(samples, augmenter=RandomSuffixAugmenter(), include_original=False, seed=7)
    second = CAHODataset(samples, augmenter=RandomSuffixAugmenter(), include_original=False, seed=7)

    assert [first[i] for i in range(len(first))] == [second[i] for i in range(len(second))]
    first.set_epoch(1)
    assert first[0] != second[0]


def test_orbit_labels_keep_benign_unique_and_group_positive_families():
    labels = orbit_labels_from_families([None, "cmd", "cmd", "sql", None])

    assert labels[1] == labels[2]
    assert labels[1] != labels[3]
    assert labels[0] not in {labels[1], labels[3], labels[4]}
    assert labels[4] not in {labels[1], labels[3]}


def test_binary_labels_map_family_presence_to_auxiliary_targets():
    assert binary_labels_from_families([None, "cmd", "sql", None]) == [0, 1, 1, 0]


def test_caho_trainers_default_to_paper_optimizer_recipe():
    contrastive_trainer = ContrastiveTrainer(object())

    assert contrastive_trainer.lr == CAHO_DEFAULT_LR == 1e-4
    assert contrastive_trainer.weight_decay == CAHO_DEFAULT_WEIGHT_DECAY


def test_caho_trainers_reject_invalid_optimizer_hyperparameters():
    with pytest.raises(ValueError, match="lr"):
        ContrastiveTrainer(object(), lr=float("nan"))
    with pytest.raises(ValueError, match="weight_decay"):
        ContrastiveTrainer(object(), weight_decay=-0.01)
    with pytest.raises(ValueError, match="binary_loss_weight"):
        ContrastiveTrainer(object(), binary_loss_weight=0.0)
    with pytest.raises(ValueError, match="contrastive_loss_weight"):
        ContrastiveTrainer(object(), contrastive_loss_weight=0.0)
    with pytest.raises(ValueError, match="binary_hidden_dim"):
        ContrastiveTrainer(object(), binary_hidden_dim=0)


def test_caho_training_default_deviation_warning(capsys):
    args = SimpleNamespace(lr=2e-4, epochs=20)
    defaults = {"lr": CAHO_DEFAULT_LR, "epochs": 20}

    deviations = warn_if_caho_training_defaults_changed(
        args,
        defaults=defaults,
        fields=("lr", "epochs"),
        label="unit-test CAHO",
    )

    captured = capsys.readouterr()
    assert deviations == ["--lr=0.0002 (default 0.0001)"]
    assert "WARNING: unit-test CAHO settings differ" in captured.err
    assert "Results should be expected to differ" in captured.err


def test_caho_training_default_deviation_fields_cover_core_training_args():
    assert {
        "epochs",
        "batch_size",
        "lr",
        "weight_decay",
        "binary_loss_weight",
        "contrastive_loss_weight",
        "binary_hidden_dim",
        "seed",
    }.issubset(
        set(CAHO_TRAINING_SETTING_FIELDS)
    )
    assert caho_training_default_deviations(
        SimpleNamespace(epochs=20, lr=CAHO_DEFAULT_LR),
        {"epochs": 20, "lr": CAHO_DEFAULT_LR},
        ("epochs", "lr"),
    ) == []
    assert caho_training_default_deviations(
        SimpleNamespace(batch_size=CAHO_94GB_GRAD_CACHE_BATCH_SIZE, grad_cache=True),
        {"batch_size": None, "grad_cache": True},
        ("batch_size",),
    ) == []


def test_pairwise_contrastive_loss_aligned():
    torch = pytest.importorskip("torch")
    embeddings = torch.eye(2)
    loss = pairwise_contrastive_loss(embeddings, embeddings, temperature=0.1)
    assert loss.item() < 0.01


def test_pairwise_contrastive_loss_misaligned():
    torch = pytest.importorskip("torch")
    embeddings1 = torch.eye(2)
    embeddings2 = torch.flip(embeddings1, dims=[0])
    loss = pairwise_contrastive_loss(embeddings1, embeddings2, temperature=0.1)
    assert loss.item() > 1.0


def test_supervised_orbit_contrastive_loss_backpropagates():
    torch = pytest.importorskip("torch")
    embeddings1 = torch.eye(3, requires_grad=True)
    embeddings2 = torch.eye(3).detach().clone().requires_grad_(True)
    labels = torch.tensor([1, 1, -1], dtype=torch.long)

    loss = supervised_orbit_contrastive_loss(embeddings1, embeddings2, labels, temperature=0.1)
    loss.backward()

    assert loss.item() >= 0.0
    assert embeddings1.grad is not None
    assert embeddings2.grad is not None


def test_contrastive_trainer_training_loss_includes_binary_auxiliary_head():
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(2, 2)
    trainer = ContrastiveTrainer(model, binary_hidden_dim=3)
    trainer._binary_classifier = trainer._build_binary_classifier(2)

    embeddings1 = torch.eye(2, requires_grad=True)
    embeddings2 = torch.eye(2).detach().clone().requires_grad_(True)
    loss = trainer._training_loss(
        embeddings1,
        embeddings2,
        labels=[-1, 0],
        binary_labels=[0, 1],
    )
    loss.backward()

    head_grads = [param.grad for param in trainer._binary_classifier.parameters()]
    assert loss.item() > 0.0
    assert embeddings1.grad is not None
    assert embeddings2.grad is not None
    assert any(grad is not None for grad in head_grads)


def test_contrastive_trainer_grad_cache_receives_orbit_labels(monkeypatch):
    torch = pytest.importorskip("torch")
    captured = {}

    class FakeGradCache:
        def __init__(self, **kwargs):
            captured["loss_fn"] = kwargs["loss_fn"]

        def __call__(self, *model_inputs, **loss_kwargs):
            captured["model_inputs"] = model_inputs
            captured["labels"] = list(loss_kwargs["labels"])
            captured["binary_labels"] = list(loss_kwargs["binary_labels"])
            return torch.tensor(0.0)

    monkeypatch.setitem(sys.modules, "grad_cache", SimpleNamespace(GradCache=FakeGradCache))

    model = torch.nn.Linear(1, 1)
    dataset = [
        ("safe.example", "www.safe.example", None),
        ("evil.example", "www.evil.example", "cmd"),
        ("evil2.example", "www.evil2.example", "cmd"),
    ]
    trainer = ContrastiveTrainer(
        model,
        batch_size=3,
        temperature=0.1,
        lr=1e-2,
        scheduler="none",
        min_lr=0.0,
        use_grad_cache=True,
        grad_cache_chunk_size=2,
        num_workers=0,
        loss_mode="fixed",
    )

    trainer.fit(dataset, epochs=1)

    assert captured["model_inputs"][0] == ["safe.example", "evil.example", "evil2.example"]
    assert captured["labels"][1] == captured["labels"][2]
    assert captured["labels"][0] != captured["labels"][1]
    assert captured["binary_labels"] == [0, 1, 1]
