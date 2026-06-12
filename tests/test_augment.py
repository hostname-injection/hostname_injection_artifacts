import random

from ccd import augment as augment_module
from ccd.augment import (
    AUG_FUNCTIONS,
    AUG_HOMOGLYPHS,
    AugmentConfig,
    CAHOAugmenter,
    DEFAULT_WEIGHTED_BENIGN_WEIGHTS,
    DEFAULT_WEIGHTED_MALICIOUS_WEIGHTS,
    WeightedAugmentConfig,
    _apply_weighted_augmentations,
)
from ccd.edit_model import EditModel


def _random_hostname(rng: random.Random, min_labels: int = 2, max_labels: int = 4) -> str:
    labels = []
    num_labels = rng.randint(min_labels, max_labels)
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    for _ in range(num_labels):
        length = rng.randint(1, 8)
        label = "".join(rng.choice(alphabet) for _ in range(length))
        labels.append(label)
    return ".".join(labels)


def test_augment_noop_when_disabled():
    cfg = AugmentConfig(
        normalize_input=False,
        use_edit_model=False,
        use_weighted_augs=False,
    )
    augmenter = CAHOAugmenter(config=cfg)
    hostname = "Example.COM"
    out = augmenter.augment(hostname, is_malicious=False, rng=random.Random(0))
    assert out == hostname


def test_augment_normalizes_when_enabled():
    cfg = AugmentConfig(
        normalize_input=True,
        use_edit_model=False,
        use_weighted_augs=False,
    )
    augmenter = CAHOAugmenter(config=cfg)
    hostname = "HTTP://WWW.Example.COM/path"
    out = augmenter.augment(hostname, is_malicious=False, rng=random.Random(0))
    assert out == "www.example.com"


def test_weighted_defaults_have_matching_functions():
    benign_keys = set(DEFAULT_WEIGHTED_BENIGN_WEIGHTS.keys())
    malicious_keys = set(DEFAULT_WEIGHTED_MALICIOUS_WEIGHTS.keys())
    fn_keys = set(AUG_FUNCTIONS.keys())
    assert benign_keys.issubset(fn_keys)
    assert malicious_keys.issubset(fn_keys)
    assert "toggle_protocol" not in benign_keys
    assert "toggle_protocol" in malicious_keys
    assert benign_keys.issubset(malicious_keys)


def test_weighted_default_weights_positive():
    for value in DEFAULT_WEIGHTED_BENIGN_WEIGHTS.values():
        assert value > 0
    for value in DEFAULT_WEIGHTED_MALICIOUS_WEIGHTS.values():
        assert value > 0


def test_weighted_truncate_subdomain_changes_when_possible():
    weighted = WeightedAugmentConfig(
        num_augs=1,
        benign_weights={"truncate_subdomain": 1.0},
        malicious_weights={"truncate_subdomain": 1.0},
        retry_on_no_change=True,
        max_attempts=2,
    )
    cfg = AugmentConfig(
        normalize_input=False,
        use_edit_model=False,
        use_weighted_augs=True,
        weighted=weighted,
    )
    augmenter = CAHOAugmenter(config=cfg)
    out = augmenter.augment("a.b.com", is_malicious=False, rng=random.Random(0))
    assert out == "b.com"


def test_weighted_retry_no_change_stable_when_uneditable():
    weighted = WeightedAugmentConfig(
        num_augs=1,
        benign_weights={"truncate_subdomain": 1.0},
        malicious_weights={"truncate_subdomain": 1.0},
        retry_on_no_change=True,
        max_attempts=2,
    )
    cfg = AugmentConfig(
        normalize_input=False,
        use_edit_model=False,
        use_weighted_augs=True,
        weighted=weighted,
    )
    augmenter = CAHOAugmenter(config=cfg)
    hostname = "example.com"
    out = augmenter.augment(hostname, is_malicious=False, rng=random.Random(0))
    assert out == hostname


def test_edit_model_applies_when_enabled():
    cfg = AugmentConfig(
        normalize_input=False,
        use_edit_model=True,
        use_weighted_augs=False,
        benign_min_edits=1,
        benign_max_edits=1,
    )
    benign_edits = EditModel(edits=["E3_delimiter"])
    augmenter = CAHOAugmenter(config=cfg, benign_edits=benign_edits)
    hostname = "example.com"
    out = augmenter.augment(hostname, is_malicious=False, rng=random.Random(0))
    assert out != hostname


def test_aug_functions_return_string():
    hostname = "a.b.com"
    for name, func in AUG_FUNCTIONS.items():
        out = func(hostname, random.Random(0))
        assert isinstance(out, str), f"{name} did not return str"
        assert out, f"{name} returned empty string"


def test_random_case_variation_preserves_letters():
    hostname = "Example.COM"
    out = augment_module._random_case_variation(hostname, random.Random(0))
    assert out.casefold() == hostname.casefold()


def test_shuffle_subdomains_preserves_domain_and_labels():
    hostname = "a.b.c.example.com"
    out = augment_module._shuffle_subdomains(hostname, random.Random(1))
    parts = hostname.split(".")
    out_parts = out.split(".")
    assert out_parts[-2:] == parts[-2:]
    assert sorted(out_parts[:-2]) == sorted(parts[:-2])


def test_truncate_subdomain_removes_first_label():
    hostname = "a.b.c"
    out = augment_module._truncate_subdomain(hostname, random.Random(0))
    assert out == "b.c"


def test_dropout_random_char_preserves_dots():
    hostname = "aa.bb.cc"
    out = augment_module._dropout_random_char(hostname, random.Random(0), dropout_prob=0.9)
    assert out.count(".") == hostname.count(".")
    assert len(out) <= len(hostname)


def test_punctuation_replace_rules():
    assert augment_module._punctuation_replace("a-b.com", random.Random(0)) == "ab.com"
    assert augment_module._punctuation_replace("a_b.com", random.Random(0)) == "a-b.com"
    assert augment_module._punctuation_replace("a.b.com", random.Random(0)) == "a-b.com"


def test_letter_swap_typo_preserves_multiset():
    hostname = "ab1c"
    out = augment_module._letter_swap_typo(hostname, random.Random(0))
    assert sorted(out) == sorted(hostname)
    assert len(out) == len(hostname)


def test_typo_swap_preserves_multiset():
    hostname = "abcd"
    out = augment_module._typo_swap(hostname, random.Random(0))
    assert sorted(out) == sorted(hostname)
    assert len(out) == len(hostname)


def test_base64_encode_parts_all_labels():
    hostname = "ab.cd"
    out = augment_module._base64_encode_parts(hostname, random.Random(0), encode_ratio=1.0)
    assert len(out.split(".")) == len(hostname.split("."))
    assert out != hostname


def test_hex_encode_parts_all_labels():
    hostname = "ab.cd"
    out = augment_module._hex_encode_parts(hostname, random.Random(0), encode_ratio=1.0)
    orig_parts = hostname.split(".")
    out_parts = out.split(".")
    assert len(out_parts) == len(orig_parts)
    for original, encoded in zip(orig_parts, out_parts):
        assert len(encoded) == len(original) * 2


def test_url_encode_parts_all_labels():
    hostname = "ab.cd"
    out = augment_module._url_encode_parts(hostname, random.Random(0), encode_ratio=1.0)
    assert len(out.split(".")) == len(hostname.split("."))
    assert "%" in out


def test_toggle_protocol_adds_or_removes_www():
    out = augment_module._toggle_protocol("example.com", random.Random(0))
    host = out.replace("http://", "").replace("https://", "")
    assert host.startswith("www.")

    out = augment_module._toggle_protocol("www.example.com", random.Random(0))
    host = out.replace("http://", "").replace("https://", "")
    assert not host.startswith("www.")


def test_synonym_swap_replaces_known_token():
    hostname = "myapplication.example.com"
    out = augment_module._synonym_swap(hostname, random.Random(0))
    assert "application" not in out
    assert out.endswith(".example.com")


def test_random_homoglyph_substitution_charset():
    hostname = "aaaa"
    out = augment_module._random_homoglyph_substitution(hostname, random.Random(0))
    allowed = {"a", *AUG_HOMOGLYPHS["a"]}
    assert all(ch in allowed for ch in out)


def test_apply_weighted_augmentations_noop_on_zero_augs():
    hostname = "example.com"
    out = _apply_weighted_augmentations(
        hostname,
        weights={"toggle_protocol": 1.0},
        num_augs=0,
        rng=random.Random(0),
        retry_on_no_change=True,
        max_attempts=2,
    )
    assert out == hostname


def test_apply_weighted_augmentations_noop_on_empty_weights():
    hostname = "example.com"
    out = _apply_weighted_augmentations(
        hostname,
        weights={},
        num_augs=2,
        rng=random.Random(0),
        retry_on_no_change=True,
        max_attempts=2,
    )
    assert out == hostname


def test_weighted_augments_randomized_properties():
    rng = random.Random(123)
    weighted = WeightedAugmentConfig(
        num_augs=2,
        benign_weights=DEFAULT_WEIGHTED_BENIGN_WEIGHTS,
        malicious_weights=DEFAULT_WEIGHTED_MALICIOUS_WEIGHTS,
        retry_on_no_change=True,
        max_attempts=3,
    )
    cfg = AugmentConfig(
        normalize_input=False,
        use_edit_model=False,
        use_weighted_augs=True,
        weighted=weighted,
    )
    augmenter = CAHOAugmenter(config=cfg)

    for _ in range(50):
        hostname = _random_hostname(rng, min_labels=2, max_labels=4)
        out = augmenter.augment(hostname, is_malicious=rng.choice([True, False]), rng=rng)
        assert isinstance(out, str)
        assert out
        assert not any(ch.isspace() for ch in out)


def test_augments_deterministic_with_seed():
    hostname = "a.b.com"
    weighted = WeightedAugmentConfig(
        num_augs=2,
        benign_weights=DEFAULT_WEIGHTED_BENIGN_WEIGHTS,
        malicious_weights=DEFAULT_WEIGHTED_MALICIOUS_WEIGHTS,
        retry_on_no_change=True,
        max_attempts=3,
    )
    cfg = AugmentConfig(
        normalize_input=False,
        use_edit_model=False,
        use_weighted_augs=True,
        weighted=weighted,
    )
    augmenter = CAHOAugmenter(config=cfg)

    out1 = augmenter.augment(hostname, is_malicious=True, rng=random.Random(7))
    out2 = augmenter.augment(hostname, is_malicious=True, rng=random.Random(7))
    assert out1 == out2


def test_shuffle_subdomains_randomized_properties():
    rng = random.Random(11)
    for _ in range(30):
        hostname = _random_hostname(rng, min_labels=3, max_labels=5)
        out = augment_module._shuffle_subdomains(hostname, rng)
        assert out.split(".")[-2:] == hostname.split(".")[-2:]
        assert sorted(out.split(".")[:-2]) == sorted(hostname.split(".")[:-2])


def test_truncate_subdomain_randomized_properties():
    rng = random.Random(12)
    for _ in range(30):
        hostname = _random_hostname(rng, min_labels=2, max_labels=5)
        out = augment_module._truncate_subdomain(hostname, rng)
        parts = hostname.split(".")
        out_parts = out.split(".")
        if len(parts) > 2:
            assert len(out_parts) == len(parts) - 1
            assert out_parts == parts[1:]
        else:
            assert out == hostname


def test_dropout_random_char_randomized_properties():
    rng = random.Random(13)
    for _ in range(30):
        hostname = _random_hostname(rng, min_labels=2, max_labels=5)
        out = augment_module._dropout_random_char(hostname, rng, dropout_prob=0.5)
        assert out.count(".") == hostname.count(".")
        assert len(out) <= len(hostname)


def test_encoders_preserve_label_count_randomized():
    rng = random.Random(14)
    for _ in range(30):
        hostname = _random_hostname(rng, min_labels=2, max_labels=5)
        base = hostname.count(".")
        out64 = augment_module._base64_encode_parts(hostname, rng, encode_ratio=0.8)
        outhex = augment_module._hex_encode_parts(hostname, rng, encode_ratio=0.8)
        outurl = augment_module._url_encode_parts(hostname, rng, encode_ratio=0.8)
        assert out64.count(".") == base
        assert outhex.count(".") == base
        assert outurl.count(".") == base


def test_typo_swaps_preserve_multiset_randomized():
    rng = random.Random(15)
    for _ in range(30):
        hostname = _random_hostname(rng, min_labels=2, max_labels=4)
        out1 = augment_module._typo_swap(hostname, rng)
        out2 = augment_module._letter_swap_typo(hostname, rng)
        assert sorted(out1) == sorted(hostname)
        assert sorted(out2) == sorted(hostname)


def test_punctuation_replace_randomized_properties():
    rng = random.Random(16)
    cases = ["a-b.com", "a_b.com", "a.b.com"]
    for hostname in cases:
        out = augment_module._punctuation_replace(hostname, rng)
        assert out
        assert len(out) <= len(hostname)


def test_edit_model_randomized_properties():
    rng = random.Random(17)
    benign_edits = EditModel(edits=["E3_delimiter", "E4_label_split", "E5_case"])
    cfg = AugmentConfig(
        normalize_input=False,
        use_edit_model=True,
        use_weighted_augs=False,
        benign_min_edits=1,
        benign_max_edits=2,
    )
    augmenter = CAHOAugmenter(config=cfg, benign_edits=benign_edits)
    for _ in range(30):
        hostname = _random_hostname(rng, min_labels=2, max_labels=4)
        out = augmenter.augment(hostname, is_malicious=False, rng=rng)
        assert out
        assert len(out) >= 1
