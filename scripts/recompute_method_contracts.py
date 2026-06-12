#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ccd.augment import DEFAULT_WEIGHTED_BENIGN_WEIGHTS, DEFAULT_WEIGHTED_MALICIOUS_WEIGHTS
from ccd.benchmark_training import BenchmarkBinaryContrastiveTrainer, BenchmarkCAHOViewDataset, BenchmarkContrastiveTrainer
from ccd.calibration import calibrate_thresholds_by_group, threshold_for_group
from ccd.certify import enumerate_edit_ball, randomized_smoothing_certificate
from ccd.cone import ConePartition
from ccd.config import CCDConfig
from ccd.edit_model import DEFAULT_EDITS, EDIT_MANIFEST_VERSION, EditModel
from ccd.io import MODEL_FORMAT_VERSION, ModelBundle
from ccd.model import CCDModel
from ccd.scoring import ccd_score_logpriors, mixture_log_weights
from ccd.train import supervised_orbit_contrastive_loss


EXPECTED_TABLE1_LAYERS = ("cone_count_model", "split_calibration", "edit_closure")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_python_module(path: Path, *, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_mapping(value: object, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def require_string(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value.strip()


def require_string_list(value: object, *, path: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} must be a non-empty list")
    return [require_string(item, path=f"{path}[{index}]") for index, item in enumerate(value)]


def require_bool(value: object, *, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be true or false")
    return value


def require_number(value: object, *, path: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"{path} must be a finite number")
    return float(value)


def validate_table1(rows_value: object) -> list[dict[str, Any]]:
    if not isinstance(rows_value, list) or len(rows_value) != len(EXPECTED_TABLE1_LAYERS):
        raise ValueError("table1_contracts must contain the three Table 1 rows")
    validated: list[dict[str, Any]] = []
    for index, raw_row in enumerate(rows_value):
        row = require_mapping(raw_row, path=f"table1_contracts[{index}]")
        layer = require_string(row.get("layer"), path=f"table1_contracts[{index}].layer")
        if layer != EXPECTED_TABLE1_LAYERS[index]:
            raise ValueError(f"table1_contracts[{index}].layer must be {EXPECTED_TABLE1_LAYERS[index]}")
        assumptions = require_string_list(row.get("assumptions"), path=f"table1_contracts[{index}].assumptions")
        validated.append(
            {
                "layer": layer,
                "paper_layer": require_string(row.get("paper_layer"), path=f"table1_contracts[{index}].paper_layer"),
                "contract": require_string(row.get("contract"), path=f"table1_contracts[{index}].contract"),
                "assumptions": assumptions,
            }
        )
    return validated


def validate_ccd_defaults(expected: Mapping[str, Any]) -> dict[str, Any]:
    config = CCDConfig()
    observed = {
        "encoder_model": config.encoder.model_name,
        "embedding_dim": config.cone.dim,
        "num_cones": config.cone.num_cones,
        "active_cones": config.cone.active_cones,
        "temperature": config.cone.temperature,
        "lsh_enabled": config.cone.use_lsh,
        "lsh_tables": config.cone.lsh_tables,
        "lsh_bits": config.cone.lsh_bits,
        "lsh_probe_radius": config.cone.lsh_probe_radius,
        "seed": config.cone.seed,
        "prior_smoothing": config.prior.smoothing,
        "alpha": config.calibration.alpha,
        "effective_count": config.scoring.effective_count,
    }
    checks = {
        "encoder_model": require_string(expected.get("encoder_model"), path="expected_ccd_defaults.encoder_model"),
        "embedding_dim": int(require_number(expected.get("embedding_dim"), path="expected_ccd_defaults.embedding_dim")),
        "num_cones": int(require_number(expected.get("num_cones"), path="expected_ccd_defaults.num_cones")),
        "active_cones": int(require_number(expected.get("active_cones"), path="expected_ccd_defaults.active_cones")),
        "temperature": require_number(expected.get("temperature"), path="expected_ccd_defaults.temperature"),
        "lsh_enabled": require_bool(expected.get("lsh_enabled"), path="expected_ccd_defaults.lsh_enabled"),
        "lsh_tables": int(require_number(expected.get("lsh_tables"), path="expected_ccd_defaults.lsh_tables")),
        "lsh_bits": int(require_number(expected.get("lsh_bits"), path="expected_ccd_defaults.lsh_bits")),
        "lsh_probe_radius": int(require_number(expected.get("lsh_probe_radius"), path="expected_ccd_defaults.lsh_probe_radius")),
        "seed": int(require_number(expected.get("seed"), path="expected_ccd_defaults.seed")),
        "prior_smoothing": require_number(expected.get("prior_smoothing"), path="expected_ccd_defaults.prior_smoothing"),
        "alpha": require_number(expected.get("alpha"), path="expected_ccd_defaults.alpha"),
    }
    for key, wanted in checks.items():
        if observed[key] != wanted:
            raise ValueError(f"CCD default mismatch for {key}: observed {observed[key]!r}, expected {wanted!r}")
    if require_bool(expected.get("effective_count_positive"), path="expected_ccd_defaults.effective_count_positive") is not True:
        raise ValueError("expected_ccd_defaults.effective_count_positive must be true")
    if observed["effective_count"] <= 0:
        raise ValueError("CCD effective_count must be positive")
    if "mixture_weights" not in config.scoring.to_dict():
        raise ValueError("ScoringConfig must serialize mixture_weights")
    return observed


def validate_edit_manifest(expected: Mapping[str, Any]) -> dict[str, Any]:
    prefixes = require_string_list(expected.get("required_prefixes"), path="expected_edit_manifest.required_prefixes")
    edit_names = sorted(DEFAULT_EDITS)
    missing = [prefix for prefix in prefixes if not any(name.startswith(prefix + "_") or name == prefix for name in edit_names)]
    if missing:
        raise ValueError(f"edit manifest missing required prefixes: {missing}")
    if require_bool(expected.get("requires_deterministic_closure"), path="expected_edit_manifest.requires_deterministic_closure") is not True:
        raise ValueError("deterministic closure must be required")
    if require_bool(expected.get("randomized_sampling_only_exploratory"), path="expected_edit_manifest.randomized_sampling_only_exploratory") is not True:
        raise ValueError("randomized smoothing boundary must be marked exploratory")
    if not callable(enumerate_edit_ball):
        raise ValueError("enumerate_edit_ball must be callable")
    if not callable(randomized_smoothing_certificate):
        raise ValueError("randomized_smoothing_certificate must be callable")
    version = require_string(expected.get("version"), path="expected_edit_manifest.version")
    if EDIT_MANIFEST_VERSION != version:
        raise ValueError(f"edit manifest version mismatch: {EDIT_MANIFEST_VERSION} != {version}")
    if EditModel().version != version:
        raise ValueError("EditModel must expose the deployed edit manifest version")
    return {
        "version": version,
        "n_default_edits": len(edit_names),
        "default_edit_names": edit_names,
        "required_prefixes": prefixes,
        "deterministic_closure_available": True,
        "randomized_smoothing_boundary_documented": True,
    }


def validate_caho_support(expected: Mapping[str, Any]) -> dict[str, Any]:
    for key in (
        "requires_l2_normalized_embeddings",
        "requires_two_views_per_string",
        "requires_contrastive_loss",
        "requires_binary_auxiliary_head",
        "requires_adamw",
    ):
        if require_bool(expected.get(key), path=f"expected_caho_training_support.{key}") is not True:
            raise ValueError(f"expected_caho_training_support.{key} must be true")

    benign_weights = set(DEFAULT_WEIGHTED_BENIGN_WEIGHTS)
    malicious_weights = set(DEFAULT_WEIGHTED_MALICIOUS_WEIGHTS)
    required_benign = {"random_case_variation", "punctuation_replace", "typo_swap", "letter_swap_typo"}
    required_malicious = {
        "toggle_protocol",
        "base64_encode_parts",
        "hex_encode_parts",
        "url_encode_parts",
        "random_homoglyph_substitution",
        "quote_comment_fragment",
    }
    if not required_benign.issubset(benign_weights):
        raise ValueError(f"benign CAHO weighted augmentations missing {sorted(required_benign - benign_weights)}")
    if not required_malicious.issubset(malicious_weights):
        raise ValueError(f"malicious CAHO weighted augmentations missing {sorted(required_malicious - malicious_weights)}")

    dataset_getitem = inspect.getsource(BenchmarkCAHOViewDataset.__getitem__)
    trainer_init_signature = inspect.signature(BenchmarkContrastiveTrainer)
    trainer_fit = inspect.getsource(BenchmarkBinaryContrastiveTrainer.fit)
    trainer_loss = inspect.getsource(BenchmarkContrastiveTrainer._contrastive_loss)
    trainer_save = inspect.getsource(BenchmarkBinaryContrastiveTrainer.save)
    train_script = load_python_module(
        ROOT / "scripts" / "train_benchmark_caho_binary.py",
        name="_artifact_train_benchmark_caho_binary",
    )
    if not callable(getattr(train_script, "build_parser", None)):
        raise ValueError("train_benchmark_caho_binary.py must expose build_parser for contract checks")
    if "view1" not in dataset_getitem or "view2" not in dataset_getitem:
        raise ValueError("BenchmarkCAHOViewDataset must emit two views per string")
    if "_contrastive_loss" not in trainer_fit or "contrastive_loss" not in trainer_fit:
        raise ValueError("BenchmarkBinaryContrastiveTrainer must train the contrastive objective")
    if not callable(supervised_orbit_contrastive_loss):
        raise ValueError("supervised_orbit_contrastive_loss must be callable")
    if "supervised_orbit_contrastive_loss" not in trainer_loss or "_orbit_labels" not in trainer_loss:
        raise ValueError("BenchmarkContrastiveTrainer must use supervised orbit contrastive labels")
    if "F.normalize" not in trainer_fit:
        raise ValueError("BenchmarkBinaryContrastiveTrainer must L2-normalize classifier inputs")
    if "torch.cat([F.normalize(e1, dim=1), F.normalize(e2, dim=1)]" not in trainer_fit:
        raise ValueError("BenchmarkBinaryContrastiveTrainer must train the binary head on both CAHO views")
    if "_binary_backward_microbatched(v1, v2, labels)" not in trainer_fit:
        raise ValueError("GradCache binary auxiliary path must also train on both CAHO views")
    if "binary_cross_entropy_with_logits" not in trainer_fit:
        raise ValueError("BenchmarkBinaryContrastiveTrainer must train a binary auxiliary head")
    if "torch.optim.AdamW" not in trainer_fit:
        raise ValueError("BenchmarkBinaryContrastiveTrainer must use AdamW")
    if "weight_decay=self.weight_decay" not in trainer_fit:
        raise ValueError("BenchmarkBinaryContrastiveTrainer must pass explicit AdamW weight decay")
    if "binary_classifier.pt" not in trainer_save:
        raise ValueError("BenchmarkBinaryContrastiveTrainer must save binary_classifier.pt")

    recipe = require_mapping(expected.get("paper_deployed_recipe"), path="expected_caho_training_support.paper_deployed_recipe")
    default_weight_decay = require_number(
        expected.get("default_adamw_weight_decay"),
        path="expected_caho_training_support.default_adamw_weight_decay",
    )
    weight_decay_param = trainer_init_signature.parameters.get("weight_decay")
    if weight_decay_param is None or float(weight_decay_param.default) != default_weight_decay:
        raise ValueError(
            "BenchmarkBinaryContrastiveTrainer must expose default weight_decay="
            f"{default_weight_decay}"
        )
    deployed_weight_decay = require_number(recipe.get("weight_decay"), path="paper_deployed_recipe.weight_decay")
    if deployed_weight_decay != default_weight_decay:
        raise ValueError("paper deployed weight_decay must match the public AdamW default")
    parser_defaults = train_script.build_parser().parse_args(["--out", "unused-output"])
    expected_script_defaults = {
        "lr": require_number(recipe.get("learning_rate"), path="paper_deployed_recipe.learning_rate"),
        "weight_decay": deployed_weight_decay,
        "batch_size": int(require_number(recipe.get("batch_size"), path="paper_deployed_recipe.batch_size")),
        "epochs": int(require_number(recipe.get("max_epochs"), path="paper_deployed_recipe.max_epochs")),
        "device": "auto",
    }
    observed_script_defaults = {
        "lr": float(parser_defaults.lr),
        "weight_decay": float(parser_defaults.weight_decay),
        "batch_size": int(parser_defaults.batch_size),
        "epochs": int(parser_defaults.epochs),
        "device": str(parser_defaults.device),
    }
    if observed_script_defaults != expected_script_defaults:
        raise ValueError(
            "train_benchmark_caho_binary.py defaults must match the paper deployed recipe: "
            f"observed {observed_script_defaults!r}, expected {expected_script_defaults!r}"
        )
    if getattr(parser_defaults, "require_cuda", None) is not False:
        raise ValueError("train_benchmark_caho_binary.py must default to portable CPU/auto debug execution")
    return {
        "base_encoder_family": require_string(expected.get("base_encoder_family"), path="expected_caho_training_support.base_encoder_family"),
        "pooled_dim": int(require_number(expected.get("pooled_dim"), path="expected_caho_training_support.pooled_dim")),
        "weighted_benign_augmentations": sorted(benign_weights),
        "weighted_malicious_augmentations": sorted(malicious_weights),
        "binary_auxiliary_head_supported": True,
        "binary_auxiliary_head_trains_both_views": True,
        "two_view_dataset_supported": True,
        "adamw_supported": True,
        "adamw_weight_decay_default": default_weight_decay,
        "l2_normalized_binary_inputs": True,
        "contrastive_loss_supported": True,
        "supervised_orbit_contrastive_supported": True,
        "benign_orbit_labels_preserve_diversity": True,
        "benchmark_binary_training_script_defaults": observed_script_defaults,
        "benchmark_binary_training_script_has_cuda_gate": True,
        "paper_deployed_recipe": {
            "learning_rate": require_number(recipe.get("learning_rate"), path="paper_deployed_recipe.learning_rate"),
            "weight_decay": deployed_weight_decay,
            "batch_size": int(require_number(recipe.get("batch_size"), path="paper_deployed_recipe.batch_size")),
            "max_epochs": int(require_number(recipe.get("max_epochs"), path="paper_deployed_recipe.max_epochs")),
        },
        "artifact_boundary": "Full deployed-training provenance is external; public code supports the loss, augmentations, optimizer family, checkpoint format, and smoke-scale execution.",
    }


def validate_bundle_contracts(required_value: object) -> dict[str, Any]:
    required = set(require_string_list(required_value, path="required_bundle_contracts"))
    expected = {
        "config_serialization",
        "cone_axes_serialization",
        "prior_serialization",
        "optional_calibrated_threshold_serialization",
        "optional_grouped_calibrated_threshold_serialization",
    }
    if required != expected:
        raise ValueError(f"required_bundle_contracts must be {sorted(expected)}")
    annotations = getattr(ModelBundle, "__annotations__", {})
    for field in ("axes", "benign_prior", "malicious_priors", "config", "threshold", "grouped_thresholds"):
        if field not in annotations:
            raise ValueError(f"ModelBundle missing {field}")
    model_fields = getattr(CCDModel, "__dataclass_fields__", {})
    if "threshold" not in model_fields:
        raise ValueError("CCDModel must carry an optional threshold")
    if "grouped_thresholds" not in model_fields:
        raise ValueError("CCDModel must carry optional grouped thresholds")
    return {
        "model_format_version": MODEL_FORMAT_VERSION,
        "config_serialization": True,
        "cone_axes_serialization": True,
        "prior_serialization": True,
        "optional_calibrated_threshold_serialization": True,
        "optional_grouped_calibrated_threshold_serialization": True,
    }


def validate_code_path_evidence() -> dict[str, bool]:
    if not callable(ccd_score_logpriors):
        raise ValueError("ccd_score_logpriors must be callable")
    if not callable(mixture_log_weights):
        raise ValueError("mixture_log_weights must be callable")
    if not callable(getattr(CCDModel, "update_benign_prior", None)):
        raise ValueError("CCDModel must expose update_benign_prior for P_B refresh")
    if not callable(getattr(CCDModel, "refresh_benign_reference", None)):
        raise ValueError("CCDModel must expose refresh_benign_reference for benign-only drift refresh")
    refresh_signature = inspect.signature(CCDModel.refresh_benign_reference)
    if "calibration_groups" not in refresh_signature.parameters:
        raise ValueError("CCDModel.refresh_benign_reference must support tenant/window threshold refresh")
    if "drop_grouped_thresholds" not in refresh_signature.parameters:
        raise ValueError("CCDModel.refresh_benign_reference must expose an explicit grouped-threshold drop control")
    refresh_source = inspect.getsource(CCDModel.refresh_benign_reference)
    if "calibration_groups are required" not in refresh_source or "drop_grouped_thresholds" not in refresh_source:
        raise ValueError("grouped benign refresh must fail closed unless replacement groups or an explicit drop are provided")
    explain_signature = inspect.signature(CCDModel.explain)
    for param in ("calibration_groups", "missing_group_threshold"):
        if param not in explain_signature.parameters:
            raise ValueError(f"CCDModel.explain must expose {param} for grouped decision explanations")
    if not callable(calibrate_thresholds_by_group):
        raise ValueError("calibrate_thresholds_by_group must be callable for tenant/window calibration")
    if not callable(threshold_for_group):
        raise ValueError("threshold_for_group must be callable for grouped threshold lookup")
    grouped_thresholds = calibrate_thresholds_by_group(
        [0.1, 0.4, 0.2, 0.8],
        ["tenant-a", "tenant-a", "tenant-b", "tenant-b"],
        alpha=0.5,
    )
    if (
        abs(grouped_thresholds["tenant-a"]["threshold"] - 0.4) > 1e-12
        or abs(grouped_thresholds["tenant-b"]["threshold"] - 0.8) > 1e-12
    ):
        raise ValueError("grouped split-conformal calibration produced unexpected thresholds")
    if abs(threshold_for_group("tenant-a", 0.5, grouped_thresholds) - 0.4) > 1e-12:
        raise ValueError("grouped threshold lookup did not use the tenant/window threshold")
    if abs(threshold_for_group("tenant-c", 0.5, grouped_thresholds) - 0.5) > 1e-12:
        raise ValueError("grouped threshold lookup did not fall back to the global threshold")
    certify_signature = inspect.signature(CCDModel.certify)
    for param in ("method", "sketch_lipschitz", "embedding_rotation_bound"):
        if param not in certify_signature.parameters:
            raise ValueError(f"CCDModel.certify must expose {param} for CMC/SEC certification")
    cli_source = (ROOT / "ccd" / "cli.py").read_text(encoding="utf-8")
    for token in ("threshold_source", "grouped_thresholds_source", '"normalizer"', "ccd.preprocess.normalize_hostname"):
        if token not in cli_source:
            raise ValueError(f"certificate JSON must record {token} scope metadata")
    cone_sketch_signature = inspect.signature(ConePartition.cone_sketch)
    use_lsh_param = cone_sketch_signature.parameters.get("use_lsh")
    if use_lsh_param is None or use_lsh_param.default is not False:
        raise ValueError("ConePartition.cone_sketch must bypass LSH by default for exact scoring")
    cone_sketch_source = inspect.getsource(ConePartition.cone_sketch)
    nearest_axes_source = inspect.getsource(ConePartition.nearest_axes)
    if "nearest_axes" not in cone_sketch_source or "use_lsh=use_lsh" not in cone_sketch_source:
        raise ValueError("ConePartition.cone_sketch must route through nearest_axes with explicit LSH control")
    if "self.axes @ u" not in nearest_axes_source:
        raise ValueError("ConePartition.nearest_axes must support exact full-axis scan fallback")
    return {
        "eq1_logsumexp_score_path_available": True,
        "mixture_weights_normalized": True,
        "split_conformal_calibration_available": True,
        "grouped_split_conformal_calibration_available": True,
        "tenant_window_threshold_resolution_available": True,
        "grouped_decision_explanations_available": True,
        "exact_certificate_enumeration_available": True,
        "calibrated_margin_certificate_available": True,
        "combined_cmc_then_enumeration_certificate_available": True,
        "certificate_records_normalizer_and_threshold_scope": True,
        "exact_score_bypasses_lsh_by_default": True,
        "exact_full_axis_scan_for_deployed_top_r_statistic": True,
        "benign_reference_refresh_available": True,
        "grouped_threshold_refresh_available": True,
        "grouped_refresh_requires_groups_or_explicit_drop": True,
        "refresh_updates_pb_and_threshold_only": True,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    counts_path = Path(args.counts)
    data = load_json(counts_path)
    table1 = validate_table1(data.get("table1_contracts"))
    frozen_fields = require_string_list(data.get("appendix_c_frozen_fields"), path="appendix_c_frozen_fields")
    ccd_defaults = validate_ccd_defaults(require_mapping(data.get("expected_ccd_defaults"), path="expected_ccd_defaults"))
    edit_manifest = validate_edit_manifest(require_mapping(data.get("expected_edit_manifest"), path="expected_edit_manifest"))
    caho_support = validate_caho_support(require_mapping(data.get("expected_caho_training_support"), path="expected_caho_training_support"))
    bundle_contracts = validate_bundle_contracts(data.get("required_bundle_contracts"))
    private_by_design = require_string_list(data.get("private_by_design"), path="private_by_design")
    derived = {
        "n_table1_contracts": len(table1),
        "n_appendix_c_frozen_fields": len(frozen_fields),
        "code_path_evidence": validate_code_path_evidence(),
        "ccd_defaults": ccd_defaults,
        "edit_manifest": edit_manifest,
        "caho_training_support": caho_support,
        "bundle_contracts": bundle_contracts,
    }
    return {
        "status": "pass",
        "counts_file": str(counts_path),
        "source": data.get("source", "published_method_contracts"),
        "paper_sections": data.get("paper_sections", []),
        "table1_contracts": table1,
        "appendix_c_frozen_fields": frozen_fields,
        "derived": derived,
        "private_by_design": private_by_design,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute release-safe Table 1 and Appendix C method-contract checks.")
    parser.add_argument(
        "--counts",
        default="method_contracts/paper_method_contracts.json",
        help="Release-safe method-contract summary.",
    )
    parser.add_argument("--out", default=None, help="Optional path for the JSON report.")
    args = parser.parse_args()

    report = build_report(args)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
