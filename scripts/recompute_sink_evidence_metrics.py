#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


EXPECTED_CASE_IDS = ("diagnostic_helper", "alert_action", "analytics_query")
REQUIRED_SCOPE_FLAGS = (
    "evidence_trace_not_compromise_claim",
    "controlled_matching_sink_replay",
    "marker_only_strings_not_positive_without_downstream_support",
)
RELEASE_SAFETY_FLAGS = (
    "raw_retained_value_included",
    "callback_domain_included",
    "private_sink_detail_included",
    "production_compromise_claim",
)
DISALLOWED_RELEASE_TOKENS = ("$(", "nslookup", "curl ", " oast", ".oast", "://")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_bool(value: object, *, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be true or false")
    return value


def require_string(value: object, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value.strip()


def require_mapping(value: object, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def require_string_list(value: object, *, path: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} must be a non-empty list")
    out: list[str] = []
    for index, item in enumerate(value):
        out.append(require_string(item, path=f"{path}[{index}]"))
    return out


def check_release_safe_text(value: object, *, path: str) -> None:
    text = json.dumps(value, sort_keys=True).lower()
    for token in DISALLOWED_RELEASE_TOKENS:
        if token in text:
            raise ValueError(f"{path} contains raw callback or executable-token detail: {token.strip()}")


def validate_scope(scope: Mapping[str, Any]) -> dict[str, Any]:
    for flag in REQUIRED_SCOPE_FLAGS:
        if require_bool(scope.get(flag), path=f"scope.{flag}") is not True:
            raise ValueError(f"scope.{flag} must be true")
    if require_bool(scope.get("production_compromise_claim"), path="scope.production_compromise_claim") is not False:
        raise ValueError("scope.production_compromise_claim must be false")
    side_effect_policy = require_string(scope.get("side_effect_policy"), path="scope.side_effect_policy")
    if side_effect_policy != "blocked_or_restricted_to_researcher_controlled_endpoints":
        raise ValueError("scope.side_effect_policy must block or restrict side effects")
    return {
        "evidence_trace_not_compromise_claim": True,
        "production_compromise_claim": False,
        "controlled_matching_sink_replay": True,
        "marker_only_strings_not_positive_without_downstream_support": True,
        "side_effect_policy": side_effect_policy,
    }


def validate_case(
    case: Mapping[str, Any],
    *,
    index: int,
    required_fields: set[str],
    allowed_effects: set[str],
    allowed_guards: set[str],
) -> dict[str, Any]:
    case_id = require_string(case.get("case_id"), path=f"cases[{index}].case_id")
    if case_id not in EXPECTED_CASE_IDS:
        raise ValueError(f"cases[{index}].case_id is unsupported: {case_id}")
    for field in required_fields:
        require_string(case.get(field), path=f"cases[{index}].{field}")
    sink_family = require_string(case.get("sink_family"), path=f"cases[{index}].sink_family")
    evidence_tier = require_string(case.get("evidence_tier"), path=f"cases[{index}].evidence_tier")
    if evidence_tier != "controlled_matching_sink_replay":
        raise ValueError(f"cases[{index}].evidence_tier must be controlled_matching_sink_replay")

    effect = require_string(case.get("controlled_effect"), path=f"cases[{index}].controlled_effect")
    if effect not in allowed_effects:
        raise ValueError(f"cases[{index}].controlled_effect is unsupported: {effect}")
    guard = require_string(case.get("side_effect_guard"), path=f"cases[{index}].side_effect_guard")
    if guard not in allowed_guards:
        raise ValueError(f"cases[{index}].side_effect_guard is unsupported: {guard}")

    if require_bool(case.get("controlled_replay"), path=f"cases[{index}].controlled_replay") is not True:
        raise ValueError(f"cases[{index}].controlled_replay must be true")
    if require_bool(case.get("ccd_fired_before_consumption"), path=f"cases[{index}].ccd_fired_before_consumption") is not True:
        raise ValueError(f"cases[{index}].ccd_fired_before_consumption must be true")
    for flag in RELEASE_SAFETY_FLAGS:
        if require_bool(case.get(flag), path=f"cases[{index}].{flag}") is not False:
            raise ValueError(f"cases[{index}].{flag} must be false")
    check_release_safe_text(case, path=f"cases[{index}]")

    return {
        "case_id": case_id,
        "paper_case": require_string(case.get("paper_case"), path=f"cases[{index}].paper_case"),
        "sink_family": sink_family,
        "controlled_effect": effect,
        "side_effect_guard": guard,
        "evidence_tier": evidence_tier,
        "controlled_replay": True,
        "ccd_fired_before_consumption": True,
        "production_compromise_claim": False,
    }


def validate_cases(
    cases: object,
    *,
    required_fields: set[str],
    allowed_effects: set[str],
    allowed_guards: set[str],
) -> list[dict[str, Any]]:
    if not isinstance(cases, list):
        raise ValueError("cases must be a list")
    if len(cases) != len(EXPECTED_CASE_IDS):
        raise ValueError(f"cases must contain exactly {len(EXPECTED_CASE_IDS)} Table 8 cases")
    validated = [
        validate_case(
            require_mapping(case, path=f"cases[{index}]"),
            index=index,
            required_fields=required_fields,
            allowed_effects=allowed_effects,
            allowed_guards=allowed_guards,
        )
        for index, case in enumerate(cases)
    ]
    observed_ids = tuple(item["case_id"] for item in validated)
    if observed_ids != EXPECTED_CASE_IDS:
        raise ValueError(f"case order must be {EXPECTED_CASE_IDS}, observed {observed_ids}")
    return validated


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    counts_path = Path(args.counts)
    data = load_json(counts_path)
    scope = validate_scope(require_mapping(data.get("scope"), path="scope"))
    required_fields = set(require_string_list(data.get("required_case_fields"), path="required_case_fields"))
    expected_fields = {"parser_boundary", "persistence", "consumer", "controlled_effect", "detector_boundary"}
    if required_fields != expected_fields:
        raise ValueError(f"required_case_fields must be {sorted(expected_fields)}")
    allowed_effects = set(require_string_list(data.get("allowed_controlled_effects"), path="allowed_controlled_effects"))
    allowed_guards = set(require_string_list(data.get("allowed_side_effect_guards"), path="allowed_side_effect_guards"))
    cases = validate_cases(
        data.get("cases"),
        required_fields=required_fields,
        allowed_effects=allowed_effects,
        allowed_guards=allowed_guards,
    )
    private_by_design = require_string_list(data.get("private_by_design"), path="private_by_design")
    check_release_safe_text(private_by_design, path="private_by_design")

    sink_families = [case["sink_family"] for case in cases]
    controlled_effects = [case["controlled_effect"] for case in cases]
    side_effect_guards = [case["side_effect_guard"] for case in cases]
    derived = {
        "n_cases": len(cases),
        "case_ids": [case["case_id"] for case in cases],
        "sink_families": sink_families,
        "controlled_effects": controlled_effects,
        "side_effect_guards": side_effect_guards,
        "all_cases_controlled_replay": all(case["controlled_replay"] for case in cases),
        "all_cases_ccd_before_consumption": all(case["ccd_fired_before_consumption"] for case in cases),
        "all_cases_non_compromise_claims": all(
            case["production_compromise_claim"] is False for case in cases
        )
        and scope["production_compromise_claim"] is False,
        "release_safe_case_text": True,
    }

    return {
        "status": "pass",
        "counts_file": str(counts_path),
        "source": data.get("source", "published_table8_sink_evidence_accounting"),
        "paper_sections": data.get("paper_sections", []),
        "scope": scope,
        "cases": cases,
        "derived": derived,
        "private_by_design": private_by_design,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute release-safe Table 8 sink-evidence accounting.")
    parser.add_argument(
        "--counts",
        default="sink_evidence/paper_sink_evidence_counts.json",
        help="Release-safe Table 8 sink-evidence aggregate counts.",
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
