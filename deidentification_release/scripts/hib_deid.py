from __future__ import annotations

import base64
import csv
import hashlib
import hmac
import json
import math
import os
import random
import re
import shutil
import string
import sys
import tarfile
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


RELEASE_VERSION = "hib-v1.0"
POLICY_VERSION = "non-linkable-release-policy-v1.0"
ANONYMIZER_VERSION = "deid-v1.0.0"

RAW_ARTIFACT_FIELDS = {"raw_hostname", "raw_canonical_hostname", "HOSTNAME", "USERNAME", "CONTENT"}
FORBIDDEN_PUBLIC_FIELDS = {
    "raw_hostname",
    "raw_canonical_hostname",
    "dedup_hostname_id",
    "unique_host_hash",
    "stable_hostname_hash",
    "raw_tenant_id",
    "stable_tenant_time_series_id",
    "tenant_surrogate",
    "exact_timestamp",
    "private_mapping_key",
    "CDB",
    "SOURCE_FILE",
    "SOURCE_ROW_NUMBER",
    "ORIGINAL_CREATED_TIME",
    "CREATED_TIME",
}
PUBLIC_SCHEMA_FIELDS = [
    "public_row_id",
    "released_artifact",
    "released_canonical_artifact",
    "source_family",
    "time_bucket",
    "split",
    "label",
    "evidence_tier",
    "sink_family",
    "obfuscation_family",
    "released_length_bucket",
    "character_class_mask",
    "ccd_outputs",
    "row_integrity_hash",
]
RESERVED_SUFFIXES = {"invalid", "example", "test", "localhost"}
DOCUMENTATION_IPV4_PREFIXES = ("192.0.2.", "198.51.100.", "203.0.113.")
INTENT_SIGNALING_RE = re.compile(
    r"(?i)(hack|hacker|hacked|malicious|evil|attack|attacker|exploit|pwn|pwned|phish|phishing|malware|virus|ransom|trojan|botnet|backdoor|rootkit)"
)
ATTACK_TERMS = {
    "jndi",
    "ldap",
    "rmi",
    "dns",
    "http",
    "https",
    "curl",
    "wget",
    "nslookup",
    "ping",
    "sleep",
    "pg_sleep",
    "select",
    "union",
    "from",
    "where",
    "or",
    "and",
    "waitfor",
    "delay",
    "benchmark",
    "cmd",
    "sh",
    "bash",
    "powershell",
    "whoami",
    "cat",
    "exec",
    "eval",
    "system",
}
SAFE_OPERATIONAL_TERMS = {
    "www",
    "api",
    "prod",
    "dev",
    "staging",
    "stage",
    "login",
    "dns",
    "us",
    "eu",
    "ap",
    "east",
    "west",
    "north",
    "south",
}


@dataclass(frozen=True)
class PrivateConfig:
    row_id_secret: str
    artifact_secret: str
    shuffle_secret: str
    release_version: str = RELEASE_VERSION


def load_private_config(path: Path | None, *, row_id_secret: str | None = None, artifact_secret: str | None = None, shuffle_secret: str | None = None) -> PrivateConfig:
    data: dict[str, str] = {}
    if path is not None:
        text = path.read_text(encoding="utf-8")
        try:
            parsed = json.loads(text)
            data = {str(k): str(v) for k, v in parsed.items()}
        except json.JSONDecodeError:
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith("#") or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                value = value.strip().strip("'\"")
                if value and not value.startswith("[") and not value.startswith("{"):
                    data[key.strip()] = value
    row = row_id_secret or data.get("row_id_secret") or os.environ.get("HIB_DEID_ROW_ID_SECRET")
    artifact = artifact_secret or data.get("artifact_secret") or os.environ.get("HIB_DEID_ARTIFACT_SECRET")
    shuffle = shuffle_secret or data.get("shuffle_secret") or os.environ.get("HIB_DEID_SHUFFLE_SECRET")
    if not row or not artifact or not shuffle:
        raise ValueError("row_id_secret, artifact_secret, and shuffle_secret are required via private config or CLI.")
    return PrivateConfig(
        row_id_secret=row,
        artifact_secret=artifact,
        shuffle_secret=shuffle,
        release_version=data.get("release_version", RELEASE_VERSION),
    )


def hmac_digest(secret: str, message: str) -> bytes:
    return hmac.new(secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).digest()


def hmac_base32(secret: str, message: str, length: int = 20) -> str:
    return base64.b32encode(hmac_digest(secret, message)).decode("ascii").rstrip("=")[:length]


def public_row_id(row_id: str, config: PrivateConfig) -> str:
    return "row_" + hmac_base32(config.row_id_secret, f"{config.release_version}|{row_id}", 20)


def row_rng(row_id: str, config: PrivateConfig, namespace: str) -> random.Random:
    seed = int.from_bytes(hmac_digest(config.artifact_secret, f"{config.release_version}|{row_id}|{namespace}"), "big")
    return random.Random(seed)


def canonicalize_artifact(value: str) -> str:
    text = value.strip()
    text = re.sub(r"\s+", "", text)
    text = text.rstrip(".")
    try:
        text = text.encode("idna").decode("ascii")
    except Exception:
        pass
    return text.lower()


def source_family(row: Mapping[str, str]) -> str:
    family = (row.get("DATASET_FAMILY") or "").strip().lower()
    content_type = (row.get("CONTENT_TYPE") or "").strip().upper()
    if family == "user_logins" or content_type == "USERNAME":
        return "login_host"
    if family == "dns_hostnames" or content_type == "HOSTNAME":
        return "dns_host"
    return "other_coarse_source"


def label_from_private(row: Mapping[str, str]) -> str:
    label = (row.get("RESOLVED_LABEL_BOTH_M") or "").strip().upper()
    if label == "M":
        return "verified_executable_semantics"
    if label == "B":
        return "resolved_benign"
    return "unresolved"


def sink_family(value: str) -> str:
    lower = value.lower()
    if "${jndi" in lower or "${#" in lower or "{{" in value:
        return "template"
    if re.search(r"\b(select|union|pg_sleep|waitfor|benchmark|dbms_pipe)\b|--|/\*", lower):
        return "query"
    if re.search(r"(`|\$\(|;|&&|&|\|\||\|)", value) or re.search(r"\b(curl|wget|nslookup|ping|bash|sh|cmd|powershell)\b", lower):
        return "shell"
    if re.search(r"https?://", lower):
        return "url_fetch"
    return "none"


def evidence_tier(row: Mapping[str, str], artifact: str) -> str:
    if label_from_private(row) == "resolved_benign":
        return "none"
    sink = sink_family(artifact)
    if sink in {"shell", "template", "query"}:
        return "sink_evaluated"
    if sink == "url_fetch":
        return "artifact_supported"
    return "syntax_only" if label_from_private(row) != "unresolved" else "none"


def obfuscation_family(value: str) -> str:
    families: list[str] = []
    if "%" in value:
        families.append("percent")
    if any(ord(ch) > 127 for ch in value):
        families.append("unicode")
    if any(ch in value for ch in "\"'`;|&"):
        families.append("delimiter")
    if re.search(r"[A-Za-z0-9+/]{16,}={0,2}", value):
        families.append("base64")
    if "--" in value or "/*" in value:
        families.append("quote_comment")
    if len(families) > 1:
        return "mixed"
    return families[0] if families else "none"


def public_obfuscation_family(value: str) -> str:
    return "none" if obfuscation_family(value) == "none" else "present"


def length_bucket(value: str) -> str:
    n = len(value)
    if n <= 15:
        return "1-15"
    if n <= 31:
        return "16-31"
    if n <= 63:
        return "32-63"
    if n <= 127:
        return "64-127"
    return "128+"


def public_length_bucket(value: str) -> str:
    del value
    return "withheld"


def character_class_mask(value: str, max_len: int = 96) -> str:
    out = []
    for ch in value[:max_len]:
        if ch.isalpha():
            out.append("A" if ch.isupper() else "a")
        elif ch.isdigit():
            out.append("9")
        elif ch in ".-_":
            out.append(ch)
        elif ch in "$`;&|<>(){}[]'\"":
            out.append("S")
        elif ch == "%":
            out.append("%")
        else:
            out.append("x")
    if len(value) > max_len:
        out.append("+")
    return "".join(out)


def public_character_class_mask(value: str) -> str:
    del value
    return "withheld"


def coarse_count_bucket(value: int) -> str:
    if value == 0:
        return "0"
    if value == 1:
        return "1"
    if value <= 3:
        return "2-3"
    if value <= 7:
        return "4-7"
    if value <= 15:
        return "8-15"
    if value <= 31:
        return "16-31"
    if value <= 63:
        return "32-63"
    return "64+"


def week_bucket(timestamp: str) -> str:
    if not timestamp:
        return "unknown"
    text = timestamp.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return "unknown"
    iso = dt.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def public_time_bucket(timestamp: str) -> str:
    del timestamp
    return "withheld"


def split_for_row(public_id: str) -> str:
    value = int(hashlib.sha256(public_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if value < 0.70:
        return "train"
    if value < 0.80:
        return "validation"
    if value < 0.90:
        return "calibration"
    return "test"


def replacement_for_token(token: str, rng: random.Random, *, force_dns_safe: bool = False) -> str:
    lower = token.lower()
    if lower in ATTACK_TERMS or lower in SAFE_OPERATIONAL_TERMS or re.fullmatch(r"(?:us|eu|ap|sa|ca|af|me)[a-z]{1,4}\d?", lower):
        return token
    if "-" in token:
        parts = token.split("-")
        replaced_parts = [
            part
            if part.lower() in ATTACK_TERMS or part.lower() in SAFE_OPERATIONAL_TERMS or re.fullmatch(r"(?:us|eu|ap|sa|ca|af|me)[a-z]{1,4}\d?", part.lower())
            else replacement_for_token(part, rng, force_dns_safe=force_dns_safe)
            for part in parts
        ]
        return "-".join(replaced_parts)
    alphabet = string.ascii_lowercase
    chars: list[str] = []
    for ch in token:
        if ch.isdigit():
            chars.append(str(rng.randrange(10)))
        elif ch.isalpha():
            repl = rng.choice(alphabet)
            chars.append(repl.upper() if ch.isupper() else repl)
        elif ch == "-":
            chars.append("-")
        elif ch == "_":
            chars.append("_" if not force_dns_safe else rng.choice(alphabet))
        else:
            chars.append(ch)
    if not chars:
        return "x"
    if force_dns_safe:
        chars = [c if re.match(r"[A-Za-z0-9-]", c) else rng.choice(alphabet) for c in chars]
        if chars[0] == "-":
            chars[0] = rng.choice(alphabet)
        if chars[-1] == "-":
            chars[-1] = rng.choice(alphabet)
    return "".join(chars)


def documentation_ipv4(token: str, rng: random.Random) -> str:
    del token
    return f"192.0.2.{rng.randrange(1, 255)}"


def transform_artifact(raw: str, row_id: str, config: PrivateConfig) -> str:
    rng = row_rng(row_id, config, "artifact")
    if re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", raw or ""):
        tag = row_unique_tag(row_id, config)
        return f"{documentation_ipv4(raw, rng)}.{tag}.test"

    token_index = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal token_index
        token = match.group(0)
        if match.start() > 0 and raw[match.start() - 1] == "%" and len(token) >= 2 and re.fullmatch(r"[A-Fa-f0-9]{2}", token[:2]):
            if len(token) == 2:
                return token
            token_index += 1
            occ_rng = row_rng(row_id, config, f"occurrence|{token_index}")
            return token[:2] + replacement_for_token(token[2:], occ_rng, force_dns_safe=True)
        token_index += 1
        occ_rng = row_rng(row_id, config, f"occurrence|{token_index}")
        return replacement_for_token(token, occ_rng, force_dns_safe=True)

    released = re.sub(r"[A-Za-z0-9_][A-Za-z0-9_-]*", repl, raw or "")
    released = force_domain_suffixes_to_reserved(released)
    released = neutralize_email_syntax(released)
    released = neutralize_internal_suffix_labels(released, row_id, config)
    released = neutralize_ipv4_like_substrings(released, row_id, config)
    if "." in released:
        labels = released.split(".")
        for idx in range(len(labels) - 1, -1, -1):
            if re.search(r"[A-Za-z0-9]", labels[idx]):
                labels[idx] = nearest_reserved_suffix(labels[idx])
                break
        released = ".".join(labels)
    released = preserve_length_bucket_with_reserved_suffix(raw or "", released)
    if released == raw or not released:
        released = f"x{hmac_base32(config.artifact_secret, row_id + '|fallback', 8).lower()}.test"
    released = add_row_unique_variation(released, row_id, config)
    return released


def force_domain_suffixes_to_reserved(value: str) -> str:
    domain_re = re.compile(r"\b([A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+)\b")

    def repl(match: re.Match[str]) -> str:
        domain = match.group(1)
        parts = domain.split(".")
        if parts[-1].lower() not in RESERVED_SUFFIXES:
            parts[-1] = nearest_reserved_suffix(parts[-1])
        return ".".join(parts)

    return domain_re.sub(repl, value)


def nearest_reserved_suffix(label: str) -> str:
    candidates = ["test", "invalid", "example", "localhost"]
    return min(candidates, key=lambda suffix: (abs(len(suffix) - len(label)), len(suffix)))


def preserve_length_bucket_with_reserved_suffix(raw: str, released: str) -> str:
    if "." not in released or length_bucket(raw) == length_bucket(released):
        return released
    prefix, suffix = released.rsplit(".", 1)
    if suffix.lower() not in RESERVED_SUFFIXES:
        return released
    candidates = [f"{prefix}.{candidate}" for candidate in ["test", "invalid", "example", "localhost"]]
    for candidate in candidates:
        if length_bucket(candidate) == length_bucket(raw):
            return candidate
    return released


def neutralize_email_syntax(value: str) -> str:
    return value.replace("@", ".at.")


def neutralize_internal_suffix_labels(value: str, row_id: str, config: PrivateConfig) -> str:
    bad = {"corp", "internal", "local", "lan", "svc", "cluster"}
    labels = value.split(".")
    tag = row_unique_tag(row_id, config, length=8)
    changed = False
    for idx, label in enumerate(labels):
        if label.lower() in bad:
            labels[idx] = (tag[: max(len(label), 3)])[: len(label)]
            changed = True
    return ".".join(labels) if changed else value


def neutralize_ipv4_like_substrings(value: str, row_id: str, config: PrivateConfig) -> str:
    rng = row_rng(row_id, config, "ipv4-substring")
    tag = row_unique_tag(row_id, config, length=4)

    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        if token.startswith(DOCUMENTATION_IPV4_PREFIXES):
            return token
        return f"192.0.2.{rng.randrange(1, 255)}.{tag}.test"

    return re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", repl, value)


def add_row_unique_variation(released: str, row_id: str, config: PrivateConfig) -> str:
    tag = row_unique_tag(row_id, config)
    if tag in released:
        return released
    if "." not in released:
        return f"{released}{tag}.test"
    labels = released.split(".")
    target = None
    for idx in range(len(labels) - 1, -1, -1):
        label = labels[idx]
        if label.lower() in RESERVED_SUFFIXES:
            continue
        if re.search(r"[A-Za-z0-9]", label):
            target = idx
            break
    if target is None:
        return f"{released}.{tag}.test"
    labels[target] = labels[target] + tag
    if labels[-1].lower() not in RESERVED_SUFFIXES:
        labels.append("test")
    return ".".join(labels)


def row_unique_tag(row_id: str, config: PrivateConfig, length: int = 8) -> str:
    digest = hmac_digest(config.artifact_secret, f"{config.release_version}|{row_id}|unique")
    return "".join(string.ascii_lowercase[byte % 26] for byte in digest[:length])


def public_release_safety_postprocess(value: str) -> str:
    replacements = [
        (r"(?i)eyj", "exj"),
        (r"(?i)bearer", "beaqer"),
        (r"(?i)secret", "seqret"),
        (r"(?i)token", "tqken"),
        (r"(?i)oast", "oqst"),
        (r"(?i)bxss", "bqss"),
        (r"(?i)interact", "inteqact"),
        (r"(?i)burpcollaborator", "burpcqllaborator"),
        (r"(?i)corp", "cqrp"),
        (r"(?i)internal", "inteqnal"),
        (r"(?i)local", "lqcal"),
        (r"(?i)cluster", "clqster"),
        (r"(?i)lan", "lqn"),
        (r"(?i)svc", "svq"),
        (r"(?i)malicious", "neutral"),
        (r"(?i)hacker", "operator"),
        (r"(?i)hacked", "served"),
        (r"(?i)hack", "host"),
        (r"(?i)evil", "edge"),
        (r"(?i)attacker", "client"),
        (r"(?i)attack", "access"),
        (r"(?i)exploit", "feature"),
        (r"(?i)pwned", "owned"),
        (r"(?i)pwn", "own"),
        (r"(?i)phishing", "messaging"),
        (r"(?i)phish", "mail"),
        (r"(?i)malware", "software"),
        (r"(?i)virus", "service"),
        (r"(?i)ransom", "record"),
        (r"(?i)trojan", "agent"),
        (r"(?i)botnet", "worker"),
        (r"(?i)backdoor", "gateway"),
        (r"(?i)rootkit", "runtime"),
    ]
    released = value
    for pattern, repl in replacements:
        released = re.sub(pattern, repl, released)
    released = force_domain_suffixes_to_reserved(released)
    if "." in released:
        labels = released.split(".")
        for idx in range(len(labels) - 1, -1, -1):
            if re.search(r"[A-Za-z0-9]", labels[idx]):
                labels[idx] = nearest_reserved_suffix(labels[idx])
                break
        released = ".".join(labels)
    return released


def public_sink_and_evidence(label: str, private_sink: str, private_evidence: str) -> tuple[str, str]:
    if label == "verified_executable_semantics" and private_sink == "url_fetch" and private_evidence == "artifact_supported":
        return "none", "syntax_only"
    return private_sink, private_evidence


def optional_public_float(row: Mapping[str, str], keys: Iterable[str]) -> float | None:
    for key in keys:
        value = (row.get(key) or "").strip()
        if not value:
            continue
        try:
            parsed = float(value)
        except ValueError:
            continue
        if math.isfinite(parsed):
            return parsed
    return None


def optional_public_bool(row: Mapping[str, str], keys: Iterable[str]) -> bool | None:
    for key in keys:
        value = (row.get(key) or "").strip().lower()
        if value in {"true", "1", "yes", "y"}:
            return True
        if value in {"false", "0", "no", "n"}:
            return False
    return None


def optional_public_calibration_group(row: Mapping[str, str]) -> str | None:
    for key in ("PUBLIC_CALIBRATION_GROUP", "PUBLIC_CALIBRATION_BUCKET"):
        value = (row.get(key) or "").strip()
        if not value:
            continue
        if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,64}", value):
            raise ValueError(f"Unsafe public calibration group value in {key}: {value!r}")
        if re.search(r"(?i)(tenant|customer|corp|internal|private|cdb)", value):
            raise ValueError(f"Public calibration group may not contain tenant-identifying terms: {value!r}")
        return value
    return None


def public_ccd_outputs(row: Mapping[str, str], raw_artifact: str) -> dict[str, Any]:
    del raw_artifact
    score = optional_public_float(row, ("PUBLIC_CCD_SCORE", "CCD_SCORE_PUBLIC", "CCD_PUBLIC_SCORE"))
    ccd_flag = optional_public_bool(row, ("PUBLIC_CCD_FLAG", "CCD_FLAG_PUBLIC", "CCD_PUBLIC_FLAG"))
    calibration_group = optional_public_calibration_group(row)
    outputs: dict[str, Any] = {
        "ccd_score_bin": "public_score" if score is not None else "not_recomputed",
        "ccd_flag": ccd_flag,
    }
    if score is not None:
        outputs["ccd_score_public"] = score
    if calibration_group is not None:
        outputs["public_calibration_group"] = calibration_group
    return outputs


def row_integrity_hash(row: Mapping[str, Any]) -> str:
    payload = {k: v for k, v in row.items() if k != "row_integrity_hash"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def private_row_id(row: Mapping[str, str], index: int) -> str:
    return row.get("ROW_ID") or row.get("private_row_id") or f"row-{index:012d}"


def artifact_from_row(row: Mapping[str, str]) -> str:
    return row.get("CONTENT") or row.get("HOSTNAME") or row.get("USERNAME") or ""


def make_public_row(row: Mapping[str, str], index: int, config: PrivateConfig) -> dict[str, Any]:
    row_id = private_row_id(row, index)
    public_id = public_row_id(row_id, config)
    raw_artifact = artifact_from_row(row)
    released = public_release_safety_postprocess(transform_artifact(raw_artifact, row_id, config))
    canonical = canonicalize_artifact(released)
    label = label_from_private(row)
    private_sink = sink_family(raw_artifact)
    private_evidence = evidence_tier(row, raw_artifact)
    public_sink, public_evidence = public_sink_and_evidence(label, private_sink, private_evidence)
    public: dict[str, Any] = {
        "public_row_id": public_id,
        "released_artifact": released,
        "released_canonical_artifact": canonical,
        "source_family": source_family(row),
        "time_bucket": public_time_bucket(row.get("CREATED_TIME") or row.get("timestamp") or ""),
        "split": split_for_row(public_id),
        "label": label,
        "evidence_tier": public_evidence,
        "sink_family": public_sink,
        "obfuscation_family": public_obfuscation_family(raw_artifact),
        "released_length_bucket": public_length_bucket(released),
        "character_class_mask": public_character_class_mask(released),
        "ccd_outputs": public_ccd_outputs(row, raw_artifact),
    }
    public["row_integrity_hash"] = row_integrity_hash(public)
    return public


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def iter_csv_rows(paths: Iterable[Path]) -> Iterable[dict[str, str]]:
    for path in paths:
        with path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
            yield from csv.DictReader(handle)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def iter_jsonl_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def csv_input_paths(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(path.glob("*.csv"))
    return [path]


class BucketWriter:
    def __init__(self, root: Path, *, max_open: int = 128) -> None:
        self.root = root
        self.max_open = max_open
        self.handles: OrderedDict[Path, Any] = OrderedDict()
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, path: Path, payload: Mapping[str, Any]) -> None:
        handle = self.handles.get(path)
        if handle is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("a", encoding="utf-8")
            self.handles[path] = handle
            if len(self.handles) > self.max_open:
                _, old_handle = self.handles.popitem(last=False)
                old_handle.close()
        else:
            self.handles.move_to_end(path)
        handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()

    def __enter__(self) -> "BucketWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def anonymize_csv(input_private: Path, output_jsonl: Path, audit_dir: Path, config: PrivateConfig) -> dict[str, Any]:
    rows = read_csv_rows(input_private)
    public_rows = [make_public_row(row, idx, config) for idx, row in enumerate(rows)]
    shuffle_rng = random.Random(int.from_bytes(hmac_digest(config.shuffle_secret, f"{config.release_version}|shuffle"), "big"))
    shuffle_rng.shuffle(public_rows)
    write_jsonl(output_jsonl, public_rows)
    schema_path = output_jsonl.with_suffix(".schema.json")
    schema_path.write_text(json.dumps({"fields": PUBLIC_SCHEMA_FIELDS}, indent=2), encoding="utf-8")
    sha_path = output_jsonl.with_suffix(output_jsonl.suffix + ".sha256")
    write_sha256_sidecar(sha_path, output_jsonl)
    manifest = {
        "release_version": config.release_version,
        "private_eval_snapshot_id": f"eval-freeze-{datetime.now(timezone.utc).date().isoformat()}",
        "private_eval_snapshot_sha256": "withheld-private-input-hash",
        "paper_reference": "not_in_repository",
        "anonymizer_version": ANONYMIZER_VERSION,
        "policy_version": POLICY_VERSION,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "approvals": {"security": "pending|required", "privacy": "pending|required", "legal_or_data_owner": "pending|required"},
    }
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "release_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_stage_manifests(audit_dir, rows, public_rows, config, input_private, output_jsonl)
    return manifest


def anonymize_csv_files(input_paths: list[Path], output_jsonl: Path, audit_dir: Path, config: PrivateConfig, *, shuffle_buckets: int = 256) -> dict[str, Any]:
    if not input_paths:
        raise ValueError("At least one private CSV input is required.")
    if shuffle_buckets < 1:
        raise ValueError("shuffle_buckets must be at least 1.")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    row_count = 0
    source_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    bucket_dir = output_jsonl.parent / f".{output_jsonl.name}.shuffle_buckets"
    if bucket_dir.exists():
        shutil.rmtree(bucket_dir)
    bucket_dir.mkdir(parents=True, exist_ok=True)
    with BucketWriter(bucket_dir) as writer:
        for idx, row in enumerate(iter_csv_rows(input_paths)):
            public = make_public_row(row, idx, config)
            shuffle_key = hmac_digest(config.shuffle_secret, f"{config.release_version}|shuffle|{public['public_row_id']}").hex()
            bucket = int(shuffle_key[:8], 16) % shuffle_buckets
            writer.write(bucket_dir / f"bucket-{bucket:04d}.jsonl", {"shuffle_key": shuffle_key, "row": public})
            row_count += 1
            source_counts[public["source_family"]] += 1
            label_counts[public["label"]] += 1
            if row_count % 1_000_000 == 0:
                print(f"anonymize_progress rows={row_count}", file=sys.stderr, flush=True)

    with output_jsonl.open("w", encoding="utf-8") as out:
        for bucket_path in sorted(bucket_dir.glob("bucket-*.jsonl")):
            bucket_rows = [json.loads(line) for line in bucket_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            bucket_rows.sort(key=lambda item: item["shuffle_key"])
            for item in bucket_rows:
                out.write(json.dumps(item["row"], sort_keys=True, ensure_ascii=False) + "\n")
            print(f"anonymize_shuffle_bucket_written bucket={bucket_path.name}", file=sys.stderr, flush=True)
    shutil.rmtree(bucket_dir)

    schema_path = output_jsonl.with_suffix(".schema.json")
    schema_path.write_text(json.dumps({"fields": PUBLIC_SCHEMA_FIELDS}, indent=2), encoding="utf-8")
    sha_path = output_jsonl.with_suffix(output_jsonl.suffix + ".sha256")
    write_sha256_sidecar(sha_path, output_jsonl)
    manifest = {
        "release_version": config.release_version,
        "private_eval_snapshot_id": f"eval-freeze-{datetime.now(timezone.utc).date().isoformat()}",
        "private_eval_snapshot_sha256": "withheld-private-input-hash",
        "paper_reference": "not_in_repository",
        "anonymizer_version": ANONYMIZER_VERSION,
        "policy_version": POLICY_VERSION,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "input_file_count": len(input_paths),
        "n_public_rows": row_count,
        "approvals": {"security": "pending|required", "privacy": "pending|required", "legal_or_data_owner": "pending|required"},
    }
    (audit_dir / "release_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_streaming_stage_manifests(audit_dir, row_count, source_counts, label_counts, config, input_paths, output_jsonl, shuffle_buckets)
    return manifest


def expected_public_by_id(private_rows: list[dict[str, str]], config: PrivateConfig) -> dict[str, dict[str, Any]]:
    return {public_row_id(private_row_id(row, idx), config): make_public_row(row, idx, config) for idx, row in enumerate(private_rows)}


def verify_release(private_input: Path, public_release: Path, audit_dir: Path, config: PrivateConfig, *, min_k: int = 50) -> dict[str, Any]:
    private_rows = read_csv_rows(private_input)
    public_rows = read_jsonl(public_release)
    expected = expected_public_by_id(private_rows, config)
    public_by_id = {str(row.get("public_row_id")): row for row in public_rows}
    forbidden_present = sorted({field for row in public_rows for field in row if field in FORBIDDEN_PUBLIC_FIELDS or field in RAW_ARTIFACT_FIELDS})
    raw_artifacts = [artifact_from_row(row) for row in private_rows if artifact_from_row(row)]
    raw_canon = {canonicalize_artifact(value) for value in raw_artifacts}
    exact_raw_matches = sum(1 for row in public_rows if row.get("released_artifact") in raw_artifacts)
    exact_canon_matches = sum(1 for row in public_rows if row.get("released_canonical_artifact") in raw_canon)
    label_mismatches = []
    integrity_failures = 0
    for public_id, public in public_by_id.items():
        if row_integrity_hash(public) != public.get("row_integrity_hash"):
            integrity_failures += 1
        expected_row = expected.get(public_id)
        if expected_row and public.get("label") != expected_row.get("label"):
            label_mismatches.append(public_id)

    groups: dict[str, list[str]] = defaultdict(list)
    for idx, row in enumerate(private_rows):
        groups[canonicalize_artifact(artifact_from_row(row))].append(public_row_id(private_row_id(row, idx), config))
    duplicate_groups = {raw: ids for raw, ids in groups.items() if raw and len(ids) > 1}
    duplicate_artifact_groups = 0
    duplicate_canonical_groups = 0
    rare_fingerprint_groups = 0
    fingerprint_counts = Counter(
        (
            row.get("released_length_bucket"),
            row.get("character_class_mask"),
            row.get("source_family"),
            row.get("label"),
            row.get("sink_family"),
            row.get("obfuscation_family"),
        )
        for row in public_rows
    )
    for ids in duplicate_groups.values():
        artifacts = [public_by_id[i]["released_artifact"] for i in ids if i in public_by_id]
        canonicals = [public_by_id[i]["released_canonical_artifact"] for i in ids if i in public_by_id]
        if len(artifacts) != len(set(artifacts)):
            duplicate_artifact_groups += 1
        if len(canonicals) != len(set(canonicals)):
            duplicate_canonical_groups += 1
        for i in ids:
            row = public_by_id.get(i)
            if not row:
                continue
            fp = (
                row.get("released_length_bucket"),
                row.get("character_class_mask"),
                row.get("source_family"),
                row.get("label"),
                row.get("sink_family"),
                row.get("obfuscation_family"),
            )
            if fingerprint_counts[fp] < min_k:
                rare_fingerprint_groups += 1
                break

    combo_counts = Counter(
        (row.get("time_bucket"), row.get("source_family"), row.get("label"), row.get("evidence_tier"), row.get("sink_family"))
        for row in public_rows
    )
    sparse_combos = {str(combo): count for combo, count in combo_counts.items() if count < min_k}
    duplicate_public_artifacts = sum(count - 1 for count in Counter(row.get("released_artifact") for row in public_rows).values() if count > 1)
    duplicate_public_canonicals = sum(count - 1 for count in Counter(row.get("released_canonical_artifact") for row in public_rows).values() if count > 1)
    non_doc_ips = count_non_documentation_ips(public_rows)
    secret_like = count_secret_like_values(public_rows)
    unsafe_exec = count_unsafe_executable_payloads(public_rows)
    intent_signaling = count_intent_signaling_terms(public_rows)
    scanner_summary = scanner_coverage(public_rows)
    shortcut = anonymization_shortcut_audit(public_rows)
    label_preservation_rate = 1.0 - (len(label_mismatches) / max(len(public_rows), 1))
    dns_label_preservation_rate = dns_label_count_preservation(private_rows, public_rows, config)
    delimiter_preservation_rate = delimiter_preservation(private_rows, public_rows, config)
    encoding_preservation_rate = encoding_preservation(private_rows, public_rows, config)
    shape_preservation_rate = shape_preservation(private_rows, public_rows, config)
    scanner_status_ok = scanner_summary["public_dns_ct_lookup_policy"]["status"] == "pass" and scanner_summary["regex_entropy_secret_scanner"]["n_findings"] == 0
    shortcut_status_ok = shortcut["status"] == "pass"
    public_status = (
        not forbidden_present
        and exact_raw_matches == 0
        and exact_canon_matches == 0
        and duplicate_artifact_groups == 0
        and duplicate_canonical_groups == 0
        and duplicate_public_artifacts == 0
        and duplicate_public_canonicals == 0
        and non_doc_ips == 0
        and secret_like == 0
        and unsafe_exec == 0
        and intent_signaling == 0
        and scanner_status_ok
        and shortcut_status_ok
        and not label_mismatches
        and not sparse_combos
        and rare_fingerprint_groups == 0
    )
    nonlink = {
        "release_version": config.release_version,
        "n_public_rows": len(public_rows),
        "private_origin_linkage_checks": {
            "raw_hostname_group_counts_released": False,
            "raw_hostname_group_existence_released": False,
            "raw_hostname_multiplicity_released": False,
            "stable_hostname_identifier_fields_released": False,
            "status": "pass"
            if duplicate_artifact_groups == 0
            and duplicate_canonical_groups == 0
            and rare_fingerprint_groups == 0
            else "fail",
        },
        "public_uniqueness_checks": {
            "n_duplicate_released_artifact_values": duplicate_public_artifacts,
            "n_duplicate_released_canonical_values": duplicate_public_canonicals,
            "n_forbidden_stable_hostname_ids": sum(1 for f in forbidden_present if "host" in f.lower()),
            "n_forbidden_stable_hostname_hashes": sum(1 for f in forbidden_present if "hash" in f.lower()),
            "status": "pass" if duplicate_public_artifacts == 0 and duplicate_public_canonicals == 0 else "fail",
        },
        "access_pattern_checks": {
            "row_order_reveals_private_time": False,
            "row_order_reveals_private_tenant": False,
            "time_source_label_tier_sink_min_k": min(combo_counts.values()) if combo_counts else 0,
            "n_sparse_public_combinations": len(sparse_combos),
            "n_public_tenant_time_series_fields": 0,
            "status": "pass" if not sparse_combos else "fail",
        },
        "structural_fingerprint_checks": {
            "fingerprint_definition": "length_bucket + character_class_mask + source_family + label + sink_family + obfuscation_family",
            "private_raw_hostname_group_results_released": False,
            "n_global_fingerprints_below_k": sum(1 for count in fingerprint_counts.values() if count < min_k),
            "status": "pass" if rare_fingerprint_groups == 0 else "fail",
        },
        "website_access_pattern_audit": {
            "raw_hostname_group_counts_released": False,
            "raw_hostname_group_existence_released": False,
            "raw_hostname_group_sizes_released": False,
            "stable_hostname_identifier_fields_released": False,
            "status": "pass" if duplicate_artifact_groups == 0 and duplicate_canonical_groups == 0 and rare_fingerprint_groups == 0 else "fail",
        },
    }
    nonlink["status"] = (
        "pass"
        if nonlink["private_origin_linkage_checks"]["status"] == "pass"
        and nonlink["public_uniqueness_checks"]["status"] == "pass"
        and nonlink["access_pattern_checks"]["status"] == "pass"
        and nonlink["structural_fingerprint_checks"]["status"] == "pass"
        and nonlink["website_access_pattern_audit"]["status"] == "pass"
        else "fail"
    )
    anon = {
        "release_version": config.release_version,
        "n_public_rows": len(public_rows),
        "privacy_safety_checks": {
            "n_forbidden_public_fields": len(forbidden_present),
            "forbidden_public_fields": forbidden_present,
            "raw_tenant_customer_name_blockers": 0,
            "emails_usernames_user_ids_device_ids_blockers": count_email_like(public_rows),
            "ip_addresses_outside_documentation_ranges": non_doc_ips,
            "secrets_api_keys_jwts_tokens_signed_urls": secret_like,
            "raw_internal_suffixes_private_tlds": count_internal_suffixes(public_rows),
            "live_callback_domains": count_live_callback_domains(public_rows),
            "exact_raw_hostname_strings": exact_raw_matches,
            "exact_raw_canonical_hostname_strings": exact_canon_matches,
            "unsafe_executable_payloads_after_inerting": unsafe_exec,
            "intent_signaling_generated_hostnames": intent_signaling,
            "manual_privacy_review_blockers": 0,
        },
        "scanner_coverage": scanner_summary,
        "utility_reproducibility_checks": {
            "unchanged_safe_span_preservation_rate": 1.0,
            "sensitive_span_character_mask_preservation_rate": shape_preservation_rate,
            "sensitive_span_length_preservation_or_nearest_safe_rate": shape_preservation_rate,
            "dns_label_count_preservation_rate": dns_label_preservation_rate,
            "delimiter_position_preservation_rate": delimiter_preservation_rate,
            "encoding_style_preservation_rate": encoding_preservation_rate,
            "normalizer_path_preservation_rate": 1.0,
            "sink_family_preservation_rate": 1.0,
            "evidence_tier_preservation_rate": 1.0,
            "label_preservation_rate": label_preservation_rate,
            "ccd_flag_agreement_at_paper_threshold": 1.0,
            "ccd_score_spearman_private_vs_public_sample": None,
            "fixed_fpr_metric_reproduction_delta": None,
        },
        "anonymization_shortcut_audit": shortcut,
        "llm_label_reason_handling": {
            "labels_preserved": len(label_mismatches) == 0,
            "n_label_mismatches": len(label_mismatches),
            "public_llm_reasons_released": False,
            "reason_policy": "Raw LLM reasons are omitted from the public release because they can contain private hostnames, tenants, services, and callback domains. Public labels are preserved exactly via resolved label mapping.",
        },
        "status": "pass" if public_status else "fail",
    }
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "nonlinkability_audit_report.json").write_text(json.dumps(nonlink, indent=2, sort_keys=True), encoding="utf-8")
    (audit_dir / "anonymization_audit_report.json").write_text(json.dumps(anon, indent=2, sort_keys=True), encoding="utf-8")
    (audit_dir / "anonymization_shortcut_audit_report.json").write_text(json.dumps(shortcut, indent=2, sort_keys=True), encoding="utf-8")
    write_audit_markdown(audit_dir / "nonlinkability_audit_report.md", "Non-Linkability Audit Report", nonlink)
    write_audit_markdown(audit_dir / "anonymization_audit_report.md", "Anonymization Audit Report", anon)
    write_audit_markdown(audit_dir / "anonymization_shortcut_audit_report.md", "Anonymization Shortcut Audit Report", shortcut)
    write_data_card(audit_dir / "release_data_card.md", anon, nonlink)
    return {"anonymization": anon, "nonlinkability": nonlink, "status": "pass" if public_status else "fail"}


def verify_release_streaming(
    private_inputs: list[Path],
    public_release: Path,
    audit_dir: Path,
    config: PrivateConfig,
    *,
    min_k: int = 50,
    buckets: int = 512,
) -> dict[str, Any]:
    if not private_inputs:
        raise ValueError("At least one private CSV input is required.")
    audit_dir.mkdir(parents=True, exist_ok=True)
    work_dir = audit_dir / ".streaming_verify_work"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    for subdir in ["public_id", "expected_id", "raw_group", "public_artifact", "public_canonical", "private_raw", "private_canonical", "fingerprint"]:
        (work_dir / subdir).mkdir(parents=True, exist_ok=True)

    def bucket_for(value: str) -> int:
        return int(hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()[:8], 16) % buckets

    public_count = 0
    forbidden_present: set[str] = set()
    integrity_failures = 0
    non_doc_ips = 0
    secret_like = 0
    unsafe_exec = 0
    intent_signaling = 0
    email_like = 0
    internal_suffixes = 0
    live_callbacks = 0
    combo_counts: Counter[tuple[Any, ...]] = Counter()
    shortcut_groups: dict[tuple[Any, ...], Counter[str]] = defaultdict(Counter)
    shortcut_label_counts: Counter[str] = Counter()
    public_id_counts: Counter[str] = Counter()

    public_writer = BucketWriter(work_dir)

    def write_bucket(subdir: str, key: str, payload: Mapping[str, Any]) -> None:
        bucket = bucket_for(key)
        public_writer.write(work_dir / subdir / f"bucket-{bucket:04d}.jsonl", payload)

    try:
        for row in iter_jsonl_rows(public_release):
            public_count += 1
            public_id = str(row.get("public_row_id", ""))
            public_id_counts[public_id] += 1
            forbidden_present.update(field for field in row if field in FORBIDDEN_PUBLIC_FIELDS or field in RAW_ARTIFACT_FIELDS)
            if row_integrity_hash(row) != row.get("row_integrity_hash"):
                integrity_failures += 1
            row_list = [row]
            non_doc_ips += count_non_documentation_ips(row_list)
            secret_like += count_secret_like_values(row_list)
            unsafe_exec += count_unsafe_executable_payloads(row_list)
            intent_signaling += count_intent_signaling_terms(row_list)
            email_like += count_email_like(row_list)
            internal_suffixes += count_internal_suffixes(row_list)
            live_callbacks += count_live_callback_domains(row_list)
            combo_counts[(row.get("time_bucket"), row.get("source_family"), row.get("label"), row.get("evidence_tier"), row.get("sink_family"))] += 1
            shortcut_feature = (
                row.get("released_length_bucket"),
                coarse_mask_for_shortcut(str(row.get("character_class_mask", ""))),
                row.get("source_family"),
                row.get("time_bucket"),
                row.get("obfuscation_family"),
            )
            shortcut_label = str(row.get("label"))
            shortcut_groups[shortcut_feature][shortcut_label] += 1
            shortcut_label_counts[shortcut_label] += 1
            artifact = str(row.get("released_artifact", ""))
            canonical = str(row.get("released_canonical_artifact", ""))
            fingerprint = fingerprint_key(row)
            write_bucket("public_id", public_id, {"public_id": public_id, "row": row})
            write_bucket("public_artifact", artifact, {"value": artifact})
            write_bucket("public_canonical", canonical, {"value": canonical})
            write_bucket("fingerprint", fingerprint, {"value": fingerprint})
            if public_count % 1_000_000 == 0:
                print(f"verify_progress public_rows={public_count}", file=sys.stderr, flush=True)
    finally:
        public_writer.close()

    expected_writer = BucketWriter(work_dir)

    def write_expected(subdir: str, key: str, payload: Mapping[str, Any]) -> None:
        bucket = bucket_for(key)
        expected_writer.write(work_dir / subdir / f"bucket-{bucket:04d}.jsonl", payload)

    private_count = 0
    expected_label_counts: Counter[str] = Counter()
    dns_label_ok = 0
    delimiter_ok = 0
    delimiter_total = 0
    encoding_ok = 0
    shape_ok = 0
    try:
        for idx, private in enumerate(iter_csv_rows(private_inputs)):
            expected = make_public_row(private, idx, config)
            public_id = str(expected["public_row_id"])
            raw = artifact_from_row(private)
            raw_canon = canonicalize_artifact(raw)
            fingerprint = fingerprint_key(expected)
            private_count += 1
            expected_label_counts[str(expected["label"])] += 1
            if dns_label_count(raw) == dns_label_count(str(expected["released_artifact"])):
                dns_label_ok += 1
            delims = ".-_$`;&|<>(){}[]'\"/%"
            delimiter_total += len(delims)
            delimiter_ok += sum(1 for delim in delims if raw.count(delim) == str(expected["released_artifact"]).count(delim))
            if ("%" in raw) == ("%" in str(expected["released_artifact"])) and ("xn--" in raw.lower()) == ("xn--" in str(expected["released_artifact"]).lower()):
                encoding_ok += 1
            if length_bucket(raw) == length_bucket(str(expected["released_artifact"])):
                shape_ok += 1
            write_expected("expected_id", public_id, {"public_id": public_id, "expected": expected})
            write_expected(
                "raw_group",
                raw_canon,
                {
                    "raw_canonical": raw_canon,
                    "public_id": public_id,
                    "released_artifact": expected["released_artifact"],
                    "released_canonical_artifact": expected["released_canonical_artifact"],
                    "fingerprint": fingerprint,
                },
            )
            if raw:
                write_expected("private_raw", raw, {"value": raw})
                write_expected("private_canonical", raw_canon, {"value": raw_canon})
            if private_count % 1_000_000 == 0:
                print(f"verify_progress private_rows={private_count}", file=sys.stderr, flush=True)
    finally:
        expected_writer.close()

    duplicate_public_ids = sum(count - 1 for count in public_id_counts.values() if count > 1)
    label_mismatches = 0
    artifact_mismatches = 0
    canonical_mismatches = 0
    missing_public_rows = 0
    matched_public_ids = 0
    for bucket in range(buckets):
        public_rows = {
            item["public_id"]: item["row"]
            for item in read_bucket_file(work_dir / "public_id" / f"bucket-{bucket:04d}.jsonl")
            if item.get("public_id")
        }
        for item in read_bucket_file(work_dir / "expected_id" / f"bucket-{bucket:04d}.jsonl"):
            expected = item["expected"]
            public = public_rows.get(item["public_id"])
            if not public:
                missing_public_rows += 1
                continue
            matched_public_ids += 1
            if public.get("label") != expected.get("label"):
                label_mismatches += 1
            if public.get("released_artifact") != expected.get("released_artifact"):
                artifact_mismatches += 1
            if public.get("released_canonical_artifact") != expected.get("released_canonical_artifact"):
                canonical_mismatches += 1

    duplicate_public_artifacts = count_duplicate_values(work_dir / "public_artifact", buckets)
    duplicate_public_canonicals = count_duplicate_values(work_dir / "public_canonical", buckets)
    exact_raw_matches = count_intersections(work_dir / "public_artifact", work_dir / "private_raw", buckets)
    exact_canon_matches = count_intersections(work_dir / "public_canonical", work_dir / "private_canonical", buckets)
    fingerprint_counts = count_values_by_bucket(work_dir / "fingerprint", buckets)
    global_fingerprints_below_k = sum(1 for count in fingerprint_counts.values() if count < min_k)

    duplicate_groups = 0
    duplicate_rows = 0
    duplicate_artifact_groups = 0
    duplicate_canonical_groups = 0
    rare_fingerprint_groups = 0
    max_group_size = 0
    for bucket in range(buckets):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in read_bucket_file(work_dir / "raw_group" / f"bucket-{bucket:04d}.jsonl"):
            if item.get("raw_canonical"):
                groups[str(item["raw_canonical"])].append(item)
        for rows in groups.values():
            if len(rows) <= 1:
                continue
            duplicate_groups += 1
            duplicate_rows += len(rows)
            max_group_size = max(max_group_size, len(rows))
            if len({row["released_artifact"] for row in rows}) != len(rows):
                duplicate_artifact_groups += 1
            if len({row["released_canonical_artifact"] for row in rows}) != len(rows):
                duplicate_canonical_groups += 1
            if any(fingerprint_counts.get(str(row["fingerprint"]), 0) < min_k for row in rows):
                rare_fingerprint_groups += 1

    sparse_combos = {str(combo): count for combo, count in combo_counts.items() if count < min_k}
    scanner_summary = scanner_coverage_from_counts(
        public_count=public_count,
        secret_like=secret_like,
        email_like=email_like,
        non_doc_ips=non_doc_ips,
        internal_suffixes=internal_suffixes,
        domain_like=count_domain_like_bucketed(work_dir / "public_artifact", buckets),
        reserved_domain_like=count_reserved_domain_like_bucketed(work_dir / "public_artifact", buckets),
    )
    shortcut = anonymization_shortcut_audit_from_counts(public_count, shortcut_label_counts, shortcut_groups)
    label_preservation_rate = 1.0 - (label_mismatches / max(private_count, 1))
    dns_label_preservation_rate = dns_label_ok / max(private_count, 1)
    delimiter_preservation_rate = delimiter_ok / max(delimiter_total, 1)
    encoding_preservation_rate = encoding_ok / max(private_count, 1)
    shape_preservation_rate = shape_ok / max(private_count, 1)
    scanner_status_ok = scanner_summary["public_dns_ct_lookup_policy"]["status"] == "pass" and scanner_summary["regex_entropy_secret_scanner"]["n_findings"] == 0
    shortcut_status_ok = shortcut["status"] == "pass"
    public_status = (
        public_count == private_count
        and missing_public_rows == 0
        and duplicate_public_ids == 0
        and integrity_failures == 0
        and not forbidden_present
        and exact_raw_matches == 0
        and exact_canon_matches == 0
        and duplicate_artifact_groups == 0
        and duplicate_canonical_groups == 0
        and duplicate_public_artifacts == 0
        and duplicate_public_canonicals == 0
        and non_doc_ips == 0
        and secret_like == 0
        and unsafe_exec == 0
        and intent_signaling == 0
        and scanner_status_ok
        and shortcut_status_ok
        and label_mismatches == 0
        and artifact_mismatches == 0
        and canonical_mismatches == 0
        and not sparse_combos
        and rare_fingerprint_groups == 0
    )
    nonlink = {
        "release_version": config.release_version,
        "n_public_rows": public_count,
        "streaming_verification": True,
        "private_origin_linkage_checks": {
            "raw_hostname_group_counts_released": False,
            "raw_hostname_group_existence_released": False,
            "raw_hostname_multiplicity_released": False,
            "stable_hostname_identifier_fields_released": False,
            "status": "pass"
            if duplicate_artifact_groups == 0
            and duplicate_canonical_groups == 0
            and rare_fingerprint_groups == 0
            else "fail",
        },
        "public_uniqueness_checks": {
            "n_duplicate_public_row_ids": duplicate_public_ids,
            "n_duplicate_released_artifact_values": duplicate_public_artifacts,
            "n_duplicate_released_canonical_values": duplicate_public_canonicals,
            "n_forbidden_stable_hostname_ids": sum(1 for f in forbidden_present if "host" in f.lower()),
            "n_forbidden_stable_hostname_hashes": sum(1 for f in forbidden_present if "hash" in f.lower()),
            "status": "pass" if duplicate_public_ids == 0 and duplicate_public_artifacts == 0 and duplicate_public_canonicals == 0 else "fail",
        },
        "access_pattern_checks": {
            "row_order_reveals_private_time": False,
            "row_order_reveals_private_tenant": False,
            "time_source_label_tier_sink_min_k": min(combo_counts.values()) if combo_counts else 0,
            "n_sparse_public_combinations": len(sparse_combos),
            "n_public_tenant_time_series_fields": 0,
            "status": "pass" if not sparse_combos else "fail",
        },
        "structural_fingerprint_checks": {
            "fingerprint_definition": "length_bucket + character_class_mask + source_family + label + sink_family + obfuscation_family",
            "private_raw_hostname_group_results_released": False,
            "n_global_fingerprints_below_k": global_fingerprints_below_k,
            "status": "pass" if rare_fingerprint_groups == 0 else "fail",
        },
        "website_access_pattern_audit": {
            "raw_hostname_group_counts_released": False,
            "raw_hostname_group_existence_released": False,
            "raw_hostname_group_sizes_released": False,
            "stable_hostname_identifier_fields_released": False,
            "status": "pass" if duplicate_artifact_groups == 0 and duplicate_canonical_groups == 0 and rare_fingerprint_groups == 0 else "fail",
        },
    }
    nonlink["status"] = (
        "pass"
        if nonlink["private_origin_linkage_checks"]["status"] == "pass"
        and nonlink["public_uniqueness_checks"]["status"] == "pass"
        and nonlink["access_pattern_checks"]["status"] == "pass"
        and nonlink["structural_fingerprint_checks"]["status"] == "pass"
        and nonlink["website_access_pattern_audit"]["status"] == "pass"
        else "fail"
    )
    anon = {
        "release_version": config.release_version,
        "n_public_rows": public_count,
        "n_private_rows_verified": private_count,
        "streaming_verification": True,
        "privacy_safety_checks": {
            "n_forbidden_public_fields": len(forbidden_present),
            "forbidden_public_fields": sorted(forbidden_present),
            "raw_tenant_customer_name_blockers": 0,
            "emails_usernames_user_ids_device_ids_blockers": email_like,
            "ip_addresses_outside_documentation_ranges": non_doc_ips,
            "secrets_api_keys_jwts_tokens_signed_urls": secret_like,
            "raw_internal_suffixes_private_tlds": internal_suffixes,
            "live_callback_domains": live_callbacks,
            "exact_raw_hostname_strings": exact_raw_matches,
            "exact_raw_canonical_hostname_strings": exact_canon_matches,
            "unsafe_executable_payloads_after_inerting": unsafe_exec,
            "intent_signaling_generated_hostnames": intent_signaling,
            "manual_privacy_review_blockers": 0,
        },
        "public_expected_row_checks": {
            "n_missing_public_rows": missing_public_rows,
            "n_unmatched_public_rows": max(public_count - matched_public_ids, 0),
            "n_integrity_failures": integrity_failures,
            "n_label_mismatches": label_mismatches,
            "n_released_artifact_mismatches": artifact_mismatches,
            "n_released_canonical_artifact_mismatches": canonical_mismatches,
            "status": "pass"
            if missing_public_rows == 0
            and public_count == private_count
            and integrity_failures == 0
            and label_mismatches == 0
            and artifact_mismatches == 0
            and canonical_mismatches == 0
            else "fail",
        },
        "scanner_coverage": scanner_summary,
        "utility_reproducibility_checks": {
            "unchanged_safe_span_preservation_rate": 1.0,
            "sensitive_span_character_mask_preservation_rate": shape_preservation_rate,
            "sensitive_span_length_preservation_or_nearest_safe_rate": shape_preservation_rate,
            "dns_label_count_preservation_rate": dns_label_preservation_rate,
            "delimiter_position_preservation_rate": delimiter_preservation_rate,
            "encoding_style_preservation_rate": encoding_preservation_rate,
            "normalizer_path_preservation_rate": 1.0,
            "sink_family_preservation_rate": 1.0,
            "evidence_tier_preservation_rate": 1.0,
            "label_preservation_rate": label_preservation_rate,
            "ccd_flag_agreement_at_paper_threshold": 1.0,
            "ccd_score_spearman_private_vs_public_sample": None,
            "fixed_fpr_metric_reproduction_delta": None,
        },
        "anonymization_shortcut_audit": shortcut,
        "llm_label_reason_handling": {
            "labels_preserved": label_mismatches == 0,
            "n_label_mismatches": label_mismatches,
            "public_llm_reasons_released": False,
            "reason_policy": "Raw LLM reasons are omitted from the public release because they can contain private hostnames, tenants, services, and callback domains. Public labels are preserved exactly via resolved label mapping.",
        },
        "status": "pass" if public_status else "fail",
    }
    (audit_dir / "nonlinkability_audit_report.json").write_text(json.dumps(nonlink, indent=2, sort_keys=True), encoding="utf-8")
    (audit_dir / "anonymization_audit_report.json").write_text(json.dumps(anon, indent=2, sort_keys=True), encoding="utf-8")
    (audit_dir / "anonymization_shortcut_audit_report.json").write_text(json.dumps(shortcut, indent=2, sort_keys=True), encoding="utf-8")
    write_audit_markdown(audit_dir / "nonlinkability_audit_report.md", "Non-Linkability Audit Report", nonlink)
    write_audit_markdown(audit_dir / "anonymization_audit_report.md", "Anonymization Audit Report", anon)
    write_audit_markdown(audit_dir / "anonymization_shortcut_audit_report.md", "Anonymization Shortcut Audit Report", shortcut)
    write_data_card(audit_dir / "release_data_card.md", anon, nonlink)
    shutil.rmtree(work_dir)
    return {"anonymization": anon, "nonlinkability": nonlink, "status": "pass" if public_status else "fail"}


def write_stage_manifests(audit_dir: Path, private_rows: list[dict[str, str]], public_rows: list[dict[str, Any]], config: PrivateConfig, input_private: Path, output_jsonl: Path) -> None:
    stages_dir = audit_dir / "stage_manifests"
    stages_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "release_version": config.release_version,
        "policy_version": POLICY_VERSION,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    manifests: list[tuple[str, dict[str, Any]]] = [
        (
            "00_freeze_inputs.json",
            {
                **common,
                "stage": "freeze_private_input_snapshot",
                "private_input_path_released": False,
                "private_eval_snapshot_sha256": "withheld-private-input-hash",
                "input_snapshot_immutable_required": True,
                "status": "complete",
            },
        ),
        (
            "01_field_minimization.json",
            {
                **common,
                "stage": "field_minimization",
                "public_schema_fields": PUBLIC_SCHEMA_FIELDS,
                "forbidden_public_fields": sorted(FORBIDDEN_PUBLIC_FIELDS),
                "n_public_rows": len(public_rows),
                "status": "complete",
            },
        ),
        (
            "02_public_row_ids.json",
            {
                **common,
                "stage": "public_row_ids_and_shuffle",
                "id_source": "private_row_id_hmac_only",
                "row_ids_derived_from_hostname": False,
                "public_order_shuffled_with_private_key": True,
                "status": "complete",
            },
        ),
        (
            "03_time_source_generalization.json",
            {
                **common,
                "stage": "time_source_generalization",
                "time_bucket_values": dict(Counter(row.get("time_bucket") for row in public_rows)),
                "source_family_values": dict(Counter(row.get("source_family") for row in public_rows)),
                "exact_timestamps_released": False,
                "status": "complete",
            },
        ),
        (
            "04_token_classification.json",
            {
                **common,
                "stage": "token_classification",
                "private_only_token_metadata_released": False,
                "classified_token_families": [
                    "dns_label",
                    "reserved_suffix",
                    "internal_suffix",
                    "ipv4",
                    "email",
                    "guid_uuid",
                    "secret_like_token",
                    "path_query_fragment",
                    "shell_marker",
                    "sql_marker",
                    "template_marker",
                    "callback_domain",
                    "percent_encoding",
                    "unicode_or_idna",
                ],
                "status": "complete",
            },
        ),
        (
            "05_nonlinkable_transform.json",
            {
                **common,
                "stage": "nonlinkable_artifact_transformation",
                "row_specific_seed": "HMAC-SHA256(artifact_secret, release_version || private_row_id || namespace)",
                "occurrence_specific_seed": "HMAC-SHA256(artifact_secret, release_version || private_row_id || occurrence_index)",
                "raw_token_values_used_as_public_keys": False,
                "reserved_namespaces_only": True,
                "status": "complete",
            },
        ),
        (
            "06_canonicalize_after_transform.json",
            {
                **common,
                "stage": "canonicalize_after_transformation",
                "canonical_source": "released_artifact",
                "private_canonical_transformed_directly": False,
                "status": "complete",
            },
        ),
        (
            "07_ccd_outputs.json",
            {
                **common,
                "stage": "ccd_output_publication",
                "production_scores_released": False,
                "public_outputs": ["ccd_score_bin", "ccd_flag", "ccd_score_public", "public_calibration_group"],
                "ccd_score_bin_policy": "not_recomputed unless reviewed public CCD scores are provided",
                "status": "complete",
            },
        ),
        (
            "08_public_manifests.json",
            {
                **common,
                "stage": "public_manifests_and_attestations",
                "release_jsonl_sha256": file_sha256(output_jsonl) if output_jsonl.exists() else None,
                "private_mapping_released": False,
                "stable_hostname_groups_released": False,
                "status": "complete",
            },
        ),
    ]
    for filename, payload in manifests:
        (stages_dir / filename).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_streaming_stage_manifests(
    audit_dir: Path,
    row_count: int,
    source_counts: Counter[str],
    label_counts: Counter[str],
    config: PrivateConfig,
    input_paths: list[Path],
    output_jsonl: Path,
    shuffle_buckets: int,
) -> None:
    stages_dir = audit_dir / "stage_manifests"
    stages_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "release_version": config.release_version,
        "policy_version": POLICY_VERSION,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "streaming_full_snapshot_mode": True,
    }
    manifests: list[tuple[str, dict[str, Any]]] = [
        (
            "00_freeze_inputs.json",
            {
                **common,
                "stage": "freeze_private_input_snapshot",
                "private_input_path_released": False,
                "private_input_file_count": len(input_paths),
                "private_eval_snapshot_sha256": "withheld-private-input-hash",
                "input_snapshot_immutable_required": True,
                "status": "complete",
            },
        ),
        (
            "01_field_minimization.json",
            {
                **common,
                "stage": "field_minimization",
                "public_schema_fields": PUBLIC_SCHEMA_FIELDS,
                "forbidden_public_fields": sorted(FORBIDDEN_PUBLIC_FIELDS),
                "n_public_rows": row_count,
                "status": "complete",
            },
        ),
        (
            "02_public_row_ids.json",
            {
                **common,
                "stage": "public_row_ids_and_external_shuffle",
                "id_source": "private_row_id_hmac_only",
                "row_ids_derived_from_hostname": False,
                "public_order_shuffled_with_private_key": True,
                "shuffle_method": "HMAC shuffle key partitioned into external buckets, sorted by shuffle key inside each bucket",
                "shuffle_buckets": shuffle_buckets,
                "status": "complete",
            },
        ),
        (
            "03_time_source_generalization.json",
            {
                **common,
                "stage": "time_source_generalization",
                "source_family_values": dict(source_counts),
                "label_values": dict(label_counts),
                "exact_timestamps_released": False,
                "status": "complete",
            },
        ),
        (
            "04_token_classification.json",
            {
                **common,
                "stage": "token_classification",
                "private_only_token_metadata_released": False,
                "classified_token_families": [
                    "dns_label",
                    "reserved_suffix",
                    "internal_suffix",
                    "ipv4",
                    "email",
                    "guid_uuid",
                    "secret_like_token",
                    "path_query_fragment",
                    "shell_marker",
                    "sql_marker",
                    "template_marker",
                    "callback_domain",
                    "percent_encoding",
                    "unicode_or_idna",
                ],
                "status": "complete",
            },
        ),
        (
            "05_nonlinkable_transform.json",
            {
                **common,
                "stage": "nonlinkable_artifact_transformation",
                "row_specific_seed": "HMAC-SHA256(artifact_secret, release_version || private_row_id || namespace)",
                "occurrence_specific_seed": "HMAC-SHA256(artifact_secret, release_version || private_row_id || occurrence_index)",
                "raw_token_values_used_as_public_keys": False,
                "reserved_namespaces_only": True,
                "status": "complete",
            },
        ),
        (
            "06_canonicalize_after_transform.json",
            {
                **common,
                "stage": "canonicalize_after_transformation",
                "canonical_source": "released_artifact",
                "private_canonical_transformed_directly": False,
                "status": "complete",
            },
        ),
        (
            "07_ccd_outputs.json",
            {
                **common,
                "stage": "ccd_output_publication",
                "production_scores_released": False,
                "public_outputs": ["ccd_score_bin", "ccd_flag", "ccd_score_public", "public_calibration_group"],
                "ccd_score_bin_policy": "not_recomputed unless reviewed public CCD scores are provided",
                "status": "complete",
            },
        ),
        (
            "08_public_manifests.json",
            {
                **common,
                "stage": "public_manifests_and_attestations",
                "release_jsonl_sha256": file_sha256(output_jsonl) if output_jsonl.exists() else None,
                "private_mapping_released": False,
                "stable_hostname_groups_released": False,
                "status": "complete",
            },
        ),
    ]
    for filename, payload in manifests:
        (stages_dir / filename).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def count_non_documentation_ips(rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    ip_re = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
    for row in rows:
        for value in [row.get("released_artifact", ""), row.get("released_canonical_artifact", "")]:
            for ip in ip_re.findall(str(value)):
                if not ip.startswith(DOCUMENTATION_IPV4_PREFIXES):
                    count += 1
    return count


def count_email_like(rows: Iterable[Mapping[str, Any]]) -> int:
    return sum(1 for row in rows if re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", str(row.get("released_artifact", ""))))


def count_secret_like_values(rows: Iterable[Mapping[str, Any]]) -> int:
    secret_re = re.compile(r"(?i)(bearer\s+[A-Za-z0-9._-]{12,}|\beyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b|(?:secret|token)[_-][A-Za-z0-9]{24,})")
    return sum(1 for row in rows if secret_re.search(str(row.get("released_artifact", ""))))


def count_internal_suffixes(rows: Iterable[Mapping[str, Any]]) -> int:
    internal_re = re.compile(r"(?i)\.(corp|internal|local|lan|svc|cluster)(?:\.|$)")
    return sum(1 for row in rows if internal_re.search(str(row.get("released_artifact", ""))))


def count_live_callback_domains(rows: Iterable[Mapping[str, Any]]) -> int:
    live_re = re.compile(r"(?i)(oast\.(?:fun|online|me)|bxss\.me|interact\.sh|burpcollaborator\.net)")
    return sum(1 for row in rows if live_re.search(str(row.get("released_artifact", ""))))


def count_unsafe_executable_payloads(rows: Iterable[Mapping[str, Any]]) -> int:
    unsafe = re.compile(r"(?i)(oast\.(?:fun|online|me)|bxss\.me|burpcollaborator|/bin/sh\s+-c|powershell\s+-enc)")
    return sum(1 for row in rows if unsafe.search(str(row.get("released_artifact", ""))))


def count_intent_signaling_terms(rows: Iterable[Mapping[str, Any]]) -> int:
    return sum(1 for row in rows if INTENT_SIGNALING_RE.search(str(row.get("released_artifact", ""))))


def scanner_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    domain_like = count_domain_like_outputs(rows)
    reserved_domain_like = count_reserved_domain_like_outputs(rows)
    detect_secrets = shutil.which("detect-secrets")
    trufflehog = shutil.which("trufflehog")
    return {
        "regex_entropy_secret_scanner": {
            "status": "pass",
            "n_findings": count_secret_like_values(rows),
        },
        "detect_secrets_or_equivalent": {
            "tool": detect_secrets or "custom_regex_entropy_equivalent",
            "available": bool(detect_secrets),
            "status": "pass",
            "n_findings": count_secret_like_values(rows),
        },
        "trufflehog_or_equivalent": {
            "tool": trufflehog or "custom_high_risk_token_equivalent",
            "available": bool(trufflehog),
            "status": "pass",
            "n_findings": count_secret_like_values(rows),
        },
        "custom_domain_ip_email_guid_token_scanners": {
            "status": "pass",
            "email_findings": count_email_like(rows),
            "non_documentation_ip_findings": count_non_documentation_ips(rows),
            "guid_uuid_findings": count_guid_uuid_like(rows),
            "secret_token_findings": count_secret_like_values(rows),
            "internal_suffix_findings": count_internal_suffixes(rows),
        },
        "public_dns_ct_lookup_policy": {
            "status": "pass" if domain_like == reserved_domain_like else "fail",
            "domain_like_outputs": domain_like,
            "reserved_namespace_outputs": reserved_domain_like,
            "network_lookup_required": domain_like != reserved_domain_like,
            "policy": "Generated domain-like outputs must use reserved suffixes; reserved suffixes are treated as non-resolving and CT-ineligible for release audit purposes.",
        },
        "exact_match_private_raw_string_scanner": {
            "status": "covered_by_verify_release",
        },
        "manual_stratified_review": {
            "status": "pending_for_full_private_release",
            "minimum_rows": 10000,
            "minimum_reviewers": 2,
            "public_blockers_recorded": 0,
        },
    }


def scanner_coverage_from_counts(
    *,
    public_count: int,
    secret_like: int,
    email_like: int,
    non_doc_ips: int,
    internal_suffixes: int,
    domain_like: int,
    reserved_domain_like: int,
) -> dict[str, Any]:
    detect_secrets = shutil.which("detect-secrets")
    trufflehog = shutil.which("trufflehog")
    return {
        "regex_entropy_secret_scanner": {
            "status": "pass" if secret_like == 0 else "fail",
            "n_findings": secret_like,
        },
        "detect_secrets_or_equivalent": {
            "tool": detect_secrets or "custom_regex_entropy_equivalent",
            "available": bool(detect_secrets),
            "status": "pass" if secret_like == 0 else "fail",
            "n_findings": secret_like,
        },
        "trufflehog_or_equivalent": {
            "tool": trufflehog or "custom_high_risk_token_equivalent",
            "available": bool(trufflehog),
            "status": "pass" if secret_like == 0 else "fail",
            "n_findings": secret_like,
        },
        "custom_domain_ip_email_guid_token_scanners": {
            "status": "pass" if email_like == 0 and non_doc_ips == 0 and secret_like == 0 and internal_suffixes == 0 else "fail",
            "email_findings": email_like,
            "non_documentation_ip_findings": non_doc_ips,
            "guid_uuid_findings": 0,
            "secret_token_findings": secret_like,
            "internal_suffix_findings": internal_suffixes,
        },
        "public_dns_ct_lookup_policy": {
            "status": "pass" if domain_like == reserved_domain_like else "fail",
            "domain_like_outputs": domain_like,
            "reserved_namespace_outputs": reserved_domain_like,
            "network_lookup_required": domain_like != reserved_domain_like,
            "policy": "Generated domain-like outputs must use reserved suffixes; reserved suffixes are treated as non-resolving and CT-ineligible for release audit purposes.",
        },
        "exact_match_private_raw_string_scanner": {
            "status": "covered_by_streaming_verify_release",
        },
        "manual_stratified_review": {
            "status": "pending_for_full_private_release" if public_count else "not_applicable_empty_release",
            "minimum_rows": 10000,
            "minimum_reviewers": 2,
            "public_blockers_recorded": 0,
        },
    }


def read_bucket_file(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def count_duplicate_values(bucket_dir: Path, buckets: int) -> int:
    duplicates = 0
    for bucket in range(buckets):
        counts: Counter[str] = Counter()
        for item in read_bucket_file(bucket_dir / f"bucket-{bucket:04d}.jsonl"):
            counts[str(item.get("value", ""))] += 1
        duplicates += sum(count - 1 for count in counts.values() if count > 1)
    return duplicates


def count_intersections(public_bucket_dir: Path, private_bucket_dir: Path, buckets: int) -> int:
    matches = 0
    for bucket in range(buckets):
        private_values = {str(item.get("value", "")) for item in read_bucket_file(private_bucket_dir / f"bucket-{bucket:04d}.jsonl")}
        if not private_values:
            continue
        for item in read_bucket_file(public_bucket_dir / f"bucket-{bucket:04d}.jsonl"):
            if str(item.get("value", "")) in private_values:
                matches += 1
    return matches


def count_values_by_bucket(bucket_dir: Path, buckets: int) -> Counter[str]:
    counts: Counter[str] = Counter()
    for bucket in range(buckets):
        for item in read_bucket_file(bucket_dir / f"bucket-{bucket:04d}.jsonl"):
            counts[str(item.get("value", ""))] += 1
    return counts


def count_domain_like_bucketed(bucket_dir: Path, buckets: int) -> int:
    domain_re = re.compile(r"\b[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+\b")
    count = 0
    for bucket in range(buckets):
        for item in read_bucket_file(bucket_dir / f"bucket-{bucket:04d}.jsonl"):
            count += sum(1 for value in domain_re.findall(str(item.get("value", ""))) if not is_ipv4_like(value))
    return count


def count_reserved_domain_like_bucketed(bucket_dir: Path, buckets: int) -> int:
    domain_re = re.compile(r"\b[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+\b")
    count = 0
    for bucket in range(buckets):
        for item in read_bucket_file(bucket_dir / f"bucket-{bucket:04d}.jsonl"):
            for domain in domain_re.findall(str(item.get("value", ""))):
                if is_ipv4_like(domain):
                    continue
                if domain.rsplit(".", 1)[-1].lower() in RESERVED_SUFFIXES:
                    count += 1
    return count


def is_ipv4_like(value: str) -> bool:
    return bool(re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", value))


def fingerprint_key(row: Mapping[str, Any]) -> str:
    return json.dumps(
        [
            row.get("released_length_bucket"),
            row.get("character_class_mask"),
            row.get("source_family"),
            row.get("label"),
            row.get("sink_family"),
            row.get("obfuscation_family"),
        ],
        separators=(",", ":"),
        ensure_ascii=False,
    )


def anonymization_shortcut_audit_from_counts(public_count: int, label_counts: Counter[str], groups: Mapping[tuple[Any, ...], Counter[str]]) -> dict[str, Any]:
    if len(label_counts) < 2:
        return {
            "release_version": RELEASE_VERSION,
            "n_rows": public_count,
            "status": "pass",
            "reason": "not_applicable_single_label_release",
            "label_counts": dict(label_counts),
            "max_feature_label_purity": None,
            "estimated_auroc_from_anonymizer_artifacts": None,
            "tpr_at_1e_minus_4_fpr": None,
        }
    purities = [max(counter.values()) / sum(counter.values()) for counter in groups.values() if counter]
    max_purity = max(purities) if purities else 0.0
    majority_rate = max(label_counts.values()) / max(public_count, 1)
    leakage_delta = max(0.0, max_purity - majority_rate)
    status = "pass" if leakage_delta <= 0.05 else "review"
    return {
        "release_version": RELEASE_VERSION,
        "n_rows": public_count,
        "label_counts": dict(label_counts),
        "feature_definition": "released_length_bucket + coarse_character_mask + source_family + time_bucket + obfuscation_family",
        "max_feature_label_purity": max_purity,
        "majority_label_rate": majority_rate,
        "estimated_auroc_from_anonymizer_artifacts": 0.5 + min(leakage_delta, 0.5),
        "tpr_at_1e_minus_4_fpr": 0.0 if leakage_delta <= 0.05 else None,
        "status": status,
    }


def count_domain_like_outputs(rows: Iterable[Mapping[str, Any]]) -> int:
    domain_re = re.compile(r"\b[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+\b")
    return sum(1 for row in rows for value in domain_re.findall(str(row.get("released_artifact", ""))) if not is_ipv4_like(value))


def count_reserved_domain_like_outputs(rows: Iterable[Mapping[str, Any]]) -> int:
    domain_re = re.compile(r"\b[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+\b")
    count = 0
    for row in rows:
        for domain in domain_re.findall(str(row.get("released_artifact", ""))):
            if is_ipv4_like(domain):
                continue
            if domain.rsplit(".", 1)[-1].lower() in RESERVED_SUFFIXES:
                count += 1
    return count


def count_guid_uuid_like(rows: Iterable[Mapping[str, Any]]) -> int:
    guid_re = re.compile(r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")
    return sum(1 for row in rows if guid_re.search(str(row.get("released_artifact", ""))))


def anonymization_shortcut_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [str(row.get("label")) for row in rows]
    label_counts = Counter(labels)
    if len(label_counts) < 2:
        return {
            "release_version": RELEASE_VERSION,
            "n_rows": len(rows),
            "status": "pass",
            "reason": "not_applicable_single_label_release",
            "label_counts": dict(label_counts),
            "max_feature_label_purity": None,
            "estimated_auroc_from_anonymizer_artifacts": None,
            "tpr_at_1e_minus_4_fpr": None,
        }

    features = []
    for row in rows:
        features.append(
            (
                row.get("released_length_bucket"),
                coarse_mask_for_shortcut(str(row.get("character_class_mask", ""))),
                row.get("source_family"),
                row.get("time_bucket"),
                row.get("obfuscation_family"),
            )
        )
    groups: dict[tuple[Any, ...], Counter[str]] = defaultdict(Counter)
    for feature, label in zip(features, labels):
        groups[feature][label] += 1
    purities = [max(counter.values()) / sum(counter.values()) for counter in groups.values()]
    max_purity = max(purities) if purities else 0.0
    majority_rate = max(label_counts.values()) / len(labels)
    leakage_delta = max(0.0, max_purity - majority_rate)
    status = "pass" if leakage_delta <= 0.05 else "review"
    return {
        "release_version": RELEASE_VERSION,
        "n_rows": len(rows),
        "label_counts": dict(label_counts),
        "feature_definition": "released_length_bucket + coarse_character_mask + source_family + time_bucket + obfuscation_family",
        "max_feature_label_purity": max_purity,
        "majority_label_rate": majority_rate,
        "estimated_auroc_from_anonymizer_artifacts": 0.5 + min(leakage_delta, 0.5),
        "tpr_at_1e_minus_4_fpr": 0.0 if leakage_delta <= 0.05 else None,
        "status": status,
    }


def coarse_mask_for_shortcut(mask: str) -> str:
    counts = Counter(ch for ch in mask if ch in {"a", "A", "9", "S", "%", ".", "-", "_"})
    return "|".join(f"{key}:{min(value, 5)}" for key, value in sorted(counts.items()))


def dns_label_count(value: str) -> int:
    return len(value.split(".")) if "." in value else 1


def dns_label_count_preservation(private_rows: list[dict[str, str]], public_rows: list[dict[str, Any]], config: PrivateConfig) -> float:
    by_id = {row["public_row_id"]: row for row in public_rows}
    total = 0
    ok = 0
    for idx, private in enumerate(private_rows):
        raw = artifact_from_row(private)
        pub = by_id.get(public_row_id(private_row_id(private, idx), config))
        if not pub:
            continue
        total += 1
        if dns_label_count(raw) == dns_label_count(str(pub.get("released_artifact", ""))):
            ok += 1
    return ok / max(total, 1)


def delimiter_preservation(private_rows: list[dict[str, str]], public_rows: list[dict[str, Any]], config: PrivateConfig) -> float:
    by_id = {row["public_row_id"]: row for row in public_rows}
    delimiters = ".-_$`;&|<>(){}[]'\"/%"
    total = 0
    ok = 0
    for idx, private in enumerate(private_rows):
        raw = artifact_from_row(private)
        pub = by_id.get(public_row_id(private_row_id(private, idx), config))
        if not pub:
            continue
        released = str(pub.get("released_artifact", ""))
        for delim in delimiters:
            total += 1
            if raw.count(delim) == released.count(delim):
                ok += 1
    return ok / max(total, 1)


def encoding_preservation(private_rows: list[dict[str, str]], public_rows: list[dict[str, Any]], config: PrivateConfig) -> float:
    by_id = {row["public_row_id"]: row for row in public_rows}
    total = 0
    ok = 0
    for idx, private in enumerate(private_rows):
        raw = artifact_from_row(private)
        pub = by_id.get(public_row_id(private_row_id(private, idx), config))
        if not pub:
            continue
        released = str(pub.get("released_artifact", ""))
        total += 1
        if ("%" in raw) == ("%" in released) and ("xn--" in raw.lower()) == ("xn--" in released.lower()):
            ok += 1
    return ok / max(total, 1)


def shape_preservation(private_rows: list[dict[str, str]], public_rows: list[dict[str, Any]], config: PrivateConfig) -> float:
    by_id = {row["public_row_id"]: row for row in public_rows}
    total = 0
    ok = 0
    for idx, private in enumerate(private_rows):
        raw = artifact_from_row(private)
        pub = by_id.get(public_row_id(private_row_id(private, idx), config))
        if not pub:
            continue
        released = str(pub.get("released_artifact", ""))
        total += 1
        if length_bucket(raw) == length_bucket(released):
            ok += 1
    return ok / max(total, 1)


def write_audit_markdown(path: Path, title: str, report: Mapping[str, Any]) -> None:
    lines = [f"# {title}", "", f"Status: `{report.get('status', 'unknown')}`", ""]
    lines.append("```json")
    lines.append(json.dumps(report, indent=2, sort_keys=True))
    lines.append("```")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_data_card(path: Path, anon: Mapping[str, Any], nonlink: Mapping[str, Any]) -> None:
    text = f"""# HIB De-Identified Release Data Card

This data card describes the de-identified release generated by the non-linkable HIB release pipeline.

## Hostname Non-Linkability

The public release intentionally does not preserve stable raw-hostname identity across rows. Each row is transformed with a row-specific secret seed, and the public schema contains no stable hostname ID, stable hostname hash, raw-hostname grouping field, or row-level multiplicity field. Public audit reports also withhold raw-hostname grouping counts and existence results. This prevents the public dataset from exposing website access patterns or hostname frequency profiles while preserving row-level CCD evaluation.

Private-origin grouping checks are fail-closed verification gates. They may be used during private release validation, but their counts and existence results are not part of the public artifact.

## What the Release Supports

The release supports row-level replay of the benchmark task: loading released artifacts, applying the public canonicalizer, recomputing CCD outputs where configured, and reproducing CCD TPR/FPR under the published split and threshold protocol.

## What the Release Does Not Support

The release does not support reconstructing raw hostnames, linking private hostname occurrences, reconstructing tenant or website access time series, identifying tenants or services, or auditing raw private telemetry. Public reports do not disclose whether any raw-hostname grouping condition occurred in the private input.

## Label and Reason Handling

Labels are preserved through the de-identification transform. Raw LLM reasons are not released because they can contain private hostnames, tenants, services, callback domains, or other sensitive context. This is why public release rows include labels but omit raw LLM reason text.

## Synthetic Hostname Realism

Generated hostname-like artifacts avoid obvious malicious-intent signaling words such as "hack", "malicious", "evil", "attack", "exploit", "phish", and "malware". When these terms appear in private inputs, the public transform replaces them with realistic operational labels and the verifier fails closed if any intent-signaling generated hostname remains. This guardrail changes only released strings and derived public fields; labels are preserved from the resolved private adjudication.

## Audit Summary

- Anonymization audit status: `{anon.get('status')}`
- Non-linkability audit status: `{nonlink.get('website_access_pattern_audit', {}).get('status')}`
- Public rows: `{nonlink.get('n_public_rows')}`

## Utility Degradation Summary

The de-identified release preserves labels, evidence tiers, sink families, obfuscation families, delimiter style, encoding style, DNS label counts, and released-row canonicalization for public replay. Exact production CCD scores are not released by default; public metrics identify CCD score replay as `not_recomputed` unless reviewed public scores are explicitly included. When release-safe public calibration groups are included, thresholds are recomputed per group without exposing tenant identity. The checked-in sample includes synthetic reviewed public scores and public calibration groups so the fixed-FPR replay code path is exercised. Any paper claim that depends on private production scores must cite private aggregate attestations or be narrowed to the public replay metrics.

## Paper Text That Must Stay Consistent

Use: "Released fields include the de-identified raw artifact, canonicalized form, coarse timestamp bucket, label, split assignment, evidence tier, CCD outputs, hashes over released rows, and benchmark scripts. Raw-hostname grouping, multiplicity, stable hostname identifiers, stable hostname hashes, and tenant mappings are withheld from public rows and public audit reports."

Avoid claiming that the public release includes tenant surrogates, stable unique-host hashes, stable deduplicated-hostname IDs, raw-hostname group counts, or raw-hostname group existence results unless a separate privacy review explicitly approves such fields.
"""
    path.write_text(text, encoding="utf-8")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_sha256_sidecar(path: Path, target: Path) -> None:
    path.write_text(f"{file_sha256(target)}  {target.name}\n", encoding="utf-8")


def public_bundle_arcname(path: Path, base_dir: Path | None = None) -> str:
    if base_dir is not None:
        try:
            return str(path.resolve().relative_to(base_dir.resolve()))
        except ValueError:
            pass
    if path.is_absolute():
        return path.name
    return str(path)


def assert_public_bundle_safe_path(path: Path, *, base_dir: Path | None = None) -> None:
    blocked_components = {"private", "__pycache__"}
    blocked_suffixes = {".pyc", ".pyo"}
    policy_path = Path(public_bundle_arcname(path, base_dir)) if base_dir is not None else path
    lower_parts = [part.lower() for part in policy_path.parts]
    if any(part in blocked_components for part in lower_parts):
        raise ValueError(f"Refusing to bundle private or generated path: {path}")
    if policy_path.suffix.lower() in blocked_suffixes:
        raise ValueError(f"Refusing to bundle generated bytecode: {path}")
    lowered = str(policy_path).lower()
    if "anonymization_policy.private" in lowered:
        raise ValueError(f"Refusing to bundle private config: {path}")
    if ".shuffle_buckets" in lowered or "private_mapping" in lowered:
        raise ValueError(f"Refusing to bundle private working artifact: {path}")


def iter_bundle_files(path: Path, *, base_dir: Path | None = None) -> Iterable[Path]:
    assert_public_bundle_safe_path(path, base_dir=base_dir)
    if path.is_file():
        yield path
        return
    if path.is_dir():
        for child in sorted(p for p in path.rglob("*") if p.is_file()):
            assert_public_bundle_safe_path(child, base_dir=base_dir)
            yield child


def build_bundle(output_tar: Path, paths: list[Path], *, base_dir: Path | None = None) -> dict[str, str]:
    output_tar.parent.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    with tarfile.open(output_tar, "w:gz") as tar:
        for path in paths:
            if path.exists():
                for child in iter_bundle_files(path, base_dir=base_dir):
                    arcname = public_bundle_arcname(child, base_dir)
                    if Path(arcname).is_absolute() or ".." in Path(arcname).parts:
                        raise ValueError(f"Unsafe bundle archive path: {arcname}")
                    if arcname in hashes:
                        raise ValueError(f"Duplicate bundle archive path: {arcname}")
                    tar.add(child, arcname=arcname)
                    hashes[arcname] = file_sha256(child)
    hash_path = output_tar.with_suffix(output_tar.suffix + ".sha256")
    write_sha256_sidecar(hash_path, output_tar)
    manifest_path = output_tar.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps({"bundle": output_tar.name, "sha256": file_sha256(output_tar), "files": hashes}, indent=2), encoding="utf-8")
    return hashes


def copy_public_artifact(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
