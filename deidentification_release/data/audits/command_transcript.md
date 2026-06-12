# Reproduction Command Transcript

All implementation work for this de-identification release is isolated under
`deidentification_release/`. The original benchmark files are not overwritten by
these commands.

## Static Checks

```sh
python -m py_compile deidentification_release/scripts/*.py
```

Result: passed.

The synthetic de-identification tests exercise row-specific transformation,
streaming chunk-directory anonymization, streaming verification, fail-closed
label tamper checks, public bundle validation, final release-gate validation,
sha256sum-compatible sidecar checks, repair-path checks, and the
realistic-hostname/no intent-signaling guardrail.

## Public Sample Release Generation

The checked-in public sample artifacts were regenerated from a temporary
synthetic private CSV containing 150 DNS rows: 100 resolved-benign rows and 50
verified executable-semantics rows. The temporary private CSV included reviewed
synthetic public CCD scores and detector flags so the checked-in sample
exercises split-conformal fixed-FPR replay. The temporary private CSV was
deleted after generation. No private input, mapping table, salt, HMAC key,
raw-hostname grouping result, raw-hostname multiplicity result, or raw LLM
reason is included in the public bundle.

```sh
python deidentification_release/scripts/anonymize_hib_release.py \
  --input-private "$TEMP_SYNTHETIC_PRIVATE_CSV" \
  --output deidentification_release/data/release/hib_release.jsonl \
  --audit-dir deidentification_release/data/audits \
  --row-id-secret "$PRIVATE_ROW_ID_SECRET" \
  --artifact-secret "$PRIVATE_ARTIFACT_SECRET" \
  --shuffle-secret "$PRIVATE_SHUFFLE_SECRET"

python deidentification_release/scripts/verify_anonymization.py \
  --private-input "$TEMP_SYNTHETIC_PRIVATE_CSV" \
  --public-release deidentification_release/data/release/hib_release.jsonl \
  --audit-dir deidentification_release/data/audits \
  --row-id-secret "$PRIVATE_ROW_ID_SECRET" \
  --artifact-secret "$PRIVATE_ARTIFACT_SECRET" \
  --shuffle-secret "$PRIVATE_SHUFFLE_SECRET" \
  --min-k 50

python deidentification_release/scripts/recompute_metrics.py \
  --public-release deidentification_release/data/release/hib_release.jsonl \
  --alpha 1e-4 \
  --out deidentification_release/data/audits/recomputed_public_metrics.json
```

Verification result:

- top-level verification status: `pass`
- anonymization audit status: `pass`
- non-linkability audit status: `pass`
- anonymization shortcut audit status: `pass`
- public rows in synthetic measured release: `150`
- scored benign calibration rows: `9`
- public calibration groups: `2`
- fixed-FPR replay status: `available`
- fixed-FPR sample thresholds: `public_group_a=0.5`, `public_group_b=0.58`
- sample metric positives / negatives: `45` / `91`
- sample fixed-FPR TP / FP / TN / FN: `41` / `6` / `85` / `4`
- raw-hostname grouping counts released: `false`
- raw-hostname grouping existence released: `false`

## Public Bundle

```sh
python deidentification_release/scripts/build_release_bundle.py \
  --output deidentification_release/data/release/hib_release_public_bundle.tar.gz \
  --base-dir . \
  deidentification_release/README.md \
  deidentification_release/data/release/hib_release.jsonl \
  deidentification_release/data/release/hib_release.schema.json \
  deidentification_release/data/release/hib_release.jsonl.sha256 \
  deidentification_release/data/audits/anonymization_audit_report.json \
  deidentification_release/data/audits/anonymization_audit_report.md \
  deidentification_release/data/audits/anonymization_shortcut_audit_report.json \
  deidentification_release/data/audits/anonymization_shortcut_audit_report.md \
  deidentification_release/data/audits/nonlinkability_audit_report.json \
  deidentification_release/data/audits/nonlinkability_audit_report.md \
  deidentification_release/data/audits/recomputed_public_metrics.json \
  deidentification_release/data/audits/release_manifest.json \
  deidentification_release/data/audits/stage_manifests \
  deidentification_release/data/audits/release_data_card.md \
  deidentification_release/data/audits/command_transcript.md \
  deidentification_release/configs/anonymization_policy.public.yaml \
  deidentification_release/scripts/anonymize_hib_release.py \
  deidentification_release/scripts/build_release_bundle.py \
  deidentification_release/scripts/hib_deid.py \
  deidentification_release/scripts/repair_public_release_fields.py \
  deidentification_release/scripts/verify_anonymization.py \
  deidentification_release/scripts/recompute_metrics.py \
  deidentification_release/scripts/validate_public_bundle.py \
  deidentification_release/scripts/validate_release_gate.py
```

Bundle SHA-256 is written in `sha256sum -c` compatible format to
`deidentification_release/data/release/hib_release_public_bundle.tar.gz.sha256`.

Public bundle and final release gate validation:

```sh
python deidentification_release/scripts/validate_public_bundle.py \
  --bundle deidentification_release/data/release/hib_release_public_bundle.tar.gz

python deidentification_release/scripts/validate_release_gate.py \
  --public-release deidentification_release/data/release/hib_release.jsonl \
  --audit-dir deidentification_release/data/audits \
  --bundle deidentification_release/data/release/hib_release_public_bundle.tar.gz \
  --count-rows
```

Result:

- public bundle validation status: `pass`
- final release gate status on synthetic sample: `pass`
- sample release rows checked by final gate: `150`

## Publication Consistency Note

Use the term "de-identified release" in public documentation. Do not claim that
the public release includes tenant surrogates, stable unique-host hashes, stable
deduplicated-hostname IDs, raw-hostname group counts, raw-hostname group
existence results, or raw-hostname multiplicity results. Raw LLM reasons are
omitted from public rows; labels are preserved exactly through the resolved-label
mapping.

## Full Release Handling

Full release builds use the same public reporting contract. The verifier may use
private raw-hostname grouping checks internally as fail-closed gates, but public
rows, public audit reports, bundle manifests, and data cards must not reveal
whether any such grouping condition occurred in the private input.

Private exports that contain raw original attack-vector text, raw model reasons,
or any grouping property remain private-only and must not be bundled into the
public de-identified release.
