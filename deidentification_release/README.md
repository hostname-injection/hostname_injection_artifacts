# HIB De-Identification Release Pipeline

This folder contains the non-linkable HIB de-identification implementation. It
is intentionally isolated from the original benchmark files so the private
inputs and existing benchmark artifacts are not overwritten.

The public release uses row-specific secrets to transform hostname-like
artifacts. The public schema and public audit reports do not include tenant
surrogates, stable hostname hashes, raw-hostname grouping fields,
deduplication IDs, raw-hostname multiplicity, raw timestamps, or raw LLM
reasons.

## Build

```sh
python deidentification_release/scripts/anonymize_hib_release.py \
  --input-private data/private/hib_eval_snapshot.csv \
  --private-config deidentification_release/configs/anonymization_policy.private.yaml \
  --public-policy deidentification_release/configs/anonymization_policy.public.yaml \
  --output deidentification_release/data/release/hib_release.jsonl \
  --audit-dir deidentification_release/data/audits
```

`--input-private` may also point at a directory of CSV chunks. In directory
mode, the anonymizer streams all `*.csv` files in lexical order and writes the
public release through deterministic HMAC shuffle buckets so original chunk,
time, tenant, or hostname order is not preserved:

```sh
python deidentification_release/scripts/anonymize_hib_release.py \
  --input-private /path/to/HostnameCommandInjectionBenchmark/data/dns_hostnames/chunks \
  --private-config /path/to/HostnameCommandInjectionBenchmark/deidentification_release_full/private/anonymization_policy.private.yaml \
  --public-policy deidentification_release/configs/anonymization_policy.public.yaml \
  --output /path/to/HostnameCommandInjectionBenchmark/deidentification_release_full/data/release/hib_release.full.jsonl \
  --audit-dir /path/to/HostnameCommandInjectionBenchmark/deidentification_release_full/data/audits \
  --shuffle-buckets 1024
```

For full runs, provide private secrets through environment variables rather
than command-line arguments:

```sh
export HIB_DEID_ROW_ID_SECRET=...
export HIB_DEID_ARTIFACT_SECRET=...
export HIB_DEID_SHUFFLE_SECRET=...
```

## Verify

```sh
python deidentification_release/scripts/verify_anonymization.py \
  --private-input data/private/hib_eval_snapshot.csv \
  --public-release deidentification_release/data/release/hib_release.jsonl \
  --private-config deidentification_release/configs/anonymization_policy.private.yaml \
  --policy deidentification_release/configs/anonymization_policy.public.yaml \
  --audit-dir deidentification_release/data/audits
```

Directory inputs use streaming verification and on-disk buckets:

```sh
python deidentification_release/scripts/verify_anonymization.py \
  --private-input /path/to/HostnameCommandInjectionBenchmark/data/dns_hostnames/chunks \
  --public-release /path/to/HostnameCommandInjectionBenchmark/deidentification_release_full/data/release/hib_release.full.jsonl \
  --private-config /path/to/HostnameCommandInjectionBenchmark/deidentification_release_full/private/anonymization_policy.private.yaml \
  --policy deidentification_release/configs/anonymization_policy.public.yaml \
  --audit-dir /path/to/HostnameCommandInjectionBenchmark/deidentification_release_full/data/audits \
  --streaming-buckets 1024
```

## Repair Existing Public Release Fields

When public policy changes only release-safe derived fields, stream-repair the
existing public JSONL into a new folder rather than overwriting the prior
artifact:

```sh
python deidentification_release/scripts/repair_public_release_fields.py \
  --input /path/to/HostnameCommandInjectionBenchmark/deidentification_release_full_v2/data/release/hib_release.full.v2.jsonl \
  --output /path/to/HostnameCommandInjectionBenchmark/deidentification_release_full_v3/data/release/hib_release.full.v3.jsonl
```

The repair recomputes the released canonical artifact, time bucket, sink/evidence
coarsening, length bucket, character-class mask, row integrity hash, schema, and
checksum sidecar. It must not change `public_row_id`, `label`, `split`,
`source_family`, `detector_outputs`, or the private-source row count. After any
repair, copy the private config into the new private folder and run the full
verifier with the current code:

```sh
mkdir -p /path/to/HostnameCommandInjectionBenchmark/deidentification_release_full_v3/private
cp /path/to/HostnameCommandInjectionBenchmark/deidentification_release_full_v2/private/anonymization_policy.private.yaml \
  /path/to/HostnameCommandInjectionBenchmark/deidentification_release_full_v3/private/anonymization_policy.private.yaml

python deidentification_release/scripts/verify_anonymization.py \
  --private-input /path/to/HostnameCommandInjectionBenchmark/data/dns_hostnames/chunks \
  --public-release /path/to/HostnameCommandInjectionBenchmark/deidentification_release_full_v3/data/release/hib_release.full.v3.jsonl \
  --private-config /path/to/HostnameCommandInjectionBenchmark/deidentification_release_full_v3/private/anonymization_policy.private.yaml \
  --policy deidentification_release/configs/anonymization_policy.public.yaml \
  --audit-dir /path/to/HostnameCommandInjectionBenchmark/deidentification_release_full_v3/data/audits \
  --streaming-buckets 1024
```

## Bundle

```sh
python deidentification_release/scripts/build_release_bundle.py \
  --output deidentification_release/data/release/hib_release_public_bundle.tar.gz \
  --base-dir . \
  deidentification_release/data/release/hib_release.jsonl \
  deidentification_release/data/release/hib_release.schema.json \
  deidentification_release/data/release/hib_release.jsonl.sha256 \
  deidentification_release/data/audits/anonymization_audit_report.json \
  deidentification_release/data/audits/anonymization_audit_report.md \
  deidentification_release/data/audits/nonlinkability_audit_report.json \
  deidentification_release/data/audits/nonlinkability_audit_report.md \
  deidentification_release/data/audits/anonymization_shortcut_audit_report.json \
  deidentification_release/data/audits/anonymization_shortcut_audit_report.md \
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
  deidentification_release/scripts/validate_release_gate.py \
  deidentification_release/README.md
```

Private configs, secrets, raw inputs, raw mappings, salts, and stable
hostname/tenant groupings must not be included in the public bundle.

Release and bundle checksum sidecars are written in `sha256sum -c` compatible
format:

```sh
(cd deidentification_release/data/release && sha256sum -c hib_release.jsonl.sha256)
(cd deidentification_release/data/release && sha256sum -c hib_release_public_bundle.tar.gz.sha256)
```

The bundle manifest and archive contents can be checked with:

```sh
python deidentification_release/scripts/validate_public_bundle.py \
  --bundle deidentification_release/data/release/hib_release_public_bundle.tar.gz
```

After full verification, run the final gate validator:

```sh
python deidentification_release/scripts/validate_release_gate.py \
  --public-release deidentification_release/data/release/hib_release.jsonl \
  --audit-dir deidentification_release/data/audits \
  --bundle deidentification_release/data/release/hib_release_public_bundle.tar.gz \
  --count-rows
```

## Replay Metrics

Recompute public row-count, calibration, fixed-FPR, unresolved-row, and detector
overlap metrics from the released JSONL:

```sh
python deidentification_release/scripts/recompute_metrics.py \
  --public-release deidentification_release/data/release/hib_release.jsonl \
  --alpha 1e-4 \
  --out deidentification_release/data/audits/recomputed_public_metrics.json
```

If public CCD scores are present in `detector_outputs`, the script recomputes
the split-conformal threshold from benign calibration rows. If scores are not
present, it records that fixed-FPR score replay is not available while still
recomputing row counts, released flag metrics, unresolved-row accounting, and
detector overlap.

The checked-in sample bundle includes reviewed synthetic public CCD scores so
this path reports `fixed_fpr_replay.status = "available"`. If
`public_calibration_group` is present in `detector_outputs`, thresholds are
recomputed per release-safe public group rather than from one global pool. Full
production score or calibration-group release remains opt-in and must be covered
by the release review.

## Private Original Attack-Vector Export

For private analysis only, the original malicious attack vectors can be
exported without modifying the source chunks:

```sh
python deidentification_release/scripts/export_original_attack_vectors.py \
  --root /path/to/HostnameCommandInjectionBenchmark \
  --output /path/to/HostnameCommandInjectionBenchmark/deidentification_release_full/private/original_attack_vectors_deduplicated.csv \
  --summary /path/to/HostnameCommandInjectionBenchmark/deidentification_release_full/private/original_attack_vectors_deduplicated.summary.json
```

This export contains raw original attack-vector text and raw model reasons. It
can reveal private input grouping properties, so it is private-only and must not
be included in the public de-identified bundle or public audit reports.
