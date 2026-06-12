# Artifact Data Provenance and Ethics Notes

This file summarizes data provenance and ethics/privacy handling for artifact
evaluation metadata. The paper contains the full study context; this artifact
keeps private operational details outside the public release boundary.

## Public Sample Bundle

The checked-in HIB public sample bundle under
`deidentification_release/data/release/` is a small de-identified fixture. It was
generated from a temporary synthetic private CSV containing 60 benign DNS rows.
The temporary private CSV and private HMAC secrets were deleted after the
release, verification, metric recomputation, and bundle build steps.

The sample bundle exists to exercise:

- public schema validation;
- checksum and bundle manifests;
- de-identification release gates;
- non-linkability and public/private boundary checks;
- public replay metric recomputation;
- extracted-bundle validation.

It is not the full paper-scale HIB-Real replay.

## Full HIB-Real Release

The paper describes a 200,339,886-row, 835-tenant HIB-Real replay. The full
artifact release must use the same public schema, validation gates, and metric
recomputation scripts, but the full JSONL bundle is external to this source
repository because of its size and privacy review requirements.

Public rows intentionally exclude tenant identities, raw operational logs,
reversible mappings, raw callback domains, private sink details, production
control internals, exact private strings, and raw-hostname recurrence facts.
Private-origin linkage checks are used only as fail-closed release gates.

## Ethics and Privacy Boundary

The artifact is designed for offline evaluation. It does not require contacting
production services, querying live domains, running payloads, or using API keys.

Public de-identification follows these principles:

- release only fields needed for row-level detector evaluation;
- preserve labels, split assignment, detector outputs where public, and
  benchmark-relevant string structure;
- withhold private raw strings, tenants, stable hostname identifiers, stable
  tenant identifiers, row-level multiplicity fields, raw LLM reasons, and
  private mappings;
- publish audits that show gates passed without disclosing private grouping
  counts or existence results.

Benchmark strings can contain shell, SQL, template, URL lookup, or callback
syntax. Evaluators should treat every string as untrusted input and only process
it through the offline scripts included in this artifact.
