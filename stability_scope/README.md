# Stability, Drift, And Scope Aggregate Accounting

This directory contains release-safe aggregate values for the paper's
finite-edit decision-stability, family-holdout/composition-depth, drift-refresh,
and public-real scope checks.

Run:

```bash
python scripts/recompute_stability_scope_metrics.py
```

The script validates the Figure 7 certificate-coverage claim, the Section 8.2
holdout/depth ranges, drift-refresh invariants, independent replay tolerance,
and public-real recall ordering. It also records the narrow certificate scope:
the deployed normalizer, cone sketch, score path, threshold, and edit-manifest
version. These aggregates do not disclose raw drift rows, tenant identities, or
per-row edit-ball stress cases.

End-to-end reproduction from rows requires the full HIB-Real release and staged
stress outputs described in `ARTIFACT_EVALUATION.md`.
