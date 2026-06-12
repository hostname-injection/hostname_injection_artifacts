# Evaluation Accounting Boundary

This directory contains release-safe accounting for the paper's Table 2
evaluation units and Appendix E/Table 11 reproducibility boundary.

Run:

```bash
python scripts/recompute_evaluation_accounting.py
```

The checker validates the published denominator vocabulary: replay entries,
verified positives, live comparison items, composite alerts,
tenant-visible alerts, and unresolved marker-like strings. It also validates
that Table 11 separates what is released or recomputable from what is withheld
for privacy and operational safety.

By default the checker reads `ARTIFACT_MANIFEST.json` and confirms that each
Table 11 external boundary has a matching external completion item. This keeps
the source artifact honest: local aggregate checks are runnable now, while the
full HIB-Real replay, masked row-level live labels, masked sink-evidence
exports, and full stress outputs remain explicit publication dependencies.
