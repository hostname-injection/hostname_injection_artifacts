# HIB Dataset-Profile Aggregate Accounting

This directory contains release-safe aggregate counts for the HIB production
replay profile reported in the paper. The counts cover Table 3, Table 4,
Appendix A Table 9, and Appendix B dataset profile claims.

Run:

```bash
python scripts/recompute_hib_profile_metrics.py
```

The script validates that source rows, resolved/unresolved rows, label counts,
verified-positive family counts, and coarse obfuscation counts are internally
consistent. It also derives source percentages, replay denominators, positive
prevalence, unresolved rate, repair rate, and the largest attack-family share.

This is aggregate accounting only. End-to-end reproduction from rows requires
the full HIB-Real de-identified release bundle described in
`ARTIFACT_EVALUATION.md`.
