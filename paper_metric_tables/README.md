# Paper Metric-Table Accounting

This directory contains release-safe aggregate accounting for paper evaluation
tables that depend on private replay rows or specialized stress-test harnesses.
It is meant to make the paper's numeric claims auditable without bundling raw
tenant telemetry, private model-serving traces, or hostile-mimicry row logs.

Recompute the aggregate summaries with:

```bash
python scripts/recompute_paper_metric_tables.py
```

The script validates and derives summary checks for:

- Table 5: fixed-FPR baseline audit at the production alert budget.
- Table 6: mechanism ablation recall-point drops.
- Table 10: LLM baseline checkpoint identifiers used in the matched replay.
- Table 12: hostile-mimicry recall.
- Generator/source comparison and Appendix F synthetic-real validity summary.

End-to-end reproduction of the underlying rows requires the full HIB-Real
release, model/config identifiers, and stress-test outputs described in
`ARTIFACT_EVALUATION.md`.
