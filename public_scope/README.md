# Public-Scope Taxonomy Accounting

The paper uses public evidence outside HIB to probe taxonomy scope. This is not
HIB replay accuracy and is not counted as HIB training or production positives.

This directory contains release-safe aggregate accounting for the public-scope
claim:

- 118 of 127 post-freeze public reports map to target hostname-injection
  categories.
- The nine exclusions are rebinding, certificate-validation, or QUIC cases,
  which require complementary controls.
- Public anchors such as Cockpit remote-login RCE, enterprise git push RCE,
  Koa `ctx.hostname`, and legacy host/diagnostic reports are taxonomy anchors
  only.

Recompute the aggregate accounting with:

```bash
python scripts/recompute_public_scope_metrics.py
```

When a public-report label export is staged, provide JSONL rows with:

- `status`: `target`/`mapped`/`in_scope`/`covered`, or
  `excluded`/`out_of_scope`/`boundary`.
- `target_category`: required for mapped rows.
- `exclusion_reason`: required for excluded rows; must be `rebinding`,
  `certificate-validation`, or `quic`.

Example:

```bash
python scripts/recompute_public_scope_metrics.py \
  --reports /path/to/public_report_labels.jsonl \
  --counts public_scope/paper_public_scope_counts.json \
  --expect-counts
```
