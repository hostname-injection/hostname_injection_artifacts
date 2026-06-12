# Live-Overlap Aggregate Accounting

The paper's live/shadow comparison is released as aggregate accounting and
masked summaries, not raw live streams. This directory contains release-safe
counts for the Table 7 overlap between CCD and maintained Regex/WAF controls.

Recompute the aggregate metrics with:

```bash
python scripts/recompute_live_overlap_metrics.py
```

The report derives:

- CCD-only verified live positives: `850`.
- CCD-bearing lower-bound reviewed-item PPV: `3150/3355`.
- Regex/WAF-bearing lower-bound reviewed-item PPV: `2300/2700`.
- Verified-benign and nonverified review load per day over the 92-day window.
- The baseline-only accounting: 200 all-alert items, 90 uncertain, 110 verified
  benign, and 0 verified positives.

When masked live labels are staged, provide JSONL rows with:

- `ccd_flag`: boolean.
- `regex_waf_flag`: boolean.
- `label`: `verified_positive`, `uncertain`, or `verified_benign` plus common
  aliases.

Example:

```bash
python scripts/recompute_live_overlap_metrics.py \
  --labels /path/to/masked_live_labels.jsonl \
  --counts live_overlap/paper_live_overlap_counts.json \
  --expect-counts
```
