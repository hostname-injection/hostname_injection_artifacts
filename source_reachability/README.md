# Source-Reachability Scope Check

The paper reports a static-analysis scope check over 50 source-available
repositories that process host or telemetry fields. The comparison is separate
from HIB replay accuracy: CodeQL 2.17.0 and Semgrep 1.60.0 were run on a
public source corpus to measure direct source-to-sink coverage and missed
delayed log/warehouse paths.

The release-safe accounting file is
`paper_source_reachability_counts.json`. It records the aggregate paper counts:

- CodeQL: 10 true positives, 2 false positives, and 38 missed delayed paths.
- Semgrep: 6 true positives, 2 false positives, and 42 missed delayed paths.

Recompute the aggregate precision/recall accounting with:

```bash
python scripts/recompute_source_reachability_metrics.py
```

When a full public-corpus run is staged, export labeled findings as JSONL with
at least `tool` and `verdict` fields and compare it against an expected-count
file:

```bash
python scripts/recompute_source_reachability_metrics.py \
  --labels /path/to/labeled_findings.jsonl \
  --counts source_reachability/paper_source_reachability_counts.json \
  --expect-counts
```

Accepted verdict values are `true_positive`, `false_positive`, and
`missed_delayed_path` plus common aliases such as `tp`, `fp`, and `fn`.

The corpus repository list, raw SARIF/JSON tool traces, and any private review
notes are not bundled in this source tree. They are publication-stage artifacts
to attach with the final public corpus release if reviewers need to rerun the
CodeQL/Semgrep jobs end to end.
