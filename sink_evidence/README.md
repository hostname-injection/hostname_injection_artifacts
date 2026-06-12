# Table 8 Sink-Evidence Accounting

This directory contains release-safe aggregate accounting for the paper's
Table 8 case-study replay. The table is an evidence trace for audited
metadata-to-code paths, not a production-compromise claim.

Run:

```bash
python scripts/recompute_sink_evidence_metrics.py
```

The checker validates that the three published cases record parser boundary,
persistence, consumer, controlled effect, and detector boundary fields. It also
checks that every case is marked as controlled matching-sink replay, CCD fires
before downstream consumption, marker-only strings are not treated as positives
without downstream support, and side effects are blocked or restricted to
researcher-controlled endpoints.

The checked-in JSON intentionally excludes raw retained host-like values, exact
callback domains, tenant/owner identities, private sink implementation details,
and production side-effect traces. If evaluators need row-level case-chain
reproduction, stage a masked export separately and keep those fields outside
the source artifact.
