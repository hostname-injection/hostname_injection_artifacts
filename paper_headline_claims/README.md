# Paper Headline Claim Audit

This directory contains a release-safe audit for the headline numeric claims in
`IEEE_S_P_Hostnames.pdf`: replay scale, CCD recall at the fixed false-positive
budget, latency, live-overlap added value, decision-stability coverage, public
scope, source-reachability, and hostile-mimicry ranges.

Run:

```bash
python scripts/recompute_paper_headline_claims.py
```

The checker recomputes these anchors from the aggregate accounting files already
used by the table-specific scripts. It does not replace the full HIB-Real replay
or raw production traces; it verifies that the release-safe aggregate artifacts
stay aligned with the paper's most visible claims and that external boundaries
remain explicit.
