# Paper Claim Coverage

This directory contains a release-safe coverage matrix for named claims in
`IEEE_S_P_Hostnames.pdf`: contributions, Eq. 1, Proposition 5.1, Lemma C.1,
Figures 1-7, Tables 1-12, and Appendices A-F.

Run:

```bash
python scripts/recompute_paper_claim_coverage.py
```

The checker verifies that each paper item maps to existing artifact paths,
declares a verification command, aligns with a manifest claim, and marks any
row-level or production-private dependency as an external completion item. It
does not claim that release-safe aggregates replace the full HIB-Real replay;
it makes those boundaries explicit and testable.
