# CCD Smoke Inputs

These small files exercise the paper pipeline without using private HIB rows.
They are not a substitute for the released HIB replay; they are intentionally
tiny inputs for installation checks, demos, and reviewer smoke tests.

Run the end-to-end smoke path from the repository root:

```bash
python scripts/run_artifact_smoke.py --skip-tests
```

The smoke command trains a temporary CAHO checkpoint, trains CCD priors from
that checkpoint, calibrates on `benign_calibration.txt`, refreshes `P_B`, scores
and certifies `queries.txt`, and validates the checked-in de-identified HIB
sample bundle.
