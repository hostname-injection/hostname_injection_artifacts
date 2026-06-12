# CCD Smoke Inputs

These small files exercise the paper pipeline without using private HIB rows.
They are not a substitute for the released HIB replay; they are intentionally
tiny inputs for installation checks, demos, and artifact kick-the-tires review.

Run the end-to-end smoke path from the repository root:

```bash
python scripts/run_artifact_smoke.py --skip-tests
```

The smoke command trains a temporary CCD prior bundle from `benign.txt` and
`malicious.csv`, calibrates on `benign_calibration.txt`, scores `queries.txt`,
and validates the checked-in de-identified HIB sample bundle.
