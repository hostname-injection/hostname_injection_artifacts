# Anonymization Shortcut Audit Report

Status: `pass`

```json
{
  "estimated_auroc_from_anonymizer_artifacts": 0.5,
  "feature_definition": "released_length_bucket + coarse_character_mask + source_family + time_bucket + obfuscation_family",
  "label_counts": {
    "resolved_benign": 100,
    "verified_executable_semantics": 50
  },
  "majority_label_baseline": 0.6666666666666666,
  "max_feature_label_purity": 0.6666666666666666,
  "n_rows": 150,
  "release_version": "hib-v1.0",
  "status": "pass",
  "tpr_at_1e_minus_4_fpr": 0.0
}
```
