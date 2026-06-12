# Non-Linkability Audit Report

Status: `pass`

```json
{
  "access_pattern_checks": {
    "n_public_tenant_time_series_fields": 0,
    "n_sparse_public_combinations": 0,
    "row_order_reveals_private_tenant": false,
    "row_order_reveals_private_time": false,
    "status": "pass",
    "time_source_label_tier_sink_min_k": 50
  },
  "n_public_rows": 150,
  "private_origin_linkage_checks": {
    "raw_hostname_group_counts_released": false,
    "raw_hostname_group_existence_released": false,
    "raw_hostname_multiplicity_released": false,
    "stable_hostname_identifier_fields_released": false,
    "status": "pass"
  },
  "public_uniqueness_checks": {
    "n_duplicate_released_artifact_values": 0,
    "n_duplicate_released_canonical_values": 0,
    "n_forbidden_stable_hostname_hashes": 0,
    "n_forbidden_stable_hostname_ids": 0,
    "status": "pass"
  },
  "release_version": "hib-v1.0",
  "status": "pass",
  "structural_fingerprint_checks": {
    "fingerprint_definition": "length_bucket + character_class_mask + source_family + label + sink_family + obfuscation_family",
    "n_global_fingerprints_below_k": 0,
    "private_raw_hostname_group_results_released": false,
    "status": "pass"
  },
  "website_access_pattern_audit": {
    "raw_hostname_group_counts_released": false,
    "raw_hostname_group_existence_released": false,
    "raw_hostname_group_sizes_released": false,
    "stable_hostname_identifier_fields_released": false,
    "status": "pass"
  }
}
```
