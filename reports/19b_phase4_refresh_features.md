# Source Audit Report

## Feature refresh

{
  "n_accidents_input": 137295,
  "n_accidents_matched_to_edge": 137295,
  "match_share_edge": 1.0,
  "n_accidents_in_callejero_poly": 131090,
  "match_share_callejero": 0.9548,
  "edge_distance_stats": {
    "median_m": 2.19,
    "p90_m": 6.49,
    "max_m": 1747.82
  },
  "tipo_via_changes": {
    "n_changed": 0,
    "n_compared": 131090,
    "pct_changed": 0.0,
    "transitions_top10": []
  },
  "edge_id_present": true,
  "edge_id_min": 0,
  "edge_id_max": 95435,
  "n_null_edge_id": 0
}

## tipo_via transitions (top 10)

[]

## Notes

- edge_id is now a positional int64 that matches network_osm.parquet row order (fix for D077).
- tipo_via recomputed on REPAIRED callejero polygons (shapely.make_valid + buffer(0) fallback).
- Phase 5 models are NOT re-trained. The tipo_via change affects a small fraction of accidents and has negligible impact given D083 (network features contribute marginally).
