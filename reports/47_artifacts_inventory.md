# Phase 11 · Iter 3 — Artifacts inventory

## Resumen

- **Scripts**: 61
- **Parquets** (top-level en data/processed/): 30
- **Tablas JSON** (reports/tables/): 88
- **Figuras PNG** (reports/figures/final/): 25

## Scripts por archivar

| script | razón |
| --- | --- |
| `scripts\41b_poisson_diagnostics.py` | Bug de fittedvalues (devuelve X·β, no exp(X·β)) |
| `scripts\41c_phase10_refit.py` | Superseded by 41d (residuales corregidos) |
| `scripts\44_phase10_diagnostics_v2.py` | Bootstrap con solo 23 hexes (diseño roto) |
| `scripts\44b_phase10_bootstrap_clean.py` | Baseline-shift no corregido; conclusión es 44c |
| `scripts\44c_bootstrap_centered.py` | Diagnóstico final: mapa FE inestable, descartado |

## Parquets huérfanos

| path | razón |
| --- | --- |
| `reports\tables\01_schema_samples.json` | no producer script |
| `reports\tables\01_url_availability.json` | no producer script |
| `reports\tables\17_callejero_load_error.json` | no producer script |
| `reports\tables\17_callejero_meta.json` | no producer script |
| `reports\tables\17_imd_fixed.json` | no producer script |
| `reports\tables\28_opcion_c_bart.json` | no producer script |

## Ficheros a revisar

_Ninguno_

## Scripts / artefactos a mantener

- Scripts: 56
- Parquets: 30
- Tablas JSON: 82
- Figuras PNG: 25

## Parquets top-level (resumen)

| path | n_rows | n_cols | size_kb | producer |
| --- | ---: | ---: | ---: | --- |
| `accidents_clean.parquet` | 137328 | 31 | 4821.1 | `06_phase2_build_clean.py` |
| `accidents_with_edges.parquet` | 137295 | 8 | 4496.0 | `36_phase7_compare_b.py` |
| `accidents_with_network_features.parquet` | 137295 | 21 | 5337.5 | `19b_phase4_refresh_features.py` |
| `accidents_with_segment.parquet` | 137295 | 11 | 5676.3 | `15_phase4_verify_and_join.py` |
| `baseline_test_predictions.parquet` | 42386 | 10 | 808.4 | `09_phase3_baselines.py` |
| `exposure_hex250m_monthly.parquet` | 74717 | 12 | 2736.1 | `40d_fix_units.py` |
| `hdbscan_clusters_all.parquet` | 74250 | 9 | 1076.3 | `31_phase6_hdbscan.py` |
| `hdbscan_clusters_injured.parquet` | 35107 | 9 | 565.9 | `31_phase6_hdbscan.py` |
| `hdbscan_persistence_all.parquet` | 137295 | 8 | 2393.6 | `32_phase6_persistence.py` |
| `hdbscan_persistence_injured.parquet` | 62460 | 8 | 1174.7 | `32_phase6_persistence.py` |
| `hex_risk_bootstrap.parquet` | 23 | 8 | 6.4 | `44_phase10_diagnostics_v2.py` |
| `hex_risk_bootstrap_v2.parquet` | 640 | 7 | 37.2 | `44b_phase10_bootstrap_clean.py` |
| `hex_risk_fe.parquet` | 929 | 5 | 34.0 | `42_phase10_fixed_effects.py` |
| `hex_risk_fe_flags.parquet` | 929 | 6 | 34.7 | `43_phase10_fe_diagnostics.py` |
| `imd_points.parquet` | 1961 | 43 | 340.3 | `16_phase4_fix_and_diagnose.py` |
| `moran_barrios.parquet` | 131 | 10 | 227.5 | `37_phase8_moran.py` |
| `moran_callejero.parquet` | 9285 | 10 | 17834.9 | `37_phase8_moran.py` |
| `moran_distritos.parquet` | 21 | 10 | 144.5 | `37_phase8_moran.py` |
| `moran_hex250m.parquet` | 3962 | 10 | 206.6 | `37_phase8_moran.py` |
| `network_callejero.parquet` | 9285 | 11 | 39630.9 | `17c_phase4_fix_callejero.py` |
| `network_kde_b.parquet` | 95437 | 37 | 27470.8 | `36_phase7_compare_b.py` |
| `network_kde_c.parquet` | 95437 | 37 | 27440.8 | `34_phase7_kde.py` |
| `network_kde_final.parquet` | 95437 | 7 | 5132.8 | `36b_kde_compare.py` |
| `network_osm.parquet` | 95437 | 12 | 13677.6 | `18b_phase4_fix_osm.py` |
| `network_segments.parquet` | 3928 | 9 | 556.3 | `14_phase4_extract_network.py` |
| `panel_hex_month_n_injured.parquet` | 74625 | 21 | 4922.4 | `41_phase10_poisson_baseline.py` |
| `persons_clean.parquet` | 322317 | 34 | 7855.3 | `06_phase2_build_clean.py` |
| `segments_with_imd.parquet` | 3928 | 16 | 629.6 | `16_phase4_fix_and_diagnose.py` |
| `splits.parquet` | 137328 | 5 | 864.0 | `08_phase3_create_splits.py` |
| `variance_by_year.parquet` | 7 | 9 | 6.1 | `44_phase10_diagnostics_v2.py` |

## Próximos pasos (script 48 — refactor)

1. **Mover scripts archivables** a `scripts/archive/`:
   - Los 5 scripts marcados en `archive/`.
2. **Mover JSONs huérfanos** a `reports/tables/archive/`.
3. **Archivar los parquets del FE descartado**:
   - `hex_risk_fe.parquet`, `hex_risk_fe_flags.parquet`,
     `hex_risk_bootstrap.parquet`, `hex_risk_bootstrap_v2.parquet`,
     `variance_by_year.parquet`.
4. **Tests unitarios** (`tests/`) para funciones críticas.
5. **environment.yml** con versiones fijadas.
