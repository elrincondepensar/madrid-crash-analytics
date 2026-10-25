# Phase 11 · Iter 4 — Refactor log

_Executed: 2026-10-25T01:20:27_

## Archive — scripts

| src | status |
| --- | --- |
| `scripts\41b_poisson_diagnostics.py` | moved |
| `scripts\41c_phase10_refit.py` | moved |
| `scripts\44_phase10_diagnostics_v2.py` | moved |
| `scripts\44b_phase10_bootstrap_clean.py` | moved |
| `scripts\44c_bootstrap_centered.py` | moved |

## Archive — tables

| src | status |
| --- | --- |
| `reports\tables\01_schema_samples.json` | moved |
| `reports\tables\01_url_availability.json` | moved |
| `reports\tables\17_callejero_load_error.json` | moved |
| `reports\tables\17_callejero_meta.json` | moved |
| `reports\tables\17_imd_fixed.json` | moved |
| `reports\tables\28_opcion_c_bart.json` | moved |

## Archive — parquets

| src | status |
| --- | --- |
| `data\processed\hex_risk_fe.parquet` | moved |
| `data\processed\hex_risk_fe_flags.parquet` | moved |
| `data\processed\hex_risk_bootstrap.parquet` | moved |
| `data\processed\hex_risk_bootstrap_v2.parquet` | moved |
| `data\processed\variance_by_year.parquet` | moved |

## Tests creados

- `tests\conftest.py`: created
- `tests\test_io.py`: created
- `tests\test_data_audit.py`: created

## Environment

- `environment.yml`: created

## Verificación manual (comandos)

```bash
conda activate sendai
pytest tests/ -v
```

## Notas

- Los `scripts/archive/README.md`, `reports/tables/archive/README.md` y `data/processed/archive/README.md` documentan la razón de cada archivo.
- Los scripts archivados siguen siendo válidos como referencia. Están fuera del flujo activo pero son legibles.
- Los 5 parquets archivados corresponden al FE descartado en D110. Los mapas de Fase 11 NO los usan (usan `moran_*.parquet` y N4 residuals).
- Los tests solo cubren funciones que ya están en `src/`. Para las funciones inline en scripts (`_worst_lesividad`, `parse_number_with_range_sanity`, `year_to_split`, etc.), el test se pospone a Fase 12 o a un refactor futuro de extracción a `src/`.
- `environment.yml` tiene versiones mínimas (>=), no pin exacto. Si se necesita reproducibilidad bit a bit, usar `conda env export` después de instalar.
