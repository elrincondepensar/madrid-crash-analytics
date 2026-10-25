"""
Script 49 — Phase 11 iter 5: README + docs consolidation.

Generates 4 files:
  - README.md                    (Spanish, main entry point)
  - README_EN.md                 (English, executive summary)
  - docs/methodology.md          (10 phases explained)
  - docs/data_dictionary.md      (columns from principal parquets)

Does NOT touch:
  - docs/decisions.md (curated manually, D001-D110)
  - Any script, parquet, test, or figure

Reads from:
  - reports/tables/{28_opcion_c,37_moran,41_poisson_baseline,
                    43_fe_diagnostics,46_maps_exposure}.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety.data_audit import dumps_jsonable  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
DATA = ROOT / "data" / "processed"
DOCS = ROOT / "docs"
OUT_JSON = TABLES / "49_readme_docs.json"

log = logging.getLogger("phase11_49")


def _load(name: str) -> dict:
    p = TABLES / name
    if not p.exists():
        log.warning(f"  missing {name}")
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        log.warning(f"  failed to parse {name}: {e}")
        return {}


def _get(d, *path, default="n/a"):
    cur = d
    for k in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def _fmt(v, digits=3):
    if v == "n/a" or v is None:
        return "n/a"
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return str(v)


def _year_mean(years, year):
    if not isinstance(years, list):
        return "n/a"
    for y in years:
        if isinstance(y, dict) and y.get("year") == year:
            return y.get("mean_exposure", "n/a")
    return "n/a"


def collect_metrics() -> dict:
    log.info("Loading metric JSONs ...")
    f5 = _load("28_opcion_c.json")
    f8 = _load("37_moran.json")
    f10a = _load("41_poisson_baseline.json")
    f10b = _load("43_fe_diagnostics.json")
    f11 = _load("46_maps_exposure.json")

    m = {}

    m["f5_lgbm_f1"] = _fmt(_get(f5, "lgbm", "test", "f1_injured"))
    m["f5_lgbm_prauc"] = _fmt(_get(f5, "lgbm", "test", "pr_auc_injured"))
    m["f5_lgbm_rocauc"] = _fmt(_get(f5, "lgbm", "test", "roc_auc_injured"))
    m["f5_lr_f1"] = _fmt(_get(f5, "lr", "test", "f1_injured"))
    m["f5_lgbm_lift5"] = _fmt(_get(f5, "lgbm", "test", "lift_at_5"))
    m["f5_lgbm_recall20"] = _fmt(_get(f5, "lgbm", "test", "recall_at_20"))

    m["f8_hex_I"] = _fmt(_get(f8, "hex250m", "variables", "counts_all", "moran_I"))
    m["f8_hex_HH"] = _fmt(_get(f8, "hex250m", "variables", "counts_all", "pct_HH"), 4)
    m["f8_barrios_I"] = _fmt(_get(f8, "barrios", "variables", "counts_all", "moran_I"))
    m["f8_distritos_I"] = _fmt(_get(f8, "distritos", "variables", "counts_all", "moran_I"))
    m["f8_callejero_I"] = _fmt(_get(f8, "callejero", "variables", "counts_all", "moran_I"))
    m["f8_hex_kde_I"] = _fmt(_get(f8, "hex250m", "variables", "kde_all", "moran_I"))

    bs_raw = _get(f10b, "variance_decomposition", "between_share", default=None)
    try:
        bs_f = float(bs_raw) if bs_raw not in (None, "n/a") else None
    except (TypeError, ValueError):
        bs_f = None
    if bs_f is not None:
        m["f10_between_pct"] = _fmt(bs_f * 100, 0)
        m["f10_within_pct"] = _fmt((1 - bs_f) * 100, 0)
        m["f10_between_pct_1d"] = _fmt(bs_f * 100, 1)
        m["f10_within_pct_1d"] = _fmt((1 - bs_f) * 100, 1)
    else:
        for k in ("f10_between_pct", "f10_within_pct",
                  "f10_between_pct_1d", "f10_within_pct_1d"):
            m[k] = "n/a"

    m["f10_coef_inj"] = _fmt(_get(f10a, "n_injured", "poisson", "alpha"))
    m["f10_fe_full_coef"] = _fmt(_get(f10b, "summary", "n_injured", "FE_full_coef"))

    ey = f11.get("exposure_years") if isinstance(f11, dict) else None
    m["f9_mean_2019"] = _fmt(_year_mean(ey, 2019), 0)
    m["f9_mean_2020"] = _fmt(_year_mean(ey, 2020), 0)
    m["f11_covid_ratio"] = _fmt(_get(f11, "covid_ratio", "median_ratio"))
    m["f11_covid_pct"] = _fmt(_get(f11, "covid_ratio", "median_pct_change"), 1)
    m["f11_hexes_with_data"] = _get(f11, "covid_ratio", "n_hexes", default=929)

    return m


PARQUETS_TO_DESCRIBE = {
    "accidents_clean.parquet": "One row per accident (severity, location, district, covariates). 137,328 rows.",
    "persons_clean.parquet": "One row per person involved. 322,317 rows.",
    "splits.parquet": "Train/val/test assignment by num_expediente.",
    "network_osm.parquet": "OSM drive network (edges as LineStrings).",
    "moran_hex250m.parquet": "LISA clusters on hex250m grid.",
    "exposure_hex250m_monthly.parquet": "Monthly traffic exposure per hex and per year.",
    "panel_hex_month_n_injured.parquet": "Hex-month panel with count + exposure + covariates (Phase 10).",
}


def introspect_parquet_columns(name: str) -> list[tuple[str, str]]:
    p = DATA / name
    if not p.exists():
        return []
    try:
        schema = pq.read_schema(p)
        return [(f.name, str(f.type)) for f in schema]
    except Exception as e:  # noqa: BLE001
        log.warning(f"  failed to read schema of {name}: {e}")
        return []


def build_readme(m: dict) -> str:
    return f"""# Madrid Road Safety — análisis de siniestralidad 2019-2025

Análisis de la siniestralidad vial en Madrid capital a partir de los
datos abiertos del Ayuntamiento y OpenStreetMap. Cubre 137,328
accidentes en el periodo 2019-2025, con modelos de gravedad,
detección de hotspots espaciales, y un modelo de frecuencia con
exposición al tráfico.

## Hallazgos clave

1. **Modelo operativo de heridos (injury)**: LGBM binario alcanza
   F1 = {m["f5_lgbm_f1"]} y PR-AUC = {m["f5_lgbm_prauc"]} en test.
   Lift@5 = {m["f5_lgbm_lift5"]} (el top 5% de accidentes concentra
   2.18× la tasa de heridos media).

2. **Fatal es estructuralmente inviable** con las features disponibles.
   PR-AUC(fatal) techo ~0.009 (≈6× base rate). No hay margen con
   covariables M2 (contexto, hora, tipo de vía).

3. **MAUP fuerte en recuentos de accidentes**: solo hex250m captura
   estructura espacial. Autocorrelación de Moran cae de I = {m["f8_hex_I"]}
   (hex250m) a I = {m["f8_barrios_I"]} (barrios) y I = {m["f8_distritos_I"]}
   (distritos). El KDE conserva la estructura (I = {m["f8_hex_kde_I"]})
   porque es un suavizado, no un recuento.

4. **Exposición al tráfico sublineal**: coef de N4 sobre `log_exposure`
   = 0.467 (injured). Doblar el tráfico se asocia con ~1.39× accidentes,
   no 2×. La varianza de exposición es {m["f10_between_pct"]}%
   between-hex y {m["f10_within_pct"]}% within-hex.

5. **COVID cuantificado independientemente**: la exposición cae un
   {m["f11_covid_pct"]}% en 2020 vs 2019. Los accidentes caen más
   (year FE -0.29 ≈ -25%): hubo cambio de comportamiento además de
   menos vehículos.

## Contenido

- [Cómo reproducir](#cómo-reproducir)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Metodología](#metodología)
- [Hallazgos detallados](#hallazgos-detallados)
- [Limitaciones](#limitaciones)

## Cómo reproducir

```bash
# 1. Crear entorno
conda env create -f environment.yml
conda activate sendai

# 2. Generar todos los artefactos (parquets + JSONs + figuras)
# Los scripts están numerados en el orden de ejecución (01..47).
# Cada script es idempotente y toma unos minutos.
bash run_all.sh  # o ejecutar scripts/NN_*.py en orden
```

Los datasets de entrada se descargan automáticamente desde el portal
de datos abiertos del Ayuntamiento de Madrid (CKAN) y desde OSM
(vía osmnx). No hay credenciales requeridas.

## Estructura del repositorio

```text
madrid-road-safety/
├── README.md              # este fichero
├── README_EN.md           # versión corta en inglés
├── environment.yml        # dependencias
├── data/
│   ├── raw/               # descargas crudas (no versionadas)
│   └── processed/         # parquets procesados
├── src/madrid_safety/     # utilidades reutilizables
├── scripts/               # pipeline numerado (01..50)
│   └── archive/           # scripts deprecados con README
├── tests/                 # pytest
├── reports/
│   ├── tables/            # JSONs de auditoría y métricas
│   ├── figures/final/     # 25 PNGs
│   └── *.md               # reportes intermedios por fase
└── docs/
    ├── decisions.md       # registro D001-D110
    ├── methodology.md     # las 10 fases
    └── data_dictionary.md # columnas de cada parquet
```

## Metodología

10 fases documentadas en docs/methodology.md:

1. Auditoría de fuentes
2. Base analítica (accidentes + personas)
3. EDA + splits temporales
4. Red viaria (OSM + callejero)
5. Modelos de gravedad (binario injured)
6. HDBSCAN (hotspots)
7. Network KDE
8. Moran local (MAUP)
9. Exposición al tráfico (208627)
10. Poisson/NB con exposición

Decisiones técnicas detalladas en docs/decisions.md (D001-D110).

## Hallazgos detallados

### Línea A — Gravedad

- Modelo principal: LGBM binario injured/no_injury.
- F1 = {m["f5_lgbm_f1"]}, PR-AUC = {m["f5_lgbm_prauc"]},
  ROC-AUC = {m["f5_lgbm_rocauc"]}.
- LR como alternativa interpretable (F1 = {m["f5_lr_f1"]}).
- Fatal: no predictible con features M2. Documentado como
  limitación estructural (D099).

### Línea B/C — Patrones espaciales

- HDBSCAN: 375 clusters (all), 307 (injured). Parámetros
  fijados manualmente (D101).
- KDE: bw = 260 m (D103). Option C (raw points) preferida a
  Option B (edge-weighted) por simplicidad (D105).
- Moran: hex250m I = {m["f8_hex_I"]}, HH = {m["f8_hex_HH"]}
  (proporción). Barrios y distritos no capturan la estructura
  espacial. La severidad no tiene geografía propia (D108).

### Línea D — Frecuencia con exposición

- Exposición: 208627 histórico (84/84 meses 2019-2025).
  Cobertura: {m["f11_hexes_with_data"]} hexes con sensor (23.4% de Madrid).
- Modelo N4: NB2 con log_exposure + tipo_via + year FE + month FE.
  Coef de exposición = 0.467 (injured).
- Varianza de exposición: {m["f10_between_pct_1d"]}%
  between-hex. La componente within-hex está dominada por COVID.
- FE por hex descartado por rank-deficiency: el mapa de riesgo
  residual que se entrega es el de N4 residuals + LISA Fase 8.

## Limitaciones

- **Fatal inviable**: el modelo binario injured es el techo
  operativo. Se documenta fatal como imposible con los datos
  actuales (D099).
- **Cobertura de exposición**: solo el 23.4% de los hexes tienen
  sensor de tráfico. El resto queda fuera del modelo Poisson.
- **Residual Moran I = 0.44**: la exposición y las covariables
  explican ~35% de la autocorrelación espacial original. Hay
  heterogeneidad espacial real no capturada.
- **Calidad del modelo FE**: los coeficientes por hex no son
  estables bajo bootstrap. Se descarta el mapa de riesgo por
  coeficientes FE.
- **Datos pre-2019**: el histórico de accidentes 2010-2018 usa un
  schema incompatible (sin coordenadas). Fuera de alcance.

## Licencia y contacto

Uso educativo y de portfolio. Datos originales:

- Portal de datos abiertos del Ayuntamiento de Madrid
  (https://datos.madrid.es).
- OpenStreetMap contributors (ODbL).

Contacto: ver perfil del repositorio.
"""


def build_readme_en(m: dict) -> str:
    return f"""# Madrid Road Safety (2019-2025)

Analysis of road traffic accidents in Madrid using municipal open data
and OpenStreetMap. 137,328 accidents, 7 years, binary injury prediction,
spatial hotspot detection, and frequency modeling with traffic exposure.

## Key findings

1. **Injury prediction (binary)**: LGBM achieves F1 = {m["f5_lgbm_f1"]},
   PR-AUC = {m["f5_lgbm_prauc"]} on 2025 test. Lift@5 = {m["f5_lgbm_lift5"]}.

2. **Fatal is structurally infeasible**: PR-AUC ceiling ~0.009 (~6× base
   rate). No feature set improves beyond noise.

3. **MAUP in accident counts**: spatial autocorrelation collapses from
   Moran I = {m["f8_hex_I"]} (250m hex) to I = {m["f8_barrios_I"]}
   (neighbourhoods) and {m["f8_distritos_I"]} (districts). Only 250m
   hexes capture the structure.

4. **Exposure-accident relationship is sublinear**: doubling traffic is
   associated with ~1.39× accidents, not 2×. Effectively: elasticity ≈ 0.47.

5. **COVID quantified**: exposure drops {m["f11_covid_pct"]}% in 2020 vs
   2019; accidents drop more (−25%), indicating behavioral change beyond
   traffic reduction.

## Reproduce

```bash
conda env create -f environment.yml
conda activate sendai
bash run_all.sh
```

## Documentation

- `docs/methodology.md` — 10-phase pipeline.
- `docs/decisions.md` — technical decisions (D001-D110).
- `docs/data_dictionary.md` — parquet columns.

## License

Educational / portfolio use. Source data: Madrid open data portal
(https://datos.madrid.es) and OpenStreetMap contributors (ODbL).
"""


def build_methodology(m: dict) -> str:
    return f"""# Methodology

The project is structured in 10 phases, each closing with a
decision record (see decisions.md) and producing a set of artefacts
(parquets, JSONs, figures).

## Phase 1 — Source audit

Discovered and characterized 4 data sources:

- Accidents (300228): 17 CSVs, 2010-2025.
- Permanent counts (300233): 89 CSVs, 2018-2025.
- Historical traffic (208627): 162 ZIPs, 15-min granularity.
- Measurement points (202468): 307 resources.

Tools: CKAN package_search + package_show, requests,
lxml for KML/ZIP parsing, pandas for CSV.

## Phase 2 — Analytic base

Merged 7 accident-year CSVs (2019-2025) into a normalized schema:

- `accidents_clean.parquet`: 137,328 rows × 31 cols.
- `persons_clean.parquet`: 322,317 rows × 34 cols (one row per person).

Severity derived by "worst victim wins" using a non-ordinal mapping
of cod_lesividad (bug D027: 4=fallecido, 14=sin asistencia, so
max over raw codes is wrong).

## Phase 3 — EDA and splits

- Time-based splits: train 2019-2022, val 2023-2024, test 2025.
- COVID 2020 affects volume, not severity composition.
- Baselines: majority, LR multinomial. Neither works well because
  fatal (0.15%) collapses under F1/kappa. Switched to ranking
  metrics (Lift@k, Recall@k).

## Phase 4 — Road network

- OSM drive network via osmnx: 95,437 edges, 9,809 km.
- Join accidents → OSM: 100% match, median distance 2.2 m.
- Derived features: edge_highway, edge_maxspeed, edge_lanes,
  tipo_via (via point-in-polygon against callejero rotulación).
- Conclusion (D083): network features do NOT improve fatal
  ranking. Retained for Phase 6-8 (spatial analysis).

## Phase 5 — Severity models

Three formulations:

- **Option A (4 classes)**: ceiling reached. Fatal PR-AUC ~0.009.
  Kappa ~0.52 with LGBM.
- **Option B (3 classes: no_injury/minor/grave)**: CatBoost + ADASYN
  with threshold tuning. recall_grave = 0.32.
- **Option C (binary injured/no_injury)**: best result. LGBM
  reaches F1 = {m["f5_lgbm_f1"]}, PR-AUC = {m["f5_lgbm_prauc"]}.
  Deployed as primary model.

Discarded with evidence: BART (D096), hierarchical Bayes (D098),
SMOTE (D056), Ridge/Lasso (D100).

## Phase 6 — HDBSCAN

- Grid search over mcs/ms. Manual selection after audit (D101):
  all=50/10 → 375 clusters; injured=30/10 → 307 clusters.
- Persistence validation (script 32/33): clusters are stable
  (ρ_train→test = 0.82) with random-split ceiling 0.85.

## Phase 7 — Network KDE

- Bandwidths: 100-500 m. Primary bw = 260 m (D103, aligned with
  HDBSCAN radius and KDE-HDBSCAN consistency).
- Two options compared: raw-point KDE (C) vs edge-weighted KDE (B).
  C wins (ρ=0.994, Jaccard=0.982).
- Key finding (D104): KDE and HDBSCAN are complementary, not
  redundant. ρ(cluster_size, KDE_density) = 0.11.

## Phase 8 — Moran local (LISA)

- Units: hex250m (principal), barrios, distritos, callejero.
- Queen contiguity weights, 999 permutations, FDR α=0.05.

MAUP confirmed:

| unit | counts_all I | kde_all I |
| --- | --- | --- |
| hex250m | {m["f8_hex_I"]} | {m["f8_hex_kde_I"]} |
| barrios | {m["f8_barrios_I"]} | 0.80 |
| distritos | {m["f8_distritos_I"]} | 0.39 |
| callejero | {m["f8_callejero_I"]} | 0.96 |

Only hex250m captures spatial structure in raw counts.
Injury rate has NO geography at any scale.

## Phase 9 — Traffic exposure

- Primary source: 208627 historical traffic (84/84 months 2019-2025).
- Correlation id (208627) → id (202468 catalog) gives UTM coords
  for 99.03% of sensors.
- Corrected aggregation (D110): intensity_day = sum(intensidad)/days.
  NOT a rate.
- Output: 929 hexes with monthly exposure (23.4% of Madrid).

## Phase 10 — Poisson/NB with exposure

- Panel: hex250m × month, 74,625 obs, 929 hexes.
- Model N4: NB2 with log_exposure + tipo_via + year/month FE.
  Coef of exposure: 0.467 (injured), sublinear.
- tipo_via hierarchy (interpretable): Plaza +1.64 > Calle +1.19 >
  Avenida +0.82 > Carretera +0.30.
- COVID: year FE 2020 = -0.29 (−25% accidentes).
- Variance decomposition: {m["f10_between_pct_1d"]}% between
  hex, {m["f10_within_pct_1d"]}% within.
- FE por hex descartado (D110): rank-deficiency + bootstrap
  instability (ρ=0.005).
- Residual Moran I = 0.44 (heterogeneity not captured).

## Phase 11 — Consolidation

- Maps LISA (17 PNGs), exposure + residuals maps (8 PNGs).
- Artifact inventory + physical archiving of deprecated files.
- Tests (6, all passing).
- This document + README + final report.

Total: 47 primary/iteration scripts (+ 5 archived), 25 figures,
~30 JSON tables, 25 parquets (data).
"""


def build_data_dictionary() -> str:
    sections = [
        "# Data dictionary\n",
        "Column schemas for the principal parquets in `data/processed/`.\n",
        "Types are PyArrow type strings.\n",
    ]
    for name, description in PARQUETS_TO_DESCRIBE.items():
        cols = introspect_parquet_columns(name)
        sections.append(f"\n## {name}\n\n{description}\n")
        if not cols:
            sections.append("\n_(not found or unreadable)_\n")
            continue
        sections.append("\n| column | type |\n| --- | --- |\n")
        for c_name, c_type in cols:
            sections.append(f"| {c_name} | {c_type} |\n")
    return "".join(sections)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    metrics = collect_metrics()
    log.info(f"Collected {len(metrics)} metrics")

    DOCS.mkdir(parents=True, exist_ok=True)

    log.info("Writing README.md ...")
    (ROOT / "README.md").write_text(build_readme(metrics), encoding="utf-8")
    log.info("  saved README.md")

    log.info("Writing README_EN.md ...")
    (ROOT / "README_EN.md").write_text(build_readme_en(metrics), encoding="utf-8")
    log.info("  saved README_EN.md")

    log.info("Writing docs/methodology.md ...")
    (DOCS / "methodology.md").write_text(build_methodology(metrics), encoding="utf-8")
    log.info("  saved docs/methodology.md")

    log.info("Writing docs/data_dictionary.md ...")
    (DOCS / "data_dictionary.md").write_text(build_data_dictionary(), encoding="utf-8")
    log.info("  saved docs/data_dictionary.md")

    result = {
        "metrics_used": metrics,
        "files_generated": [
            "README.md",
            "README_EN.md",
            "docs/methodology.md",
            "docs/data_dictionary.md",
        ],
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(dumps_jsonable(result), encoding="utf-8")
    log.info(f"saved {OUT_JSON}")

    log.info("=" * 72)
    log.info("Generated 4 files. Review README.md for narrative quality.")


if __name__ == "__main__":
    main()