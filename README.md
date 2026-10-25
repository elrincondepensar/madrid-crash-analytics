# Madrid Road Safety 2019–2025

Severity prediction, spatial hotspot analysis and exposure-adjusted crash frequency for **137,328 traffic accidents** in Madrid, built entirely from open data.

> **Status:** Phases 1–10 closed. Phase 11 (consolidation: final maps, tests, documentation) in progress.
> Every methodological decision, including the ones that were later corrected, is recorded in [`docs/decisions.md`](docs/decisions.md).

---

## Summary of findings

| # | Finding | Evidence |
|---|---------|----------|
| 1 | **Injured vs. no-injury is predictable from what is known at the scene; fatal vs. serious is not.** The binary model reaches PR-AUC 0.81–0.83 and ROC-AUC 0.87–0.88 on the 2025 hold-out. For fatal accidents, PR-AUC stays at ≈ 0.005–0.009 (at best ~5–6× the base rate of 0.15 %), which is documented as a data ceiling, not forced. | Line A |
| 2 | **Hotspots exist at street scale, not at administrative scale.** Moran's I of accident counts is 0.68 on 250 m hexagons, 0.09 on neighbourhoods and 0.01 on districts. | Line C |
| 3 | **Hotspot structure is temporally stable.** HDBSCAN cluster sizes fitted on 2019–2022 correlate with the sizes observed afterwards (Spearman ρ 0.73–0.82), close to the ceiling set by sampling noise. | Line B |
| 4 | **Severity does not create its own geography.** Density maps of all accidents and of injured-only accidents are almost identical, and the injury rate has no significant High–High clusters after FDR correction at hexagon or district scale. | Lines B, C |
| 5 | **Accidents scale sublinearly with traffic.** The elasticity of accidents to traffic exposure is ≈ 0.44–0.47, so doubling traffic is associated with ≈ 1.36–1.39× accidents. This is a between-area association, **not** a causal estimate. | Line D |
| 6 | **COVID-19 reduced accidents more than traffic.** The 2020 year effect is −25 % accidents after controlling for exposure. | Line D |

Negative and null results are part of the project and are listed in [What did not work](#what-did-not-work).

---

## Research lines

- **A. Severity prediction**: 4-class ordinal target (no injury / minor / serious / fatal) and the binary and 3-class variants derived from it.
- **B. Spatial concentrations**: HDBSCAN clusters, kernel density on the road network, temporal persistence of clusters.
- **C. Local spatial statistics**: Local Moran's I (LISA) with spatial weights, permutation tests and FDR correction.
- **D. Exposure-adjusted frequency**: negative binomial regression of accident counts on traffic exposure, per 250 m hexagon and month.

---

## Data

Raw data is **not** included in the repository. Resources are discovered through the CKAN API of the Madrid open data portal (`package_show`), because download URLs do not follow a uniform pattern (decision D001).

| Source | Role in the project | Notes |
|--------|--------------------|-------|
| [Accidents, detail (300228)](https://datos.madrid.es/dataset/300228-0-accidentes-trafico-detalle) | Core dataset | One row per **person**; `num_expediente` identifies the accident. 2019–2025 has a homogeneous schema with UTM coordinates; 2010–2018 does not, so it is out of scope. 322,317 person rows → 137,328 accidents. |
| [Traffic history (208627)](https://datos.madrid.es/dataset/208627-0-transporte-ptomedida-historico) | **Primary traffic exposure** | 162 monthly ZIP files, 15-minute data, ~4,500 sensors per month, city-wide (`URB`, `M30`, `C30`). 84/84 months of 2019–2025. |
| [Measurement points (202468)](https://datos.madrid.es/dataset/202468-0-intensidad-trafico) | Sensor catalogue | Spatial index of the sensors. 99.03 % of sensor-months matched to a location. |
| [Permanent counts (300233)](https://datos.madrid.es/dataset/300233-0-aforo-trafico-permanentes) | Audited, **not used** in the final exposure model | ~60 stations; complete 2019–2024, only Jan–Mar 2025. |
| IMD by section (203962) | Plausibility check of exposure | Static 2012–2013 snapshot (1,961 points), so not usable as temporal exposure. |
| Intensity map (208223) | Evaluated and **rejected** as road network | 7,856 line segments, but it covers only main roads and the M-30 (74 % accident match at 100 m, 21–37 % in peripheral districts). |
| Street directory SHP (Geoportal) | Street type (`tipo_via`) by point-in-polygon | Name/rotulation polygons, not a line network. |
| [OpenStreetMap](https://www.openstreetmap.org/) via `osmnx` | **Road network** (95,437 edges) | Data © OpenStreetMap contributors, ODbL. `maxspeed` is present on only 37 % of edges; the rest is imputed by road type with an explicit flag. |

Check the license of each dataset on the portal before redistributing any derived data.

---

## Methodology

### Unit of observation and target

- Accident-level table built from person-level rows (`accidents_clean.parquet`: 137,328 rows; `persons_clean.parquet`: 322,317 rows).
- Severity is derived from the **worst injured person**, mapping `cod_lesividad` to severity **before** taking the maximum, because the raw code is not ordinal (D027).
- Class distribution: no injury 54.51 %, minor 35.03 %, serious 10.31 %, fatal 0.15 %.
- Validation checks enforced in code: `severity == fatal ⇔ n_fatalities ≥ 1`, no duplicate accident IDs, nothing outside 2019–2025.

### Prediction moment and leakage control

The model predicts severity **at the moment the patrol reports the accident** (decision D029). Only information available at that point is used: hour and month (cyclical), number and type of people and vehicles, accident type, weather, district, coordinates, and road-network features. Everything derived from injury outcome, alcohol or drug tests is excluded by name (D023, D031).

### Temporal validation

Train 2019–2022 · validation 2023–2024 · **test 2025**. Splits are persisted in a single file and reused by every later phase (D034, D081). Hyperparameter search uses leave-one-year-out cross-validation inside the training years, never random folds (D040, D086).

### Metrics

Classification metrics (F1, quadratic weighted kappa, ordinal MAE) plus **ranking metrics** (PR-AUC, Lift@k, Recall@k) for rare classes. With a 0.15 % positive class, argmax-based metrics reward collapsing to the majority classes. Confidence intervals come from a 500-sample bootstrap, and a difference below 0.02 F1 counts as a tie, in which case the simpler model is preferred (D032).

---

## Results

### A. Severity prediction

**Main model: injured vs. no injury** (test = 2025).

| Model | F1 (injured) | Recall | Precision | PR-AUC | ROC-AUC | Kappa |
|-------|-------------:|-------:|----------:|-------:|--------:|------:|
| **Logistic regression** (selected) | 0.764 | 0.838 | 0.703 | 0.813 | 0.871 | 0.564 |
| LightGBM | 0.770 | 0.843 | 0.710 | 0.826 | 0.878 | 0.576 |
| CatBoost | 0.766 | 0.830 | 0.712 | 0.817 | 0.873 | 0.571 |
| BART | 0.721 | 0.771 | – | 0.772 | – | 0.494 |

Differences among the first three are within bootstrap noise, so logistic regression is selected: one effective hyperparameter, interpretable coefficients, seconds to train (D095). For LightGBM, the top 5 % of accidents by predicted risk have a 92 % injury rate (lift 2.18).

**Secondary use cases.**

- **Serious or fatal ("grave")**: CatBoost with ADASYN and thresholds calibrated on validation gives recall 0.32 [0.30, 0.34], precision 0.26, PR-AUC 0.24.
- **Fatal**: not operational. PR-AUC(fatal) is ≈ 0.005–0.009 across flat, hierarchical, boosting and regularised linear models, with 116 fatal accidents in the training set. It is reported as a structural limit (D099).

**Road-network features** (OSM road class, speed limit, lanes, street type) were tested with a proper ablation. With validation 2023–2024 and test 2025, the improvement is **not statistically significant**; an apparent +44 % in an earlier split was an artefact of a small validation set with 24 fatal cases (D081, D083).

### B. Spatial concentrations

**HDBSCAN** on accident coordinates, fitted on the training years only:

| Variant | `min_cluster_size` | Clusters | Median effective radius | Noise share |
|---------|-------------------:|---------:|------------------------:|------------:|
| All accidents | 50 | 375 | 200 m | 0.414 |
| Injured only | 30 | 307 | 184 m | 0.449 |

**Persistence** is evaluated by assigning later accidents to the nearest training cluster, without refitting:

- Spearman ρ of cluster sizes (train vs. test): 0.73–0.82 for all clusters, 0.51–0.60 for the top 100.
- A random split inside the training years gives a sampling ceiling of ρ = 0.85 (all) / 0.77 (injured). The temporal split reaches **96 % / 100 % of that ceiling**, so there is no measurable loss of stability beyond sampling noise.
- The exact top-10 ranking is volatile (30–50 % overlap), while the top-20/30 set is operationally stable (60–75 % overlap).
- The centroid-assignment procedure itself introduces ~6–8 % intrinsic dispersion (ρ = 0.94 when re-assigning the training set).

**Kernel density** (bandwidth 260 m, chosen at the stability elbow and consistent with the HDBSCAN radius):

- It is a **2D Gaussian KDE evaluated at the midpoints of the 95,437 OSM edges**, *not* a shortest-path (network-constrained) KDE. The network-constrained version was considered and dropped on cost/benefit grounds (D105). An edge-weighted alternative gave an almost identical ranking (mean Spearman ρ = 0.994).
- HDBSCAN and KDE are **complementary**: cluster size and KDE density are only weakly rank-correlated (ρ = 0.11 all / 0.07 injured). The largest clusters sit in dense zones, but about 30–35 % of clusters are local anomalies in medium-density areas (D104).

### C. Local Moran's I

Queen contiguity weights (Rook/KNN fallback), 999 permutations, Benjamini–Hochberg FDR at α = 0.05.

| Spatial unit | Units | Moran's I (counts) | I of KDE ÷ I of counts | Reading |
|--------------|------:|-------------------:|-----------------------:|---------|
| **Hexagon 250 m** | 3,962 | **0.677** | 1.4× | Strong structure |
| Neighbourhoods | 131 | 0.092 | 8.7× | Weak |
| Districts | 21 | 0.014 | 28× | Indistinguishable from chance |
| Street name polygons | 9,285 | 0.013 | 71× | Over-fragmented |

This is a clear **modifiable areal unit problem (MAUP)**: spatial autocorrelation exists only at about 250 m. The hexagon size is robust: I = 0.624 / 0.677 / 0.711 for 200 / 250 / 300 m. The empirical-Bayes injury rate has no High–High cluster after FDR at hexagon scale (4 at neighbourhood scale), consistent with finding 4.

### D. Exposure-adjusted frequency

**Exposure.** Built from the 15-minute traffic history: `intensity per day = Σ intensity / number of days`, aggregated by sensor and then by hexagon-month. `intensidad` is an absolute count and the integration period varies from 1 to 15 minutes, which an earlier version got wrong and was corrected (D110). The result covers **929 hexagons (23.4 % of the 3,962)** over 84 months, with a hexagon-month median of 20,942 vehicles/day. Hexagons without a sensor are left missing, not set to zero. It reproduces the COVID drop (2020 ≈ −31 % vs. 2019).

**Model.** Negative binomial (NB2) on the hexagon × month panel (74,625 observations) with log-exposure as a free covariate, year and month fixed effects, dominant street type and mean OSM road class.

| | Injured | All accidents |
|---|--------:|--------------:|
| Coefficient on log-exposure (N4, primary) | **0.467** | **0.440** |
| Implied effect of doubling traffic | ×1.38 | ×1.36 |
| Same coefficient with hexagon fixed effects | 0.166 | 0.147 |

- The offset-equals-1 specification is **rejected**: accidents do not scale proportionally with traffic.
- 76.4 % of the variance of log-exposure is between hexagons and 23.6 % within, concentrated in 2020. The primary model and the fixed-effects model therefore answer different questions: the former compares areas, the latter is identified mostly from the COVID shock.
- Residual spatial autocorrelation remains (Moran's I ≈ 0.44, down from 0.677 for raw counts), i.e. local heterogeneity is not explained by exposure and road class alone.

---

## What did not work

Kept in the repository on purpose, because they shaped the final design.

- **Fatal classification**: no formulation (flat, hierarchical, resampled, regularised linear) beat the structural ceiling. A stopping rule was fixed in advance (D099).
- **BART**: worse than logistic regression and boosting in all three formulations (D096).
- **SMOTE / Borderline-SMOTE / ADASYN**: marginal gains within bootstrap noise; `class_weight='balanced'` lowered PR-AUC for the fatal class (D038, D091).
- **Hexagon fixed-effect "risk map"**: discarded. The coefficients are rank-deficient (street type is constant within a hexagon), and a bootstrap showed that most of their dispersion is an identification artefact (ρ between original and bootstrap = 0.005, D110).
- **Network features for severity**: no significant gain (see line A).
- **Three candidate road networks** (intensity KML, street directory, IMD points) were rejected before settling on OSM.

---

## Limitations

- **Association, not causation.** No result supports a causal claim about traffic volume, street type or interventions.
- **Exposure covers 23 % of the city.** The frequency model says nothing about hexagons without a traffic sensor, and sensor placement is not random.
- **The "no injury" class is partly an assumption.** In 41.86 % of accidents no person has a recorded injury code; they are classified as no injury. The consistency check with registered fatalities holds, but the rest is not independently verifiable.
- **Geolocation.** Accidents are assigned to the nearest OSM edge (median distance 2.2 m). A small distance does not guarantee the correct street at intersections.
- **Network KDE** is a 2D approximation evaluated on the network (see line B).
- **Fatal-accident analysis is underpowered** (116 training cases, 32–34 in the test year); intervals are wide.
- **Imputed speed limits.** 36.6 % of edges use a road-type default, and a flag is passed to the models.
- **Temporal scope.** Permanent counts end in March 2025; the frequency model is built from the sensor history, which covers all months.

---

## Repository structure

```
madrid-road-safety/
├── README.md
├── docs/
│   └── decisions.md          # full decision log (D001 …)
├── src/madrid_safety/        # reusable code (I/O, auditing, JSON serialisation, …)
├── scripts/                  # numbered scripts: NN_name.py
├── data/
│   ├── raw/                  # not versioned
│   └── processed/            # parquet / GeoParquet outputs
├── reports/
│   ├── tables/               # JSON metrics: NN_*.json
│   └── figures/              # maps and plots
└── tests/                    
```

Conventions: the number of a script matches the number of its outputs in `reports/` (D015); every JSON is written through `dumps_jsonable` (D021); geospatial parquet files are standard GeoParquet and are read through `madrid_safety.io` (D107).

Key artefacts: `accidents_clean.parquet`, `persons_clean.parquet`, `splits_v2.parquet`, `accidents_with_network_features.parquet`, `network_osm.parquet`, `network_kde_final.parquet`, `exposure_hex250m_monthly.parquet`, `panel_hex_month_n_injured.parquet`, `moran_{hex250m,barrios,distritos,callejero}.parquet`.

---

## Reproducing the analysis

> The commands below are a template; adjust them to the final repository layout.

```bash
git clone https://github.com/elrincondepensar/madrid-crash-analytics.git
cd madrid-road-safety

# Environment (Python 3.11)
pip install -e 
```

Main libraries: `pandas`, `geopandas`, `scikit-learn ≥ 1.7`, `lightgbm`, `catboost`, `hdbscan`, `imbalanced-learn`, `osmnx`, `libpysal` / `esda`, `statsmodels`, `pymc-bart`.

```bash
# Run the numbered scripts in order; each phase reads the previous outputs
python scripts/[NN]_[name].py
```

- **Data download:** handled by the first scripts through the CKAN API; nothing needs to be downloaded manually. Large files (traffic history ZIPs) take a while.
- **Randomness:** seeds are fixed in the configuration. Splits are read from `splits_v2.parquet` and must not be regenerated, because any change invalidates downstream results (D034).

---

## Citation and license

- Data: Madrid City Council open data portal (check each dataset's license) and OpenStreetMap contributors (ODbL).