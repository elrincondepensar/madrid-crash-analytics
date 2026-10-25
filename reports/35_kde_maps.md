# Phase 7 · Iter 2 — KDE maps & HDBSCAN cross-check

_Primary bandwidth: 260 m (D103). Reference: 500 m. Colormap: log-normalized KDE intensity on the OSM network. HDBSCAN clusters from D102._

## Maps generated

Total: 12 figures under `reports/figures/`.

## HDBSCAN cross-check

For each variant, we ask: **are the HDBSCAN clusters concentrated in the highest-KDE areas, or do they reveal structure not captured by the global intensity surface?**

- `rho_size_vs_kde_median`: Spearman rho between cluster `n_train` and the median KDE density of its edges. High positive → big clusters live on high-KDE edges (redundant). Low or negative → clusters capture something the raw density doesn't.
- `top_k_frac_dense`: fraction of the top-20 clusters whose median density sits in the top-20% of the network. High → the top clusters are in obviously dense areas. Low → some top clusters are in moderately dense areas.
- `accidents_in_persisted_frac_dense`: fraction of accidents belonging to persisted clusters that live on top-20% KDE edges.

| variant | rho(size, kde) | top-k frac dense | persisted frac dense | acc in pers. frac dense |
| --- | ---: | ---: | ---: | ---: |
| all | 0.113 | 0.700 | 0.651 | 0.661 |
| injured | 0.072 | 0.900 | 0.713 | 0.706 |

## Interpretation

- **`all`**: HDBSCAN adds *substantial* structure not captured by KDE (rho=0.11, 66% of persisted accidents on top-20% KDE). The hotspot analysis carries information the intensity map misses.
- **`injured`**: HDBSCAN adds *substantial* structure not captured by KDE (rho=0.07, 71% of persisted accidents on top-20% KDE). The hotspot analysis carries information the intensity map misses.

## Notes

- **Accident → edge assignment reconstructed in this script.** Uses cKDTree on edge midpoints (D077 remediation). Documented limitation: distance to midpoint ≠ distance to line; negligible for edges < 500 m, potentially >5 m for long arterial edges. Result saved in `accidents_with_edges.parquet`.
- **Density percentiles** are computed per (bw, period, severity) from the full 95k network. Top-20% is used as a fixed threshold to avoid cherry-picking.
- **KDE is descriptive, not predictive.** No refit on train/val/test in the ML sense — the surface in each period is an independent smooth of that period's accidents. Comparisons across periods are descriptive stability, not out-of-sample prediction.
