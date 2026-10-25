# Phase 10 · Iter 1 — Poisson baseline with exposure offset
**Model:** `E[accidents] = exp(α) × exposure`. Null model with offset = log(exposure_month). Fit via Poisson GLM; NB2 if overdispersed.

## Response: `n_injured`
- n_obs = 74,625, n_hexes = 929
- **Poisson**: AIC=179649.0, α=-1.41e+01, overdispersion=30.53
- **NB2**: AIC=nan, alpha_param=inf
- **Best**: `poisson`
- **Moran on residuals**: I=0.2159 (p=0.0010, n=922)
  - Fase 8 counts I=0.677. Reduction: 0.461. If large → exposure explains spatial pattern. If small → residual spatial structure remains.

### By year fit
| year | actual | predicted | ratio |
| --- | ---: | ---: | ---: |
| 2019 | 9,018 | 8668 | 0.961 |
| 2020 | 5,633 | 5863 | 1.041 |
| 2021 | 7,054 | 7576 | 1.074 |
| 2022 | 7,791 | 6921 | 0.888 |
| 2023 | 7,736 | 7848 | 1.015 |
| 2024 | 7,772 | 8021 | 1.032 |
| 2025 | 7,686 | 7793 | 1.014 |

## Response: `n_all`
- n_obs = 74,625, n_hexes = 929
- **Poisson**: AIC=279586.7, α=-1.33e+01, overdispersion=44.83
- **NB2**: AIC=nan, alpha_param=inf
- **Best**: `poisson`
- **Moran on residuals**: I=0.2158 (p=0.0010, n=922)
  - Fase 8 counts I=0.677. Reduction: 0.461. If large → exposure explains spatial pattern. If small → residual spatial structure remains.

### By year fit
| year | actual | predicted | ratio |
| --- | ---: | ---: | ---: |
| 2019 | 18,276 | 18866 | 1.032 |
| 2020 | 11,545 | 12762 | 1.105 |
| 2021 | 14,970 | 16488 | 1.101 |
| 2022 | 16,730 | 15064 | 0.900 |
| 2023 | 17,323 | 17082 | 0.986 |
| 2024 | 17,634 | 17458 | 0.990 |
| 2025 | 18,201 | 16960 | 0.932 |

## Interpretation
- **n_injured**: exposure explains most of the spatial autocorrelation (I: 0.677 → 0.216, Δ=+0.461). Los hotspots de Fase 8 son esencialmente consecuencia del volumen de tráfico.
- **n_injured**: overdispersion=30.53. Best model: `poisson` (AIC Poisson=179649.0, NB2=nan).
- **n_all**: exposure explains most of the spatial autocorrelation (I: 0.677 → 0.216, Δ=+0.461). Los hotspots de Fase 8 son esencialmente consecuencia del volumen de tráfico.
- **n_all**: overdispersion=44.83. Best model: `poisson` (AIC Poisson=279586.7, NB2=nan).