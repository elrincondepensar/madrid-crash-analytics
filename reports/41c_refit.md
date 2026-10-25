# Phase 10 · Iter 1c — Refit with fixed NB2

## Interpretación correcta del test α=1

El 41b reportó mal: **rechazamos α=1** (z=-92 / -138, p≈0). El offset con coef=1 **está mal especificado**. El modelo correcto usa `log_exposure` como covariable libre. Coef estimado ≈ **0.53**, sublineal.

## Response: `n_injured`

| model | AIC | ΔAIC vs N0 | α_exp | α_param |
| --- | ---: | ---: | ---: | ---: |
| N0 | nan | +nan | 0.5326 | 0.000 |
| N1 | 165139 | +nan | 0.5355 | 0.642 |
| N2 | 164933 | +nan | 0.5199 | 0.634 |
| N3 | 170476 | +nan | 0.4847 | 0.000 |
| N4 | 162600 | +nan | 0.4736 | 0.539 |

**Best: N4** (AIC=162599.6)
- Residual Moran I: 0.2156 (p=0.0020)

**Fit by year:**

| year | actual | predicted | ratio |
| --- | ---: | ---: | ---: |
| 2019 | 9018 | -3194 | -0.354 |
| 2020 | 5633 | -8366 | -1.485 |
| 2021 | 7054 | -6202 | -0.879 |
| 2022 | 7791 | -4946 | -0.635 |
| 2023 | 7736 | -5714 | -0.739 |
| 2024 | 7772 | -5972 | -0.768 |
| 2025 | 7686 | -5991 | -0.779 |

## Response: `n_all`

| model | AIC | ΔAIC vs N0 | α_exp | α_param |
| --- | ---: | ---: | ---: | ---: |
| N0 | 260549 | +0 | 0.5246 | 0.000 |
| N1 | 241669 | -18880 | 0.5067 | 0.569 |
| N2 | 241199 | -19350 | 0.4914 | 0.560 |
| N3 | 240737 | -19811 | 0.4715 | 0.550 |
| N4 | 237672 | -22877 | 0.4428 | 0.486 |

**Best: N4** (AIC=237671.6)
- Residual Moran I: 0.1380 (p=0.0010)

**Fit by year:**

| year | actual | predicted | ratio |
| --- | ---: | ---: | ---: |
| 2019 | 18276 | 4430 | 0.242 |
| 2020 | 11545 | -524 | -0.045 |
| 2021 | 14970 | 2149 | 0.144 |
| 2022 | 16730 | 3442 | 0.206 |
| 2023 | 17323 | 3467 | 0.200 |
| 2024 | 17634 | 3476 | 0.197 |
| 2025 | 18201 | 3923 | 0.216 |

## Conclusions

- **n_injured**: best model `N4`, AIC=162599.6. Coef on `log_exposure` = **0.4736** (sublinear: doubling traffic → 1.39× accidents). NB2 alpha = 0.53856676722769.
  Residual Moran I = 0.2156 (p=0.0020). Spatial structure remains after accounting for exposure and year/month.
- **n_all**: best model `N4`, AIC=237671.6. Coef on `log_exposure` = **0.4428** (sublinear: doubling traffic → 1.36× accidents). NB2 alpha = 0.4858393224261906.
  Residual Moran I = 0.1380 (p=0.0010). Spatial structure remains after accounting for exposure and year/month.
