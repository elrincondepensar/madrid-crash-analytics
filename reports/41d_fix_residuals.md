# Phase 10 · Iter 1d — Residuals fix

Reconstruye `mu = exp(X·β)` desde los coeficientes de N4 en 41c.

## `n_injured`

- mu: mean=0.709, sum=52926 (vs actual sum en la tabla by-year)
- **Moran on residuals: I=0.4408** (p=0.0010, n=922)
  - Reducción vs M1 (41b): -0.2249

| year | actual | predicted | ratio |
| --- | ---: | ---: | ---: |
| 2019 | 9018 | 9068 | 1.006 |
| 2020 | 5633 | 5601 | 0.994 |
| 2021 | 7054 | 7104 | 1.007 |
| 2022 | 7791 | 7809 | 1.002 |
| 2023 | 7736 | 7780 | 1.006 |
| 2024 | 7772 | 7807 | 1.005 |
| 2025 | 7686 | 7757 | 1.009 |

## `n_all`

- mu: mean=1.543, sum=115144 (vs actual sum en la tabla by-year)
- **Moran on residuals: I=0.4437** (p=0.0010, n=922)
  - Reducción vs M1 (41b): -0.2279

| year | actual | predicted | ratio |
| --- | ---: | ---: | ---: |
| 2019 | 18276 | 18427 | 1.008 |
| 2020 | 11545 | 11488 | 0.995 |
| 2021 | 14970 | 15091 | 1.008 |
| 2022 | 16730 | 16726 | 1.000 |
| 2023 | 17323 | 17430 | 1.006 |
| 2024 | 17634 | 17653 | 1.001 |
| 2025 | 18201 | 18328 | 1.007 |

