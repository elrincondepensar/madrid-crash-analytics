# Phase 8 · Iter 1 — Local Moran's I (LISA)

_Weights: Queen/Rook/KNN cascade (islands dropped iteratively). 999 permutations (fallback 199). FDR α=0.05. Relative variance threshold 1e-10._

## `hex250m` — 3962 units

| variable | Moran I | n | HH% | LL% | HL% | LH% | NS% | W |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `counts_all` | 0.6769 | 3962 | 0.0427 | 0.0000 | 0.0000 | 0.0003 | 0.9571 | queen |
| `counts_injured` | 0.6840 | 3962 | 0.0737 | 0.3271 | 0.0000 | 0.0020 | 0.5972 | queen |
| `kde_all` | 0.9338 | 2174 | 0.1426 | 0.2548 | 0.0000 | 0.0000 | 0.6026 | queen |
| `injury_rate_eb` | 0.1261 | 3962 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | queen |

## `barrios` — 131 units

| variable | Moran I | n | HH% | LL% | HL% | LH% | NS% | W |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `counts_all` | 0.0925 | 131 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | queen |
| `counts_injured` | 0.1496 | 131 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | queen |
| `kde_all` | 0.7988 | 131 | 0.1679 | 0.1603 | 0.0000 | 0.0000 | 0.6718 | queen |
| `injury_rate_eb` | 0.3182 | 131 | 0.0305 | 0.0229 | 0.0076 | 0.0000 | 0.9389 | queen |

## `distritos` — 21 units

| variable | Moran I | n | HH% | LL% | HL% | LH% | NS% | W |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `counts_all` | 0.0136 | 21 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | queen |
| `counts_injured` | 0.1360 | 21 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | queen |
| `kde_all` | 0.3899 | 21 | 0.0476 | 0.0000 | 0.0000 | 0.0000 | 0.9524 | queen |
| `injury_rate_eb` | 0.1356 | 21 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | queen |

## `callejero` — 9285 units

| variable | Moran I | n | HH% | LL% | HL% | LH% | NS% | W |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `counts_all` | 0.0134 | 9285 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | knn_k6 |
| `counts_injured` | 0.0178 | 9285 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | knn_k6 |
| `kde_all` | 0.9593 | 7971 | 0.1630 | 0.2608 | 0.0000 | 0.0003 | 0.5760 | knn_k6 |
| `injury_rate_eb` | 0.0764 | 9285 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | knn_k6 |

## Sensitivity (hex 200/300 m, counts_all)

| hex size | n_units | Moran I | HH% |
| --- | ---: | ---: | ---: |
| hex200m | 6125 | 0.6244 | 0.0885 |
| hex300m | 2798 | 0.7112 | 0.0897 |

## Notes

- **Islands are dropped from the subset before Moran.**
- **Relative variance check**: `var / mean^2`, scale-invariant.
- **Moran on raw counts reflects volume, not risk (rule 9).**
- **Callejero** repaired via shapely.make_valid + buffer(0) in 17c.