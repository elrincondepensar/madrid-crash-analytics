# Phase 9 · Iter 2b — Exposure build
**Months processed:** 84/84

## Status per year
| year | months | ids (median) | elapsed (s) |
| --- | ---: | ---: | ---: |
| 2019 | 12 | 3990 | 524 |
| 2020 | 12 | 4066 | 439 |
| 2021 | 12 | 4275 | 406 |
| 2022 | 12 | 4256 | 418 |
| 2023 | 12 | 4554 | 468 |
| 2024 | 12 | 4679 | 471 |
| 2025 | 12 | 4722 | 519 |

## Output
- `exposure_hex250m_monthly.parquet`: 74,717 rows
- Columns: ['hex_id', 'year', 'month', 'n_sensors', 'intensity_day_mean_mean', 'intensity_day_median', 'intensity_day_p90', 'intensity_day_sum', 'coverage_mean', 'n_urb', 'n_m30_c30']

## Sanity check
- hexes with any data: 929
- median sensors per (hex, month): 4.0
- median intensity_day_mean: 65679 veh/day
- p90 intensity_day_mean: 201353 veh/day
- max intensity_day_mean: 2445750 veh/day

### Mean intensity_day by year
| year | median_of_hexes | p90_of_hexes |
| --- | ---: | ---: |
| 2019 | 77734 | 242130 |
| 2020 | 55347 | 157127 |
| 2021 | 66546 | 213507 |
| 2022 | 65002 | 177369 |
| 2023 | 66225 | 208374 |
| 2024 | 65702 | 209207 |
| 2025 | 65354 | 198255 |
