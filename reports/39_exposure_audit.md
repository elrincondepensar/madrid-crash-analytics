# Phase 9 · Iter 1 — Exposure sources audit
**Question:** What traffic exposure can we build, at what spatial/temporal granularity, from the available sources?
## Sources audited
| source | dataset | n_resources |
| --- | --- | ---: |
| `300233` | Aforo tráfico permanentes | 180 |
| `208627` | Transporte ptomedida histórico | 162 |
| `202468` | Ubicación puntos de medida | n/a |
| `203962` | Tráfico IMD anual | n/a |

## Decision matrix
| source | status | recommendation |
| --- | --- | --- |
| `300233` | **USABLE** | Permanent count stations. 60 physical stations, monthly data 2018-2025. Limited spatial coverage (arterial roads only). Suitable as Poisson offset for hex250m cells containing stations. |
| `208627` | **CITY_WIDE** | CITY-WIDE COVERAGE: 2 distinct tipo_elem values (['C30', 'M30']...). 15-min granularity, city-wide. GOLD MINE. Primary temporal exposure. |
| `202468` | **ERROR** |  |
| `203962` | **STATIC_ONLY** | Static 2012-2013 snapshot. Usable ONLY as time-invariant covariate in Phase 10, not as temporal offset. |

## 208627 (critical) — tipo_elem evidence
- `208627-0-transporte-ptomedida-historico-zip`: 2 unique tipo_elem — top values: [('C30', 4922), ('M30', 78)]
- `208627-161-transporte-ptomedida-historico`: 1 unique tipo_elem — top values: [('C30', 5000)]

## Overall recommendation
PRIMARY: 208627 (city-wide, 15-min). SECONDARY: 300233 (60 stations, 2018-2025). STATIC CONTROL: 203962 IMD.
