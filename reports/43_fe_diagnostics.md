# Phase 10 · Iter 3 — FE diagnostics

## Variance decomposition of `log_exposure`

- Total: **0.9095**
- Between-hex: **0.6946** (**76.4%**)
- Within-hex: **0.2149** (23.6%)
- Within-hex-within-year: **0.8976** (98.7%)

**Interpretación:** 76% de la varianza de exposición está entre hexes. Cuando se añade hex FE, se absorbe ese 76%, y la covariable `log_exposure` queda solo con el 24% restante, que es pequeño, ruidoso y dominado por year FE (COVID).

## Response: `n_injured`

| spec | coef | SE | drift vs N4 | AIC |
| --- | ---: | ---: | ---: | ---: |
| N4_no_hex_fe | 0.4667 | 0.0072 |  | 167236 |
| FE_no_year | 0.2690 | 0.0149 | -42.4% | 145304 |
| FE_no_hex | 0.4667 | 0.0072 | +0.0% | 167236 |
| FE_full | 0.1658 | 0.0148 | -64.5% | 144763 |

## Response: `n_all`

| spec | coef | SE | drift vs N4 | AIC |
| --- | ---: | ---: | ---: | ---: |
| N4_no_hex_fe | 0.4397 | 0.0048 |  | 252421 |
| FE_no_year | 0.2665 | 0.0100 | -39.4% | 207473 |
| FE_no_hex | 0.4397 | 0.0048 | +0.0% | 252421 |
| FE_full | 0.1474 | 0.0098 | -66.5% | 206022 |

## Hex quality flags

- well_sampled (n>=60): 876
- moderate (24<=n<60): 51
- singleton (n<24): 2

## Conclusiones metodológicas

1. **El FE no falla, mide otra cosa.** Con 90% de la varianza de exposición entre hexes, añadir hex FE absorbe esa varianza. El coef 0.156 es 'el efecto within-hex de corto plazo, condicionado a year FE', no 'el efecto total'. El coef 0.474 (N4) es el efecto total (between+within).

2. **Rechazar CAR.** CAR reproduce el mismo problema (b_i absorbe media por hex). Alternativas si se quiere mapa de riesgo:
   - Two-stage: fijar exposure coef de N4, modelar residuals con CAR.
   - Bambi/PyMC con prior explícito sobre la identificación.
   - Ambos quedan fuera del alcance si Fase 11 aprieta.

3. **Dos hallazgos científicos válidos:**
   - Cross-sectional (entre hexes): accidentes escalan 1.39× por 2× tráfico (sublineal, N4).
   - Time-series (within hex): efectos pequeños (0.156) porque el tráfico dentro de un hex es casi estacionario excepto COVID.
   - Política: '¿qué pasa si aumenta el tráfico en Madrid 10%?' → 3.6% más accidentes (within). '¿Qué hexes son más peligrosos?' → el ranking lo dicta el between (N4).

