# Phase 10 · Iter 4 — Variance by year + FE bootstrap

## PART A — Variance decomposition by year

| year | between | within | mean_le | n_obs |
| --- | ---: | ---: | ---: | ---: |
| 2019 | 89.6% | 10.4% | 13.515 | 10317 |
| 2020 | 69.4% | 30.6% | 13.110 | 10333 |
| 2021 | 85.7% | 14.3% | 13.354 | 10528 |
| 2022 | 82.3% | 17.7% | 13.327 | 10479 |
| 2023 | 87.2% | 12.8% | 13.372 | 10866 |
| 2024 | 86.7% | 13.3% | 13.372 | 11073 |
| 2025 | 89.0% | 11.0% | 13.352 | 11029 |

**2020 within share: 30.6%. Mean of other years: 13.2%. Ratio: 2.31×.**

→ **2020 tiene una concentración de varianza within significativamente mayor que el resto de años.** Esto confirma que el within-hex share está dominado por el shock COVID. El coef de FE_full (0.166) identifica principalmente la respuesta al COVID, no fluctuaciones de tráfico de política pública.

## PART C — Block bootstrap of hex FE

- Iteraciones: 50/50
- Hexes bootstrapeados: 23

### Estabilidad del coef de `log_exposure`

- Mean = **0.1849**, sd = 0.0819, CI95% = [0.0475, 0.3037]
- **Coef. de variación (sd/mean) = 0.443** (FAIL con umbral 0.15)

### Estabilidad de `fe_log_rr` por hex

- **Pearson rho (original vs bootstrap mean): 0.3518** (FAIL con umbral 0.90)
- Spearman rho: 0.2530
- **Cobertura del CI95%: 87.0%** (ideal: ~95%)
- sd media por hex: 1.2211, sd p90: 1.4706

### Hexes más estables (sd bajo)

- hex 4022: orig=-0.155, boot_mean=0.226, sd=0.6158
- hex 1757: orig=0.868, boot_mean=0.832, sd=0.9036
- hex 4315: orig=-1.569, boot_mean=-0.497, sd=0.9360
- hex 2813: orig=0.903, boot_mean=-0.121, sd=0.9912
- hex 4236: orig=-0.879, boot_mean=0.055, sd=1.0457

### Hexes menos estables (sd alto)

- hex 1532: orig=-0.935, boot_mean=-0.602, sd=2.5028
- hex 3354: orig=0.079, boot_mean=-0.025, sd=1.5181
- hex 2667: orig=1.112, boot_mean=0.301, sd=1.5041
- hex 1673: orig=-0.190, boot_mean=-0.074, sd=1.3366
- hex 2151: orig=-0.312, boot_mean=-0.037, sd=1.3165

## Interpretación

**Inestabilidad detectada.** Revisar la especificación antes de usar el mapa FE.
