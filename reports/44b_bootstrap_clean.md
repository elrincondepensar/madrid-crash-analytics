# Phase 10 · Iter 4b — Bootstrap limpio

**Diseño:** 20 iteraciones, 400 hexes well-sampled por iteración, filtro mínimo 5 apariciones.

- Bootstrap exitosos: 20/20
- Hexes bootstrapeados: 640 (vs 23 en iter 4)
- Hexes con overlap al original: 640

## Coeficiente de `log_exposure`

- Mean = **0.1716**, sd = 0.0358
- CI95% = [0.1101, 0.2329]
- CV (sd/mean) = **0.209** (FAIL con umbral 0.15)

## Estabilidad de `fe_log_rr` por hex

- **Pearson rho (orig vs bootstrap): 0.0049** (FAIL con umbral 0.90)
- Spearman rho: 0.0306
- Cobertura CI95%: 81.4%
- sd media por hex: 1.1141
- sd mediana: 1.0923
- sd p90: 1.5861

### Hexes más estables

- hex 1611: orig=0.645, boot_mean=0.448, sd=0.3265
- hex 3566: orig=0.086, boot_mean=0.293, sd=0.3746
- hex 4544: orig=-0.346, boot_mean=0.183, sd=0.3913
- hex 1088: orig=-0.701, boot_mean=-0.132, sd=0.4255
- hex 1388: orig=0.073, boot_mean=0.932, sd=0.4296

### Hexes menos estables

- hex 4085: orig=0.155, boot_mean=-0.162, sd=2.9755
- hex 2899: orig=0.948, boot_mean=-0.324, sd=2.8964
- hex 1697: orig=0.607, boot_mean=-0.473, sd=2.4719
- hex 938: orig=-0.188, boot_mean=-0.719, sd=2.4488
- hex 3558: orig=-0.027, boot_mean=-0.807, sd=2.4152

## Interpretación

**Inestabilidad real confirmada.** Incluso con 400 hexes por iteración, el FE no converge a un ranking estable. Diagnóstico: colinealidad tipo_via↔hex y year↔exposure hacen el problema mal identificado. **Descartar mapa FE.**
