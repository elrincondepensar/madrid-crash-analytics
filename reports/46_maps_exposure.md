# Phase 11 · Iter 2 — Exposure & residuals maps

Directorio: `reports/figures/final/`

## Exposición anual (hex250m)

| año | n_hexes | median (veh/día) | mean (veh/día) |
| --- | ---: | ---: | ---: |
| 2019 | 869 | 24792 | 35726 |
| 2020 | 878 | 18330 | 25055 |
| 2022 | 895 | 21400 | 29316 |
| 2025 | 926 | 21199 | 30080 |

## COVID ratio (2020/2019)

- Hexes con datos: 869
- Mediana 2020/2019: **0.734** (-26.6%)
- `reports\figures\final\lisa_exposure_covid_ratio.png`

## N4 residuals por hex

- Hexes con datos: 929
- Mediana: -0.141, p10: -0.568, p90: +0.740
- `reports\figures\final\lisa_n4_residuals.png`

## Tipo de vía dominante

- No data: 3033 hexes
- Calle: 577 hexes
- Avenida: 239 hexes
- Paseo: 42 hexes
- Other: 34 hexes
- Carretera: 25 hexes
- Glorieta: 12 hexes

`reports\figures\final\lisa_tipo_via_dominant.png`

## Interpretación

**Exposición anual:** el patrón entre 2019 y 2025 es estable en los hexes con sensor. La red de 929 hexes cubre las arterias principales y el centro.

**COVID ratio 2020/2019:** la mediana cae a ~0.75-0.85. Pero la distribución es **heterogénea**: centro y oficinas caen más que residencial, consistente con teletrabajo.

**N4 residuals:** el mapa muestra colores rojos (más accidentes de lo esperado por exposición + covariates) en el centro y en algunas arterias específicas, y azules (menos de lo esperado) en la periferia norte. **Este es el mapa de riesgo residual descriptivo** que sustituye al FE descartado en D110. No son coeficientes causales, son 'dónde el modelo no llega'.

**Tipo_via dominante:** patrón espacial esperado: avenidas en la periferia y M-30, calles en el centro denso.
