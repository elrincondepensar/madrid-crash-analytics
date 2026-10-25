# Phase 10 · Iter 4c — Centering diagnostic

## Diagnóstico del baseline-shift

- SD de `fe_log_rr` ORIGINAL (entre hexes, fit único): **1.3627**
- SD de `boot_mean` (entre hexes, promediado sobre 20 iter): **0.4283**
- SD PER-HEX (mismo hex a través de iteraciones): mean=1.1141, median=1.0923
- Ratio `var(per_hex) / var(orig)`: **0.67×** (>1 = baseline shift dominates)

**Confirmado:** la varianza per-hex es mucho mayor que la varianza entre hexes del fit original. Esto solo puede pasar si el baseline cambia entre iteraciones.

## Correlaciones

- Pearson sin centrar: **0.0049** (bug)
- Pearson centrado: **0.0049** (p=0.9017)
- Spearman centrado: **0.0306**

## Veredicto

**NO SIGNAL: even after centering, rho < 0.30. The FE coefficients are genuinely unstable. Discard the FE risk map. Use N4 residuals as descriptive alternative.**

## Recomendación de uso

Descartar el mapa FE. En Fase 11 usar residuos cuantilizados de N4 como mapa descriptivo alternativo.
