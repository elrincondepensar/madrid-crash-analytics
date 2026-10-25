# Phase 11 · Iter 1 — LISA maps

**PNGs generados:** 14 mapas individuales + 3 composites.

Directorio: `reports/figures/final/`

## Mapas individuales

| unit | variable | I | HH | LL | file |
| --- | --- | ---: | ---: | ---: | --- |
| hex250m | counts_all | 0.677 | 169 | 0 | `reports\figures\final\lisa_hex250m_counts_all.png` |
| hex250m | counts_injured | 0.684 | 292 | 1296 | `reports\figures\final\lisa_hex250m_counts_injured.png` |
| hex250m | kde_all | 0.934 | 310 | 554 | `reports\figures\final\lisa_hex250m_kde_all.png` |
| hex250m | injury_rate_eb | 0.126 | 0 | 0 | `reports\figures\final\lisa_hex250m_injury_rate_eb.png` |
| barrios | counts_all | 0.092 | 0 | 0 | `reports\figures\final\lisa_barrios_counts_all.png` |
| barrios | counts_injured | 0.150 | 0 | 0 | `reports\figures\final\lisa_barrios_counts_injured.png` |
| barrios | kde_all | 0.799 | 22 | 21 | `reports\figures\final\lisa_barrios_kde_all.png` |
| barrios | injury_rate_eb | 0.318 | 4 | 3 | `reports\figures\final\lisa_barrios_injury_rate_eb.png` |
| distritos | counts_all | 0.014 | 0 | 0 | `reports\figures\final\lisa_distritos_counts_all.png` |
| distritos | counts_injured | 0.136 | 0 | 0 | `reports\figures\final\lisa_distritos_counts_injured.png` |
| distritos | kde_all | 0.390 | 1 | 0 | `reports\figures\final\lisa_distritos_kde_all.png` |
| distritos | injury_rate_eb | 0.136 | 0 | 0 | `reports\figures\final\lisa_distritos_injury_rate_eb.png` |
| callejero | counts_all | 0.013 | 0 | 0 | `reports\figures\final\lisa_callejero_counts_all.png` |
| callejero | kde_all | 0.959 | 1299 | 2079 | `reports\figures\final\lisa_callejero_kde_all.png` |

## Composites 2×2

- `reports\figures\final\lisa_hex250m_4panel.png`
- `reports\figures\final\lisa_barrios_4panel.png`
- `reports\figures\final\lisa_distritos_4panel.png`

## Interpretación

**hex250m:** `counts_all` con I=0.677 (señal concentrada en pocos hotspots operativos). `counts_injured` muestra HH + LL bimodal. `kde_all` es el más informativo (I=0.934).

**barrios:** `counts_*` mayoritariamente NS (I=0.09-0.15). `kde_all` retiene I=0.80 con HH=16.8%. Confirma D108: la agregación a barrio diluye la señal de recuentos.

**distritos:** casi todo NS salvo `kde_all` (HH=4.8%).

**callejero:** `counts_all` vacío (I=0.013). `kde_all` preserva estructura (I=0.959). MAUP del callejero.
