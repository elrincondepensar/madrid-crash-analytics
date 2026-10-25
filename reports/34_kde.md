# Phase 7 · Iter 1 — Network KDE baseline (Option C)

_Option C = 2D KDE on raw accidents, evaluated at edge midpoints. Gaussian kernel. Bandwidths in meters._

## Bandwidth selection table

| bw (m) | stability_all | stability_injured | contrast_mean |
| ---: | ---: | ---: | ---: |
| 100 | 0.964 | 0.946 | 0.972 |
| 150 | 0.977 | 0.966 | 0.981 |
| 200 | 0.983 | 0.976 | 0.986 |
| 260 | 0.988 | 0.983 | 0.990 |
| 350 | 0.992 | 0.989 | 0.994 |
| 500 | 0.995 | 0.994 | 0.997 |

**Chosen bandwidth: 500 m** (`best_score_fallback`). Strict passes: none.

## Stability detail (chosen bw)

**all**: train_vs_val=0.995 · train_vs_test=0.995 · val_vs_test=0.996 · mean=0.995
**injured**: train_vs_val=0.993 · train_vs_test=0.995 · val_vs_test=0.993 · mean=0.994

## Notes

- **Stability** = mean Spearman rho between KDE densities across the three period pairs (train↔val, train↔test, val↔test). Measures whether the spatial pattern is preserved over time.
- **Severity contrast** = Spearman rho between `density_all` and `density_injured` per period. Values near 1 mean the bw cannot distinguish severity composition; values in 0.6–0.9 mean the bw reveals structural differences; below 0.6 suggests injured too sparse at that scale.
- **Selection** prefers the smallest bw passing both thresholds (`stability ≥ 0.80`, `contrast ∈ [0.60, 0.90]`). Fallback: score `stability + 0.5·contrast_quality`. Larger bw trivially increases stability, so smallest-passing avoids choosing bw driven by smoothing rather than signal.
- **KDE is computed independently on each period.** No refit on train/val/test in the ML sense — each period is treated as an independent sample of the same underlying process. Comparisons are descriptive, not predictive.