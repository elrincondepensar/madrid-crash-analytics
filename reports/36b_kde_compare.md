# Phase 7 · Iter 3b — C vs B comparison (fixed)

_Fix of naming inconsistency in script 36: Option B columns now share names with Option C (`kde_*`). `network_kde_b.parquet` was renamed in place; no Option B recomputation._

## C vs B comparison

Compared over **36** (bw, period, severity) combinations.

| metric | mean | min | max |
| --- | ---: | ---: | ---: |
| Spearman rho (C, B) | 0.994 | 0.987 | 1.000 |
| Jaccard (top-20% dense) | 0.982 | 0.934 | 0.998 |

**Winner: `C`** — C and B agree strongly (mean rho=0.994, mean jaccard=0.982); C is simpler (raw accident points).

## Per-combination table

| col | rho | jaccard_top20 |
| --- | ---: | ---: |
| `kde_test_all_bw100` | 0.989 | 0.959 |
| `kde_test_all_bw150` | 0.989 | 0.983 |
| `kde_test_all_bw200` | 0.992 | 0.989 |
| `kde_test_all_bw260` | 0.995 | 0.992 |
| `kde_test_all_bw350` | 0.998 | 0.996 |
| `kde_test_all_bw500` | 0.999 | 0.998 |
| `kde_test_injured_bw100` | 0.988 | 0.950 |
| `kde_test_injured_bw150` | 0.987 | 0.975 |
| `kde_test_injured_bw200` | 0.990 | 0.982 |
| `kde_test_injured_bw260` | 0.994 | 0.988 |
| `kde_test_injured_bw350` | 0.997 | 0.995 |
| `kde_test_injured_bw500` | 0.999 | 0.996 |
| `kde_train_all_bw100` | 0.991 | 0.959 |
| `kde_train_all_bw150` | 0.989 | 0.982 |
| `kde_train_all_bw200` | 0.990 | 0.988 |
| `kde_train_all_bw260` | 0.993 | 0.991 |
| `kde_train_all_bw350` | 0.996 | 0.995 |
| `kde_train_all_bw500` | 0.999 | 0.998 |
| `kde_train_injured_bw100` | 0.995 | 0.955 |
| `kde_train_injured_bw150` | 0.995 | 0.974 |
| `kde_train_injured_bw200` | 0.995 | 0.987 |
| `kde_train_injured_bw260` | 0.997 | 0.990 |
| `kde_train_injured_bw350` | 0.999 | 0.995 |
| `kde_train_injured_bw500` | 1.000 | 0.998 |
| `kde_val_all_bw100` | 0.991 | 0.948 |
| `kde_val_all_bw150` | 0.988 | 0.973 |
| `kde_val_all_bw200` | 0.990 | 0.984 |
| `kde_val_all_bw260` | 0.993 | 0.990 |
| `kde_val_all_bw350` | 0.997 | 0.993 |
| `kde_val_all_bw500` | 0.999 | 0.996 |
| `kde_val_injured_bw100` | 0.990 | 0.934 |
| `kde_val_injured_bw150` | 0.988 | 0.964 |
| `kde_val_injured_bw200` | 0.990 | 0.980 |
| `kde_val_injured_bw260` | 0.992 | 0.987 |
| `kde_val_injured_bw350` | 0.996 | 0.994 |
| `kde_val_injured_bw500` | 0.999 | 0.995 |

## Decision rule (D-36-3)

- `mean_rho ≥ 0.95` **AND** `mean_jaccard_top20 ≥ 0.70` → **C wins** (simpler).
- Otherwise → **B wins** (network-aligned source).
- `network_kde_final.parquet` contains only `C` at bw=260 m (6 columns: 3 periods × 2 severities).

## Notes

- **Option B is 20 min of compute** (36 KDE jobs with sample_weight, ball_tree, 95k eval points each). Saved to disk in script 36; this script only renamed columns.
- **Jaccard** = fraction of top-20% dense edges the two options agree on. High = identical hotspot rankings at the top.
- **Both options are 2D KDE with Gaussian kernel**, differing only in the source representation: C uses 137k raw accident points, B uses ~10-20k edge midpoints weighted by their accident count. Total mass is comparable; B loses within-edge dispersion.