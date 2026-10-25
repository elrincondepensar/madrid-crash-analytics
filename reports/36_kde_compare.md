# Phase 7 · Iter 3 — Option B & C vs B comparison

_Option B = 2D KDE fitted on edge midpoints weighted by accident counts per edge. Same evaluation grid (edge midpoints), same Gaussian kernel, same bandwidths as Option C._

## Distance fix (sjoin_nearest)

- Old (`dist_to_edge_midpoint_m`): median **18.95 m**, p90 51.61 m, max 1760 m
- New (`dist_to_line_m`): median **2.19 m**, p90 6.49 m, max 1748 m

_Persisted to `accidents_with_edges.parquet`. The old column is kept for audit._

## C vs B comparison

Compared over **0** (bw, period, severity) combinations. Per-edge Spearman rho and Jaccard overlap of the top-20% dense edges.

| metric | mean | min |
| --- | ---: | ---: |
| Spearman rho (C, B) | n/a | n/a |
| Jaccard (top-20% dense) | n/a | n/a |

**Winner: `unknown`** — missing metrics

## Per-combination table

| col | rho | jaccard_top20 |
| --- | ---: | ---: |

## Decision rule (D-36-3)

- `mean_rho ≥ 0.95` **AND** `mean_jaccard_top20 ≥ 0.70` → **C wins** (simpler, no information loss).
- Otherwise → **B wins** (network-aligned source).
- `network_kde_final.parquet` contains only the winning option at bw = 260 m (3 periods × 2 severities = 6 columns).

## Notes

- **Option B source is different, not just weighted.** C uses 137k raw accident points. B uses ~10-20k edge midpoints (the edges with at least one accident in the period), each weighted by its accident count. Total mass is comparable; within-edge dispersion is lost (each edge contributes a single point). This is the price of the approximation.
- **Jaccard** measures the fraction of top-20% dense edges that the two options agree on. High Jaccard = identical hotspot rankings at the top. Low Jaccard = the top hotspots differ.
- **If B wins**, the reason is *not* that C is wrong — C is the standard 2D KDE. B is preferred because the sources align with the network structure that the map represents, which is the conceptual target of Phase 7.