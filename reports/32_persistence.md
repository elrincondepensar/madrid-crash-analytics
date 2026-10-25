# Phase 6 · Iter 2 (v3) — Temporal persistence of HDBSCAN clusters

_Assignment rule: nearest train centroid if `dist < 1.0 × min(p90_radius, 500 m)`. Persistence threshold: `>= 10` accidents from the future split._

_No refit on val/test. Coordinates UTM 30N (EPSG:25830), distances in meters._

_Iter-2 v1 used `1.5 × max_radius`; that rule was tautologically satisfied. v3 replaces `max` by `p90` with a hard cap to remove outlier-driven envelopes, and adds two rule-independent stability metrics (Spearman ρ and top-k share ratio)._

## Variant `all` — mcs=50, ms=10

- **Train**: 74,253 accidents → **375 clusters**, noise_share 0.414 (train coverage in-cluster 0.586), fit 1.9s
- **radius_eff_m**: median 500 m … (per-cluster, capped at 500 m)
- **Val (2023-24)**: n=20,666 → coverage **0.560** (generosity gap -0.026), persisted **350** (0.933)
  - dist to assigned centroid: median 118.5 m, p90 320.8 m
  - **Spearman ρ** (all clusters, n=375): 0.797 [p=1.04e-83] | (top-100 by train): 0.575
  - **Top-10**: share_train=0.093, share_future=0.063, ratio=0.676, overlap=3/10 (0.30)
  - **Top-20**: share_train=0.161, share_future=0.131, ratio=0.813, overlap=12/20 (0.60)
  - **Top-30**: share_train=0.218, share_future=0.185, ratio=0.849, overlap=20/30 (0.67)
- **Test (2025)**: n=42,376 → coverage **0.558** (generosity gap -0.028), persisted **373** (0.995)
  - dist to assigned centroid: median 121.2 m, p90 315.2 m
  - **Spearman ρ** (all clusters, n=375): 0.821 [p=1.24e-92] | (top-100 by train): 0.595
  - **Top-10**: share_train=0.093, share_future=0.063, ratio=0.678, overlap=4/10 (0.40)
  - **Top-20**: share_train=0.161, share_future=0.135, ratio=0.840, overlap=14/20 (0.70)
  - **Top-30**: share_train=0.218, share_future=0.187, ratio=0.859, overlap=20/30 (0.67)
- **Persisted in BOTH** val and test: **350** (0.933)

### Acceptance criteria

| criterion | value | bounds | status |
| --- | ---: | :---: | :---: |
| `share_persisted_val` | 0.933 | >= 0.30 | PASS |
| `share_persisted_test` | 0.995 | >= 0.30 | PASS |
| `coverage_new_val` | 0.560 | [0.40, 0.80] | PASS |
| `coverage_new_test` | 0.558 | [0.40, 0.80] | PASS |
| `share_persistent_in_both` | 0.933 | >= 0.30 | PASS |
| `spearman_top100_val` | 0.575 | >= 0.60 | FAIL |
| `spearman_top100_test` | 0.595 | >= 0.60 | FAIL |

**Overall**: SOME FAIL

---

## Variant `injured` — mcs=30, ms=10

- **Train**: 35,108 accidents → **307 clusters**, noise_share 0.449 (train coverage in-cluster 0.551), fit 0.7s
- **radius_eff_m**: median 500 m … (per-cluster, capped at 500 m)
- **Val (2023-24)**: n=9,136 → coverage **0.507** (generosity gap -0.044), persisted **201** (0.655)
  - dist to assigned centroid: median 118.1 m, p90 325.7 m
  - **Spearman ρ** (all clusters, n=305): 0.726 [p=4.01e-51] | (top-100 by train): 0.511
  - **Top-10**: share_train=0.087, share_future=0.081, ratio=0.927, overlap=5/10 (0.50)
  - **Top-20**: share_train=0.155, share_future=0.154, ratio=0.994, overlap=14/20 (0.70)
  - **Top-30**: share_train=0.215, share_future=0.218, ratio=1.012, overlap=19/30 (0.63)
- **Test (2025)**: n=18,216 → coverage **0.502** (generosity gap -0.049), persisted **291** (0.948)
  - dist to assigned centroid: median 121.1 m, p90 323.9 m
  - **Spearman ρ** (all clusters, n=307): 0.771 [p=8.67e-62] | (top-100 by train): 0.582
  - **Top-10**: share_train=0.087, share_future=0.080, ratio=0.923, overlap=4/10 (0.40)
  - **Top-20**: share_train=0.155, share_future=0.157, ratio=1.013, overlap=15/20 (0.75)
  - **Top-30**: share_train=0.215, share_future=0.224, ratio=1.037, overlap=21/30 (0.70)
- **Persisted in BOTH** val and test: **200** (0.651)

### Acceptance criteria

| criterion | value | bounds | status |
| --- | ---: | :---: | :---: |
| `share_persisted_val` | 0.655 | >= 0.30 | PASS |
| `share_persisted_test` | 0.948 | >= 0.30 | PASS |
| `coverage_new_val` | 0.507 | [0.40, 0.80] | PASS |
| `coverage_new_test` | 0.502 | [0.40, 0.80] | PASS |
| `share_persistent_in_both` | 0.651 | >= 0.30 | PASS |
| `spearman_top100_val` | 0.511 | >= 0.60 | FAIL |
| `spearman_top100_test` | 0.582 | >= 0.60 | FAIL |

**Overall**: SOME FAIL

---

## Notes

- Cluster radius effective = **min(p90_radius, 500 m)**. Purpose: remove outlier-driven envelopes that stretch elongated clusters to multi-km radii.
- `generosity_gap = coverage_new - train_coverage`. Values close to 0 indicate the assignment rule is not more generous than HDBSCAN itself. Iter-2 v1 (max_radius × 1.5) gave +0.25 pp — tautological.
- **Spearman ρ** between `n_train` and `n_future` (cluster sizes) is invariant to the assignment threshold. Reported for all clusters present in both splits and for the top-100 by train.
- **Top-k share ratio** = `share_future(top-k-by-train) / share_train(top-k-by-train)`. Values near 1.0 indicate the k largest train clusters retain their proportional future volume. `overlap` counts how many of the top-k by train remain top-k by future.
- For the `injured` variant, val/test coverage is measured over **injured-only** future accidents, matching the fit population.
