# Script 33 — Noise-ceiling validation for HDBSCAN stability

**Question.** Is the Spearman rho from script 32 (0.51-0.60 for top-100, 0.73-0.82 for all clusters) at the methodological ceiling imposed by sample size and Poisson noise, or does it reveal real temporal drift?

**Method.** The same fit-assign pipeline used in script 32 (`fit` HDBSCAN, `radius_eff = min(p90, 500 m)`, assign points within `radius_eff`, Spearman rho on matched cluster sizes), applied to random 50/50 and 75/25 splits of train and to a 50/50 split of test. If rho(internal split) ~ rho(train -> test), the temporal dimension adds nothing beyond sampling.

**Decision rule.** if `rho(internal 50/50) ~= rho(train -> test)`: no temporal drift beyond sampling (ceiling reached). If `rho(internal) >> rho(train -> test)`: real temporal drift.

## Variant `all` (mcs=50, ms=10)

Train: 74,253  ·  Val: 20,666  ·  Test: 42,376

| comparison | rho | n_matched | n_clusters_fit | coverage_assign |
| --- | ---: | ---: | ---: | ---: |
| `sanity_train_to_train` | 0.942 | 375 | 375 | 0.580 |
| `train_full_to_test` | 0.821 | 375 | 375 | 0.558 |
| `train_full_to_val` | 0.797 | 375 | 375 | 0.560 |
| `train_50_A_to_B` | 0.853 | 178 | 178 | 0.511 |
| `train_50_B_to_A` | 0.814 | 166 | 166 | 0.495 |
| `train_75_to_25` | 0.831 | 274 | 274 | 0.529 |
| `test_50_tA_to_tB` | 0.606 | 91 | 91 | 0.419 |

**Key comparison (vs test).** rho(internal 50/50) = **0.853**  ·  rho(train -> test) = **0.821**  ·  ratio = **0.962**

**Key comparison (vs val).** rho(internal 75/25) = **0.831**  ·  rho(train -> val) = **0.797**  ·  ratio = **0.960**

→ **Ratio ≈ 1.** The temporal split does *not* introduce measurable stability loss beyond sampling noise. The rho from script 32 is essentially what you would get from any random 50/50 resample of the same data. **Ceiling reached.**

_Reference (script 32, all-clusters rho): train→val 0.797, train→test 0.821._

---

## Variant `injured` (mcs=30, ms=10)

Train: 35,108  ·  Val: 9,136  ·  Test: 18,216

| comparison | rho | n_matched | n_clusters_fit | coverage_assign |
| --- | ---: | ---: | ---: | ---: |
| `sanity_train_to_train` | 0.918 | 307 | 307 | 0.538 |
| `train_full_to_test` | 0.771 | 307 | 307 | 0.502 |
| `train_full_to_val` | 0.726 | 305 | 307 | 0.507 |
| `train_50_A_to_B` | 0.774 | 145 | 145 | 0.460 |
| `train_50_B_to_A` | 0.783 | 137 | 137 | 0.471 |
| `train_75_to_25` | 0.812 | 212 | 212 | 0.473 |
| `test_50_tA_to_tB` | n/a | 3 | 3 | 0.012 |

**Key comparison (vs test).** rho(internal 50/50) = **0.774**  ·  rho(train -> test) = **0.771**  ·  ratio = **0.997**

**Key comparison (vs val).** rho(internal 75/25) = **0.812**  ·  rho(train -> val) = **0.726**  ·  ratio = **0.893**

→ **Ratio ≈ 1.** The temporal split does *not* introduce measurable stability loss beyond sampling noise. The rho from script 32 is essentially what you would get from any random 50/50 resample of the same data. **Ceiling reached.**

_Reference (script 32, all-clusters rho): train→val 0.726, train→test 0.771._

---

## Interpretation notes

- All comparisons use the **same fit-assign pipeline as script 32**: fit HDBSCAN on one side, compute centroids and `radius_eff = min(p90, 500m)`, assign the other side to the nearest centroid if within `radius_eff`, Spearman rho on clusters with at least one assigned point.
- **`sanity_train_to_train`**: fit and assign on the same data. rho should be ~1.0. Any deviation indicates assignment ambiguity from neighbouring clusters.
- **`train_50_A_to_B` / `train_50_B_to_A`**: two random halves of train. Same data distribution, different random sample, no temporal separation.
- **`train_75_to_25`**: 75% of train for fit, 25% for assign. Ratio close to train/val (74/21), so it is the closest internal analogue of the temporal comparison against val.
- **`train_full_to_test`**: the exact comparison of script 32, reproduced to verify that the pipeline is identical.
- The **key number** is `rho(train -> test) / rho(internal 50/50)`. Values around 1.0 mean sampling noise fully explains the observed correlation; values substantially below 1.0 mean temporal drift reduces stability beyond what random resampling of the same period would produce.
