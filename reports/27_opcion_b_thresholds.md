# Source Audit Report

## Opción B — argmax vs thresholded (test=2025)

| model | kappa_argmax | kappa_thr | rec_grave_argmax | rec_grave_thr | rec_minor_thr | rec_noinj_thr | pr_auc_grave | fatal_Lift@2 | fatal_Lift@5 | fatal_Recall@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lr | 0.5315 | 0.5192 | 0.043 | 0.277 | 0.637 | 0.744 | 0.2382 | 4.683 | 4.374 | 0.4375 |
| catboost__adasyn | 0.5575 | 0.5396 | 0.074 | 0.32 | 0.63 | 0.75 | 0.2413 | 4.683 | 3.749 | 0.4688 |
| catboost__smote | 0.5572 | 0.5387 | 0.082 | 0.31 | 0.65 | 0.741 | 0.2424 | 6.243 | 3.125 | 0.4375 |
| lgbm__smote | 0.5567 | 0.533 | 0.075 | 0.238 | 0.71 | 0.726 | 0.2384 | 3.122 | 2.5 | 0.5 |

## Thresholds seleccionados en val

{
  "lr": {
    "0": 0.0,
    "1": 0.335,
    "2": 0.143
  },
  "catboost__adasyn": {
    "0": 0.0,
    "1": 0.325,
    "2": 0.153
  },
  "catboost__smote": {
    "0": 0.0,
    "1": 0.315,
    "2": 0.153
  },
  "lgbm__smote": {
    "0": 0.0,
    "1": 0.305,
    "2": 0.177
  }
}

## Bootstrap recall_grave (test, thresholded)

{
  "lr": {
    "median": 0.276672006011707,
    "ci95_low": 0.2576325224230545,
    "ci95_high": 0.29618622139583406
  },
  "catboost__adasyn": {
    "median": 0.3190315165168617,
    "ci95_low": 0.29947688061471456,
    "ci95_high": 0.3426867984789127
  },
  "catboost__smote": {
    "median": 0.3096085060700067,
    "ci95_low": 0.2911814930890389,
    "ci95_high": 0.3312621253411789
  },
  "lgbm__smote": {
    "median": 0.23808410174172567,
    "ci95_low": 0.21936101858319718,
    "ci95_high": 0.2566995813248158
  }
}

## Confusion matrices — best model (test, thresholded)

| true | no_injury | minor | grave |
| --- | --- | --- | --- |
| no_injury | 9385 | 2631 | 502 |
| minor | 1180 | 4444 | 1426 |
| grave | 200 | 1235 | 674 |

## Notes

Target: 0=no_injury, 1=minor, 2=grave (serious ∪ fatal).
Preprocesado con StandardScaler (necesario para LR y estable para árboles).
Thresholds por clase ajustados en val para maximizar F1 OvR.
Regla de decisión: si max(p_k/thr_k) > 1 → argmax(p_k/thr_k), si no → no_injury.
fatal evaluado como ranking usando P(grave).
