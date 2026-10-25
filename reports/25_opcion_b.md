# Source Audit Report

## Opción B — comparison (test=2025) sorted by PR-AUC(grave)

| key | kappa_q | f1_macro | mae_ord | pr_auc_grave | roc_auc_grave | rec_grave | rec_minor | rec_no_inj | fatal_Lift@2 | fatal_Lift@5 | fatal_Recall@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| catboost__adasyn | 0.5562 | 0.5296 | 0.2952 | 0.2434 | 0.7805 | 0.073 | 0.691 | 0.849 | 4.683 | 4.374 | 0.4375 |
| catboost__smote | 0.5565 | 0.5303 | 0.295 | 0.2427 | 0.7801 | 0.073 | 0.691 | 0.85 | 4.683 | 4.999 | 0.4688 |
| catboost__borderline | 0.5574 | 0.5302 | 0.2952 | 0.2419 | 0.7796 | 0.074 | 0.688 | 0.851 | 4.683 | 4.999 | 0.4375 |
| catboost__none | 0.5558 | 0.5273 | 0.2948 | 0.2418 | 0.7802 | 0.067 | 0.693 | 0.849 | 4.683 | 4.999 | 0.4688 |
| lgbm__smote | 0.5574 | 0.528 | 0.2955 | 0.2402 | 0.7738 | 0.071 | 0.69 | 0.848 | 6.243 | 4.374 | 0.5 |
| lgbm__none | 0.5566 | 0.5279 | 0.2957 | 0.2385 | 0.7726 | 0.07 | 0.69 | 0.849 | 4.683 | 4.999 | 0.4688 |
| lgbm__adasyn | 0.5532 | 0.5293 | 0.2972 | 0.2382 | 0.7744 | 0.073 | 0.692 | 0.846 | 4.683 | 4.374 | 0.4688 |
| lgbm__borderline | 0.5532 | 0.5261 | 0.2955 | 0.2374 | 0.7764 | 0.064 | 0.697 | 0.847 | 4.683 | 3.749 | 0.5 |

## Best by PR-AUC(grave)

catboost__adasyn

## Best params

{
  "iterations": 2165,
  "learning_rate": 0.018891200276189388,
  "depth": 5,
  "l2_leaf_reg": 0.04342298948286135,
  "random_strength": 0.016480446427978974,
  "bagging_temperature": 0.5247564316322378
}

## Bootstrap CI (best)

{
  "pr_auc_grave": {
    "median": 0.24398015696624287,
    "ci95_low": 0.2294794638217377,
    "ci95_high": 0.2584968454013019
  },
  "recall_grave": {
    "median": 0.07257941785297145,
    "ci95_low": 0.061893985713473584,
    "ci95_high": 0.0828547036792245
  },
  "precision_grave": {
    "median": 0.3060797101449275,
    "ci95_low": 0.2645856518608229,
    "ci95_high": 0.34253463533166506
  }
}

## Fatal ranking on best (P(grave) as score)

{
  "n_fatal": 32,
  "by_k": {
    "k=2%": {
      "k": 434,
      "tp": 3,
      "precision": 0.0069,
      "recall": 0.0938,
      "lift": 4.683
    },
    "k=5%": {
      "k": 1084,
      "tp": 7,
      "precision": 0.0065,
      "recall": 0.2188,
      "lift": 4.374
    },
    "k=10%": {
      "k": 2168,
      "tp": 14,
      "precision": 0.0065,
      "recall": 0.4375,
      "lift": 4.374
    }
  }
}

## Tuning time per combo (s)

{
  "lgbm__none": 633.6,
  "lgbm__smote": 1415.4,
  "lgbm__borderline": 2164.7,
  "lgbm__adasyn": 2779.2,
  "catboost__none": 1307.3,
  "catboost__smote": 2977.1,
  "catboost__borderline": 1789.7,
  "catboost__adasyn": 1764.1
}

## Reference: Opción A (4 clases)

Best LGBM: kappa 0.552, PR-AUC(fatal) 0.0053
Best CatBoost: kappa 0.551, PR-AUC(fatal) 0.0081
LR baseline: kappa 0.522, PR-AUC(fatal) 0.0083

## Notes

Target: 0=no_injury, 1=minor, 2=grave (serious ∪ fatal).
Preprocesado M4: SimpleImputer(median, indicator) + OHE. Fit solo en train.
Augmentation dentro de cada fold de LOGO.
fatal se evalúa como ranking usando P(grave) como score.
