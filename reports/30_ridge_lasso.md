# Source Audit Report

## TARGET A — Injured binario

| config | n_components | f1_injured | precision | recall | pr_auc | roc_auc | kappa |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ridge_binary_no_pca | — | 0.7664 | 0.718 | 0.8218 | 0.8127 | 0.8708 | 0.5744 |
| ridge_binary_pca | 35 | 0.7373 | 0.694 | 0.7863 | 0.7792 | 0.8451 | 0.5234 |
| lasso_binary_no_pca | — | 0.7666 | 0.7068 | 0.8374 | 0.8164 | 0.8736 | 0.5691 |
| lasso_binary_pca | 35 | 0.7379 | 0.7058 | 0.7731 | 0.7813 | 0.8449 | 0.5305 |

## Referencia TARGET A (D094, script 28)

LGBM: F1=0.7718, PR-AUC=0.8297, Recall=0.8523, Kappa=0.5754
LR:   F1=0.7647, PR-AUC=0.8142, Recall=0.8423, Kappa=0.5629

## TARGET B — 4-clases

| config | n_components | kappa_q | f1_macro | mae_ord | pr_auc_fatal | pr_auc_grave | lift@2 | recall@10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ridge_4class_no_pca | — | 0.4787 | 0.3784 | 0.3467 | 0.00345 | 0.2266 | 3.122 | 0.3125 |
| ridge_4class_pca | 35 | 0.4317 | 0.3583 | 0.3689 | 0.00823 | 0.2156 | 9.365 | 0.4375 |
| lasso_4class_no_pca | — | 0.2999 | 0.3333 | 0.5523 | 0.00478 | 0.2246 | 4.683 | 0.3438 |
| lasso_4class_pca | 35 | 0.3274 | 0.3321 | 0.5396 | 0.00574 | 0.2222 | 6.243 | 0.3125 |

## Referencia TARGET B (D098, script 29)

flat LGBM: kappa=0.554, PR-AUC(fatal)=0.0064, Lift@2=4.68
flat LR:   kappa=0.417, PR-AUC(fatal)=0.0053, Lift@2=4.68

## Bootstrap CI — injured

{
  "ridge_binary_no_pca": {
    "f1_injured": {
      "median": 0.7666726203348297,
      "ci95_low": 0.7596540352246349,
      "ci95_high": 0.7722188736685809
    },
    "pr_auc_injured": {
      "median": 0.812861986034516,
      "ci95_low": 0.8046651953542072,
      "ci95_high": 0.822434399405249
    },
    "roc_auc_injured": {
      "median": 0.8709256280895358,
      "ci95_low": 0.8665430560313722,
      "ci95_high": 0.8758062393199041
    }
  },
  "ridge_binary_pca": {
    "f1_injured": {
      "median": 0.7373866636584856,
      "ci95_low": 0.7295980838738262,
      "ci95_high": 0.7436508802009621
    },
    "pr_auc_injured": {
      "median": 0.7794822130187813,
      "ci95_low": 0.7704196543119186,
      "ci95_high": 0.7889317250975267
    },
    "roc_auc_injured": {
      "median": 0.8452140491291475,
      "ci95_low": 0.8401774717926005,
      "ci95_high": 0.8501876823456026
    }
  },
  "lasso_binary_no_pca": {
    "f1_injured": {
      "median": 0.7668977191094086,
      "ci95_low": 0.7603718766890007,
      "ci95_high": 0.7729206362376335
    },
    "pr_auc_injured": {
      "median": 0.8165788134100445,
      "ci95_low": 0.8084798757649397,
      "ci95_high": 0.8258477030740352
    },
    "roc_auc_injured": {
      "median": 0.8736703334948093,
      "ci95_low": 0.8692783291233759,
      "ci95_high": 0.8785343554198949
    }
  },
  "lasso_binary_pca": {
    "f1_injured": {
      "median": 0.7379528302158118,
      "ci95_low": 0.7302294654724738,
      "ci95_high": 0.7445006008080814
    },
    "pr_auc_injured": {
      "median": 0.781739066923873,
      "ci95_low": 0.7724756083879136,
      "ci95_high": 0.7910548390944069
    },
    "roc_auc_injured": {
      "median": 0.8449995532918773,
      "ci95_low": 0.8399616725536488,
      "ci95_high": 0.8499310356066545
    }
  }
}

## Bootstrap CI — fatal

{
  "ridge_4class_no_pca": {
    "median": 0.003777417954155147,
    "ci95_low": 0.0019819082519903332,
    "ci95_high": 0.007065254474280736
  },
  "ridge_4class_pca": {
    "median": 0.009280926647868143,
    "ci95_low": 0.003550456006510098,
    "ci95_high": 0.02579252156611271
  },
  "lasso_4class_no_pca": {
    "median": 0.005115862621942328,
    "ci95_low": 0.0027917781944107765,
    "ci95_high": 0.008834018757915861
  },
  "lasso_4class_pca": {
    "median": 0.006285041128482943,
    "ci95_low": 0.0034610363584236157,
    "ci95_high": 0.01147949297638026
  }
}

## Notes

RidgeClassifier: Ridge regression interna (OVR para multiclase). Sin predict_proba; se usa decision_function + sigmoid/softmax.
Lasso: LogisticRegression con penalty='l1', solver='saga'.
PCA: n_components=0.95 (retiene 95% varianza), fit en train.
Preprocesado: imputer + StandardScaler + OHE, fit en train.
Threshold tuning en val, evaluado en test=2025.
