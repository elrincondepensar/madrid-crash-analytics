# Source Audit Report

## Comparación (test=2025, threshold tuned en val)

| model | cv_f1 | threshold | kappa | f1_injured | precision | recall | pr_auc | roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lr | 0.7913 | 0.41 | 0.563 | 0.7647 | 0.7001 | 0.8423 | 0.8142 | 0.871 |
| lgbm | 0.805 | 0.395 | 0.5754 | 0.7718 | 0.7051 | 0.8523 | 0.8297 | 0.8809 |
| catboost | 0.8046 | 0.405 | 0.5735 | 0.7687 | 0.7096 | 0.8386 | 0.8214 | 0.8757 |

## Lift@k — best model (lgbm)

| k | count | injury_rate | lift | recall |
| --- | --- | --- | --- | --- |
| k=5% | 1084 | 0.9216 | 2.181 | 0.1091 |
| k=10% | 2168 | 0.9105 | 2.155 | 0.2155 |
| k=20% | 4336 | 0.8808 | 2.085 | 0.417 |
| k=30% | 6504 | 0.8426 | 1.994 | 0.5983 |
| k=50% | 10839 | 0.7132 | 1.688 | 0.844 |

## Bootstrap CI — best model

{
  "f1_injured": {
    "median": 0.7719786142463938,
    "ci95_low": 0.7652117468299691,
    "ci95_high": 0.7774532135370396
  },
  "pr_auc_injured": {
    "median": 0.8297288284891502,
    "ci95_low": 0.8218037377344611,
    "ci95_high": 0.838431689051675
  },
  "roc_auc_injured": {
    "median": 0.8810834450966206,
    "ci95_low": 0.8768759994297004,
    "ci95_high": 0.8857291504774667
  }
}

## Confusion matrix — best (test, threshold tuned)

| | no_injury | injured |
| --- | --- | --- |
| no_injury | 9254 | 3264 |
| injured | 1353 | 7806 |

## Confusion matrix — best (test, threshold 0.5)

| | no_injury | injured |
| --- | --- | --- |
| no_injury | 10147 | 2371 |
| injured | 1875 | 7284 |

## Best params por modelo

{
  "lr": {
    "C": 21.368329072358772,
    "class_weight": "balanced",
    "penalty": "l2",
    "solver": "lbfgs"
  },
  "lgbm": {
    "colsample_bytree": 0.9442922333025374,
    "learning_rate": 0.010210451114512976,
    "max_depth": 11,
    "min_child_samples": 31,
    "n_estimators": 848,
    "num_leaves": 76,
    "reg_alpha": 0.007734482156646798,
    "reg_lambda": 0.0030162092627967776,
    "subsample": 0.7350460685614512,
    "subsample_freq": 1
  },
  "catboost": {
    "bagging_temperature": 0.6183860093330873,
    "depth": 7,
    "iterations": 2090,
    "l2_leaf_reg": 26.230889101079164,
    "learning_rate": 0.040482984757722995,
    "random_strength": 2.752717392942941
  }
}

## BART search log

[]

## Notes

Búsqueda de hiperparámetros:
- LR/LGBM/CatBoost: RandomizedSearchCV + LeaveOneGroupOut(año) sobre train. Log trial-a-trial con ETA.
- BART: grid manual (m, alpha, beta) sobre subconjunto de train (10000 samples), refit en train completo con best.
Target: 0=no_injury, 1=any_injury (minor ∪ serious ∪ fatal).
Threshold tuned en val (max F1(injured)).
Preprocesado: imputer + OHE (árboles) / + scaler (LR).
TE fit-on-train sobre P(injured).
