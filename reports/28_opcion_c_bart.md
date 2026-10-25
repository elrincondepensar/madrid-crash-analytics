# Source Audit Report

## Comparación (test=2025, threshold tuned en val)

| model | cv_f1 | threshold | kappa | f1_injured | precision | recall | pr_auc | roc_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lr | 0.7909 | 0.415 | 0.5638 | 0.7643 | 0.7025 | 0.8381 | 0.8133 | 0.8708 |
| lgbm | 0.8045 | 0.405 | 0.5758 | 0.7704 | 0.7096 | 0.8425 | 0.8255 | 0.878 |
| catboost | 0.8045 | 0.415 | 0.5711 | 0.7663 | 0.7116 | 0.8301 | 0.8166 | 0.8728 |
| bart | N/A | 0.415 | 0.4938 | 0.7214 | 0.678 | 0.7707 | 0.7721 | 0.8303 |

## Lift@k — best model (lgbm)

| k | count | injury_rate | lift | recall |
| --- | --- | --- | --- | --- |
| k=5% | 1084 | 0.9197 | 2.177 | 0.1089 |
| k=10% | 2168 | 0.9091 | 2.152 | 0.2152 |
| k=20% | 4336 | 0.8775 | 2.077 | 0.4154 |
| k=30% | 6504 | 0.8359 | 1.978 | 0.5936 |
| k=50% | 10839 | 0.7107 | 1.682 | 0.841 |

## Bootstrap CI — best model

{
  "f1_injured": {
    "median": 0.7705627595657152,
    "ci95_low": 0.763899340550332,
    "ci95_high": 0.776213940856944
  },
  "pr_auc_injured": {
    "median": 0.8253102165168935,
    "ci95_low": 0.8175112708455718,
    "ci95_high": 0.8342770900468813
  },
  "roc_auc_injured": {
    "median": 0.8781215331627288,
    "ci95_low": 0.8737417399771776,
    "ci95_high": 0.882893231193765
  }
}

## Confusion matrix — best (test, threshold tuned)

| | no_injury | injured |
| --- | --- | --- |
| no_injury | 9361 | 3157 |
| injured | 1443 | 7716 |

## Confusion matrix — best (test, threshold 0.5)

| | no_injury | injured |
| --- | --- | --- |
| no_injury | 10143 | 2375 |
| injured | 1921 | 7238 |

## Best params por modelo

{
  "lr": {
    "C": 0.05416754583247458,
    "class_weight": "balanced",
    "penalty": "l2",
    "solver": "lbfgs"
  },
  "lgbm": {
    "colsample_bytree": 0.6733618039413735,
    "learning_rate": 0.024878734419814436,
    "max_depth": 9,
    "min_child_samples": 93,
    "n_estimators": 674,
    "num_leaves": 73,
    "reg_alpha": 0.039759772143180966,
    "reg_lambda": 0.0015369603110608836,
    "subsample": 0.9895022075365837,
    "subsample_freq": 1
  },
  "catboost": {
    "bagging_temperature": 0.15599452033620265,
    "depth": 6,
    "iterations": 1982,
    "l2_leaf_reg": 10.275417738969418,
    "learning_rate": 0.06054365855469246,
    "random_strength": 0.679657809075816
  },
  "bart": {
    "m": 50,
    "alpha": 0.95,
    "beta": 2.0
  }
}

## BART search log

[
  {
    "m": 50,
    "alpha": 0.95,
    "beta": 2.0,
    "f1_val": 0.7048,
    "seconds": 87.2
  }
]

## Notes

Búsqueda de hiperparámetros:
- LR/LGBM/CatBoost: RandomizedSearchCV + LeaveOneGroupOut(año) sobre train. Log trial-a-trial con ETA.
- BART: grid manual (m, alpha, beta) sobre subconjunto de train (10000 samples), refit en train completo con best.
Target: 0=no_injury, 1=any_injury (minor ∪ serious ∪ fatal).
Threshold tuned en val (max F1(injured)).
Preprocesado: imputer + OHE (árboles) / + scaler (LR).
TE fit-on-train sobre P(injured).
