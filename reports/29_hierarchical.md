# Source Audit Report

## Comparación (test=2025)

| model | mode | kappa_q | f1_macro | mae_ord | rec_minor | rec_serious | rec_fatal |
| --- | --- | --- | --- | --- | --- | --- | --- |
| flat_4class_lgbm | argmax | 0.5536 | 0.3979 | 0.2972 | 0.7 | 0.074 | 0.0 |
| flat_4class_lgbm | thresholded | 0.5348 | 0.4183 | 0.368 | 0.643 | 0.329 | 0.0 |
| hierarchical_lgbm | argmax | 0.5556 | 0.3965 | 0.2961 | 0.699 | 0.069 | 0.0 |
| hierarchical_lgbm | thresholded | 0.533 | 0.4163 | 0.3646 | 0.649 | 0.302 | 0.0 |
| flat_4class_lr | argmax | 0.4174 | 0.383 | 0.531 | 0.416 | 0.296 | 0.344 |
| flat_4class_lr | thresholded | 0.3741 | 0.3524 | 0.5033 | 0.671 | 0.29 | 0.0 |
| hierarchical_lr | argmax | 0.5066 | 0.3895 | 0.3874 | 0.452 | 0.178 | 0.156 |
| hierarchical_lr | thresholded | 0.4541 | 0.3781 | 0.459 | 0.574 | 0.348 | 0.0 |

## Ranking fatal y grave

| model | target | pr_auc | lift@2 | lift@5 | recall@10 |
| --- | --- | --- | --- | --- | --- |
| flat_4class_lgbm | fatal | 0.00638 | 4.683 | 3.749 | 0.4062 |
| flat_4class_lgbm | grave | 0.23526 | 3.221 | 3.091 | 0.2997 |
| hierarchical_lgbm | fatal | 0.00583 | 6.243 | 4.374 | 0.4062 |
| hierarchical_lgbm | grave | 0.23651 | 3.102 | 3.138 | 0.2935 |
| flat_4class_lr | fatal | 0.00528 | 4.683 | 4.374 | 0.4375 |
| flat_4class_lr | grave | 0.23 | 3.481 | 2.949 | 0.267 |
| hierarchical_lr | fatal | 0.00704 | 3.122 | 4.999 | 0.5 |
| hierarchical_lr | grave | 0.24335 | 3.268 | 2.939 | 0.2802 |

## Nivel 3 (fatal|grave) — calidad condicional

| model | n_grave | n_fatal | base_rate | pr_auc_fatal_given_grave |
| --- | --- | --- | --- | --- |
| hierarchical_lgbm | 2109 | 32 | 0.01517 | 0.0238 |
| hierarchical_lr | 2109 | 32 | 0.01517 | 0.0259 |

## Bootstrap CI (test, 500 iter)

{
  "flat_4class_lgbm": {
    "f1_macro": {
      "median": 0.3978489805639457,
      "ci95_low": 0.39242546268054196,
      "ci95_high": 0.4033394962832309
    },
    "kappa_q": {
      "median": 0.5538926240264195,
      "ci95_low": 0.5421879010267306,
      "ci95_high": 0.5645605603132573
    },
    "pr_auc_fatal": {
      "median": 0.0068368110136700876,
      "ci95_low": 0.003034049087053481,
      "ci95_high": 0.02213460363430789
    },
    "pr_auc_grave": {
      "median": 0.2364913464468278,
      "ci95_low": 0.22185084053688855,
      "ci95_high": 0.2515125786298453
    },
    "lift@2_fatal": {
      "median": 4.994700460829494,
      "ci95_low": 0.0,
      "ci95_high": 11.183446500576025
    }
  },
  "hierarchical_lgbm": {
    "f1_macro": {
      "median": 0.39662193540357826,
      "ci95_low": 0.3914148971088015,
      "ci95_high": 0.4024839665882778
    },
    "kappa_q": {
      "median": 0.5560116261644894,
      "ci95_low": 0.5450231486056284,
      "ci95_high": 0.5671583093115811
    },
    "pr_auc_fatal": {
      "median": 0.006497863859142449,
      "ci95_low": 0.003188318624328545,
      "ci95_high": 0.011739647937105625
    },
    "pr_auc_grave": {
      "median": 0.23761587149065871,
      "ci95_low": 0.22287022723494468,
      "ci95_high": 0.252935257881329
    },
    "lift@2_fatal": {
      "median": 6.2433755760368665,
      "ci95_low": 0.608328902280517,
      "ci95_high": 12.486751152073733
    }
  },
  "flat_4class_lr": {
    "f1_macro": {
      "median": 0.38294621902488307,
      "ci95_low": 0.3768828749526603,
      "ci95_high": 0.3885030507497475
    },
    "kappa_q": {
      "median": 0.4174233353293395,
      "ci95_low": 0.40550459964300106,
      "ci95_high": 0.4289963460288448
    },
    "pr_auc_fatal": {
      "median": 0.0057284498497016815,
      "ci95_low": 0.0030684639900123677,
      "ci95_high": 0.010900712501891112
    },
    "pr_auc_grave": {
      "median": 0.23005211355355007,
      "ci95_low": 0.21636139135561436,
      "ci95_high": 0.24470754332968536
    },
    "lift@2_fatal": {
      "median": 5.304522105956135,
      "ci95_low": 0.0,
      "ci95_high": 11.390691884280582
    }
  },
  "hierarchical_lr": {
    "f1_macro": {
      "median": 0.38975537816930184,
      "ci95_low": 0.3830750463795185,
      "ci95_high": 0.39607097287362575
    },
    "kappa_q": {
      "median": 0.5066785692232876,
      "ci95_low": 0.4961033091366594,
      "ci95_high": 0.5193484192955655
    },
    "pr_auc_fatal": {
      "median": 0.007617896277125481,
      "ci95_low": 0.0036523048110933746,
      "ci95_high": 0.015761148194271005
    },
    "pr_auc_grave": {
      "median": 0.24379893800115976,
      "ci95_low": 0.22902412565721547,
      "ci95_high": 0.25918499438414383
    },
    "lift@2_fatal": {
      "median": 2.9825751414935473,
      "ci95_low": 0.0,
      "ci95_high": 7.531781263091745
    }
  }
}

## Thresholds tuneados en val

{
  "flat_4class_lgbm": {
    "0": 0.0,
    "1": 0.327,
    "2": 0.139,
    "3": 0.01
  },
  "hierarchical_lgbm": {
    "0": 0.0,
    "1": 0.337,
    "2": 0.149,
    "3": 0.01
  },
  "flat_4class_lr": {
    "0": 0.0,
    "1": 0.208,
    "2": 0.257,
    "3": 0.95
  },
  "hierarchical_lr": {
    "0": 0.0,
    "1": 0.238,
    "2": 0.178,
    "3": 0.743
  }
}

## Confusion matrix — best (hierarchical_lr, argmax)

| | no_injury | minor | serious | fatal |
| --- | --- | --- | --- | --- |
| no_injury | 11029 | 1105 | 295 | 89 |
| minor | 2478 | 3189 | 999 | 384 |
| serious | 430 | 1056 | 369 | 222 |
| fatal | 6 | 14 | 7 | 5 |

## Best by PR-AUC(fatal)

hierarchical_lr

## Política class_weight

LGBM: None
LR:   balanced
Consistente con script 28 (funcionó con estos valores).

## Composición jerárquica

P(no_injury) = 1 − p_injured
P(minor)     = p_injured · (1 − p_grave|injured)
P(serious)   = p_injured · p_grave|injured · (1 − p_fatal|grave)
P(fatal)     = p_injured · p_grave|injured · p_fatal|grave

Nivel 1: entrenado en 74k train (47% injured)
Nivel 2: entrenado en train[injured] (~35k, 22% grave)
Nivel 3: entrenado en train[grave] (~7.8k, 1.5% fatal)
