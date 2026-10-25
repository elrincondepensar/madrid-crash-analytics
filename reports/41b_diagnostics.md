# Phase 10 · Iter 1b — Diagnostics and model comparison

## Response: `n_injured`

### Model comparison (Poisson AIC)
| model | AIC | ΔAIC vs M1 |
| --- | ---: | ---: |
| `M0_poisson_null` | 182455.1 | -2806.1 |
| `M1_poisson_offset` | 179649.0 | +0.0 |
| `M2_poisson_free_exposure` | 171191.4 | +8457.5 |
| `M3_poisson_offset_plus_covariates` | 165918.6 | +13730.4 |
| `M4_poisson_covariates_only` | 155156.8 | +24492.1 |

### Is exposure proportional? (M2 test α=1)
- coefficient on `log_exposure`: **0.5326** (SE=0.0051)
- Test H0: coef=1 → z=-92.05, p=0.0
  - **Cannot reject proportionality**. Offset is statistically justified.

### NB2 robust fitting
- Failed all attempts: [{'method': 'newton', 'ok': True, 'aic': nan}, {'method': 'newton', 'ok': False, 'error': "'numpy.ndarray' object has no attribute 'iloc'"}, {'method': 'nm', 'ok': True, 'aic': 174698.08621339384}, {'method': 'nm', 'ok': False, 'error': "'numpy.ndarray' object has no attribute 'iloc'"}, {'method': 'glm_nb_fixed_alpha', 'ok': True, 'aic': 165823.6482450571}, {'method': 'glm_nb_fixed_alpha', 'ok': False, 'error': "'numpy.ndarray' object has no attribute 'iloc'"}]

### Overdispersion
- Cameron-Trivedi alpha=-13.7833, p=0.06184716430350545
- Pearson chi2/df = 30.53

### Moran on M1 residuals
- I=0.2159 (p=0.0010, n=922)
- ΔI from Fase 8 (0.677): +0.461

### Residual hotspots (mean per hex over 84 months)
**Top 5 positive (más accidentes de lo esperado):**
- hex 2215: resid=12.630
- hex 3041: resid=9.036
- hex 1702: resid=7.046
- hex 3948: resid=6.228
- hex 1752: resid=6.108

**Top 5 negative (menos accidentes de lo esperado):**
- hex 1765: resid=-2.501
- hex 2208: resid=-2.008
- hex 2655: resid=-1.992
- hex 2656: resid=-1.880
- hex 3494: resid=-1.849

### Bootstrap CI on α
- mean α=-14.0797, 95% CI [-14.0920, -14.0682], n=500

## Response: `n_all`

### Model comparison (Poisson AIC)
| model | AIC | ΔAIC vs M1 |
| --- | ---: | ---: |
| `M0_poisson_null` | 284357.5 | -4770.8 |
| `M1_poisson_offset` | 279586.7 | +0.0 |
| `M2_poisson_free_exposure` | 260546.6 | +19040.1 |
| `M3_poisson_offset_plus_covariates` | 254296.3 | +25290.4 |
| `M4_poisson_covariates_only` | 230460.7 | +49126.1 |

### Is exposure proportional? (M2 test α=1)
- coefficient on `log_exposure`: **0.5246** (SE=0.0034)
- Test H0: coef=1 → z=-138.12, p=0.0
  - **Cannot reject proportionality**. Offset is statistically justified.

### NB2 robust fitting
- Failed all attempts: [{'method': 'newton', 'ok': True, 'aic': nan}, {'method': 'newton', 'ok': False, 'error': "'numpy.ndarray' object has no attribute 'iloc'"}, {'method': 'nm', 'ok': True, 'aic': 241816.2853426356}, {'method': 'nm', 'ok': False, 'error': "'numpy.ndarray' object has no attribute 'iloc'"}, {'method': 'glm_nb_fixed_alpha', 'ok': True, 'aic': 244215.82250609045}, {'method': 'glm_nb_fixed_alpha', 'ok': False, 'error': "'numpy.ndarray' object has no attribute 'iloc'"}]

### Overdispersion
- Cameron-Trivedi alpha=-12.2232, p=0.0030641187207154363
- Pearson chi2/df = 44.83

### Moran on M1 residuals
- I=0.2158 (p=0.0010, n=922)
- ΔI from Fase 8 (0.677): +0.461

### Residual hotspots (mean per hex over 84 months)
**Top 5 positive (más accidentes de lo esperado):**
- hex 2215: resid=19.417
- hex 2204: resid=15.771
- hex 1686: resid=12.448
- hex 1301: resid=10.298
- hex 1377: resid=10.251

**Top 5 negative (menos accidentes de lo esperado):**
- hex 1765: resid=-3.689
- hex 2208: resid=-3.009
- hex 2655: resid=-2.942
- hex 2656: resid=-2.752
- hex 3494: resid=-2.587

### Bootstrap CI on α
- mean α=-13.3017, 95% CI [-13.3108, -13.2920], n=500

## Conclusions
- **n_injured**: exposure aporta ΔAIC M0→M1 = 2806 (cuanto mayor, más útil exposure). Covariables añaden ΔAIC M1→M3 = 13730 (si > 10, aportan). Sin exposure, covariables dan ΔAIC M1→M4 = 24492 (si < 0, covariables empeoran sin exposure).
- **n_all**: exposure aporta ΔAIC M0→M1 = 4771 (cuanto mayor, más útil exposure). Covariables añaden ΔAIC M1→M3 = 25290 (si > 10, aportan). Sin exposure, covariables dan ΔAIC M1→M4 = 49126 (si < 0, covariables empeoran sin exposure).