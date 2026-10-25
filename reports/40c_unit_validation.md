# Phase 9 · Iter 2c — Unit validation
## Check 1: single-sensor daily profile
- Sensor id: 5669
- Intervals in month: 2855
- `periodo_integracion` values: {15: 1480, 14: 361, 13: 273, 12: 192, 11: 137, 10: 86, 9: 78, 8: 62, 7: 54, 3: 39, 6: 37, 5: 22, 4: 20, 2: 10, 1: 4}
- `intensidad` quantiles: {'q50': 43.0, 'q90': 84.0, 'q99': 152.29200000000037}
- **Daily sum (row-by-row sum over intervals):**
  - median: 4352
  - min: 2958
  - max: 5469
- Sample day hourly sum: {0: 179, 1: 46, 2: 65, 3: 30, 4: 19, 5: 27, 6: 68, 7: 114, 8: 215, 9: 302, 10: 415, 11: 402, 12: 378, 13: 321, 14: 382, 15: 238, 16: 280, 17: 265, 18: 234, 19: 279, 20: 209, 21: 220, 22: 165, 23: 140}
- if daily_sum ~ 10k-50k veh/day: units are veh/5min, current aggregation OK. if daily_sum > 200k: units are veh/hour (multiplied by 288 overcounts 12x).

## Check 2: 208627 vs 203962 IMD (hex overlap)
- n overlapping hexes: 612
- Spearman rho: **0.450** (expected > 0.7 if both measure intensity)
- Median ratio 208627/IMD: **5.57** (expected ~1.0 if 208627 is already in veh/day. ~12 if 208627 is overcounting by 12x (veh/hour × 288).)

## Check 3: raw `intensidad` distribution
- n rows sampled: 2,000,000
- min=0, median=252, mean=446, max=23235
- quantiles: {'q01': 2.0, 'q25': 92.0, 'q50': 252.0, 'q75': 596.0, 'q90': 1080.0, 'q99': 4488.0}
- if max per interval < 5000: units are plausible per-5min counts. if max per interval > 5000: check whether this is veh/hour.

## Conclusion
**Ratio vs IMD = 5.6** → overcounted by ~6x. Likely factor 12 (veh/hour × 288 vs veh/hour × 24).