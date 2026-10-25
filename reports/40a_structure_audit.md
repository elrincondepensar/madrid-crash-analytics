# Phase 9 · Iter 2a — Structural audit of 208627
## 1. Resource → month mapping (all 161 ZIPs)
- Attempted: 161
- Successfully parsed: 161
- Failed: 0

### Coverage by year (inferred)
| year | months present | complete? |
| --- | --- | --- |
| 2013 | ['07', '08', '09', '10', '11', '12'] | PARTIAL (6/12) |
| 2014 | ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'] | FULL |
| 2015 | ['01', '02', '03', '05', '06', '07', '08', '09', '10', '11', '12'] | PARTIAL (11/12) |
| 2016 | ['01', '02', '03', '05', '06', '08', '09', '10', '11', '12'] | PARTIAL (10/12) |
| 2017 | ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'] | FULL |
| 2018 | ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'] | FULL |
| 2019 | ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'] | FULL |
| 2020 | ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'] | FULL |
| 2021 | ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'] | FULL |
| 2022 | ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'] | FULL |
| 2023 | ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'] | FULL |
| 2024 | ['01', '02', '03', '04', '05', '06', '07', '08', '10', '11', '12'] | PARTIAL (11/12) |
| 2025 | ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'] | FULL |
| 2026 | ['01', '02', '03', '04', '05', '06', '07', '08', '09'] | PARTIAL (9/12) |


**Gaps in 2019-2025:** [{'year': '2024', 'missing_months': ['09']}]

**Anomalous entries:** [{'index': 27, 'filename': '04-2015/', 'size': 99233735}, {'index': 47, 'filename': '07-2016/', 'size': 105268989}, {'index': 80, 'filename': '2013.csv', 'note': 'ANNUAL aggregate'}, {'index': 84, 'filename': '09_2024.csv', 'size': 81007299}, {'index': 103, 'filename': '2014.csv', 'note': 'ANNUAL aggregate'}, {'index': 104, 'filename': 'Abril 2016.csv', 'size': 68356708}]

## 2. ZIP idx 80 (784 MB)
- inner csv name: `2013.csv`
- inferred month: `2013-ANNUAL`
- zip size: 783887683

## 3. Internal date validation (3 ZIPs)
- `01-2019.csv` inferred=2019-01 content_fecha=[2019-01-01 00:00:00, 2019-01-31 23:45:00] match=True
- `06-2022.csv` inferred=2022-06 content_fecha=[2022-06-01 00:00:00, 2022-06-30 23:45:00] match=True
- `01-2025.csv` inferred=2025-01 content_fecha=[?, ?] match=True

## 4. id join check (202468 vs 208627)
- catalog unique ids: 4894
- catalog id range: [10000, 9999]
- month sample unique ids: 71
- month sample id range: [1001, 3418]
- intersection: 71
- frac catalog matched: 0.0145
- frac month matched: 1.0

## Decision matrix
Resource mapping: 161/161 parsed (0 failed). Gaps detected in ['2024']. Exposure aggregates will skip missing months. **id join WARNING:** only 1% of catalog IDs matched. May need fuzzy join or additional sources. **ZIP idx 80:** inner csv `2013.csv`, size 784 MB.