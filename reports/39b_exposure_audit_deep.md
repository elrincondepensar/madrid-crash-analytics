# Phase 9 · Iter 1b — Deep exposure audit
Resolves four open questions from iter 1.

## 208627 — resource-to-month mapping and full-month ID count
- idx 0 `12-2022.csv` (month inferred): 12,512,133 rows, **4487** unique ids (min=1001, max=9999), tipo_elem=['C30', 'M30', 'URB']
- idx 40 `06-2016.csv` (month inferred): 9,934,549 rows, **0** unique ids (min=None, max=None), tipo_elem=['PUNTOS MEDIDA M-30', 'PUNTOS MEDIDA URBANOS']
- idx 80 `208627-80-transporte-ptomedida-historico-zip`: ERROR download_failed
- idx 120 `02-2021.csv` (month inferred): 10,477,478 rows, **4181** unique ids (min=10000, max=9999), tipo_elem=['M30', 'URB']
- idx 160 `09-2026.csv` (month inferred): 13,152,123 rows, **4794** unique ids (min=10000, max=9999), tipo_elem=['C30', 'M30', 'URB']

## 202468 — recovered?
- dataset_id: `202468-0-intensidad-trafico`
- title: `Tráfico. Ubicación de los puntos de medida del tráfico`
- columns: ['tipo_elem', 'distrito', 'id', 'cod_cent', 'nombre', 'utm_x', 'utm_y', 'longitud', 'latitud']
- tipo_elem values: {'URB': 459, 'other': 41}

## 205865 — forgotten dataset
- title: `Tráfico. Intensidad media en días laborables`
- columns: ['Año', 'Mes', 'Interior 1er  cinturón', 'En el 1er  cinturón', 'Entre 1er y 2º cinturón', 'En el 2er  cinturón', 'Entre  2º cinturón y M-30', 'M-30', 'Entre M-30 y M-40', 'M-40', 'Exterior a M-40']
- sample rows: [{'Año': 2023, 'Mes': nan, 'Interior 1er  cinturón': nan, 'En el 1er  cinturón': nan, 'Entre 1er y 2º cinturón': nan, 'En el 2er  cinturón': nan, 'Entre  2º cinturón y M-30': nan, 'M-30': nan, 'Entre M-30 y M-40': nan, 'M-40': nan, 'Exterior a M-40': nan}, {'Año': 2023, 'Mes': '     Enero', 'Interior 1er  cinturón': 100.994, 'En el 1er  cinturón': 224.834, 'Entre 1er y 2º cinturón': 362.963, 'En el 2er  cinturón': 189.928, 'Entre  2º cinturón y M-30': 385.221, 'M-30': 198.85, 'Entre M-30 y M-40': 318.627, 'M-40': nan, 'Exterior a M-40': 25.681}, {'Año': 2023, 'Mes': '     Febrero', 'Interior 1er  cinturón': 103.267, 'En el 1er  cinturón': 234.83, 'Entre 1er y 2º cinturón': 377.71, 'En el 2er  cinturón': 197.18, 'Entre  2º cinturón y M-30': 404.577, 'M-30': 203.704, 'Entre M-30 y M-40': 325.261, 'M-40': nan, 'Exterior a M-40': 26.142}]

## Updated decision matrix
| source | status | key facts |
| --- | --- | --- |
| `208627` | **MULTI_NETWORK** | tipo_elem=['C30', 'M30', 'PUNTOS MEDIDA M-30', 'PUNTOS MEDIDA URBANOS', 'URB'], ids_per_sample=[4487, 0, 4181, 4794] |
| `205865` | **USABLE** | columns=['Año', 'Mes', 'Interior 1er  cinturón', 'En el 1er  cinturón', 'Entre 1er y 2º cinturón', 'En el 2er  cinturón', 'Entre  2º cinturón y M-30', 'M-30', 'Entre M-30 y M-40', 'M-40', 'Exterior a M-40'] |
| `202468` | **USABLE** | columns=['tipo_elem', 'distrito', 'id', 'cod_cent', 'nombre', 'utm_x', 'utm_y', 'longitud', 'latitud'] |
| `300233` | **USABLE** | Resource names carry no date; URL structure may. Full coverage requires downloading each monthly file and checking FDIA. Not done in this audit. |

## Overall
Reassess after 205865 inspection.
