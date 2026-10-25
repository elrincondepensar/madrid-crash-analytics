# Source Audit Report

## Audit v2 – CKAN discovery and corrected parsing

## accidents

- CSV resources found: 17 (readable: 17)

  - `300228-25-accidentes-trafico-detalle-csv` | https://datos.madrid.es/dataset/300228-0-accidentes-trafico-detalle/resource/300... | status: ok
    - columns: FECHA, RANGO HORARIO, DIA SEMANA, DISTRITO, LUGAR ACCIDENTE, Nº, Nº PARTE, CPFA Granizo, CPFA Hielo, CPFA Lluvia...
  - `300228-27-accidentes-trafico-detalle-csv` | https://datos.madrid.es/dataset/300228-0-accidentes-trafico-detalle/resource/300... | status: ok
    - columns: FECHA, RANGO HORARIO, DIA SEMANA, DISTRITO, LUGAR ACCIDENTE, Nº, Nº PARTE, CPFA Granizo, CPFA Hielo, CPFA Lluvia...
  - `300228-23-accidentes-trafico-detalle-csv` | https://datos.madrid.es/dataset/300228-0-accidentes-trafico-detalle/resource/300... | status: ok
    - columns: FECHA, RANGO HORARIO, DIA SEMANA, DISTRITO, LUGAR ACCIDENTE, Nº, Nº PARTE, CPFA Granizo, CPFA Hielo, CPFA Lluvia...
  - `300228-19-accidentes-trafico-detalle-csv` | https://datos.madrid.es/dataset/300228-0-accidentes-trafico-detalle/resource/300... | status: ok
    - columns: FECHA, RANGO HORARIO, DIA SEMANA, DISTRITO, LUGAR ACCIDENTE, Nº, Nº PARTE, CPFA Granizo, CPFA Hielo, CPFA Lluvia...
  - `300228-20-accidentes-trafico-detalle-csv` | https://datos.madrid.es/dataset/300228-0-accidentes-trafico-detalle/resource/300... | status: ok
    - columns: FECHA, RANGO HORARIO, DIA SEMANA, DISTRITO, LUGAR ACCIDENTE, Nº, Nº PARTE, CPFA Granizo, CPFA Hielo, CPFA Lluvia...

## permanent_counts

- CSV resources found: 89 (readable: 89)

  - `300233-0-aforo-trafico-permanentes-csv` | https://datos.madrid.es/dataset/300233-0-aforo-trafico-permanentes/resource/3002... | status: ok
    - columns: ï»¿FDIA, FEST, FSEN, HOR1, HOR2, HOR3, HOR4, HOR5, HOR6, HOR7...
  - `300233-1-aforo-trafico-permanentes-csv` | https://datos.madrid.es/dataset/300233-0-aforo-trafico-permanentes/resource/3002... | status: ok
    - columns: ï»¿FDIA, FEST, FSEN, HOR1, HOR2, HOR3, HOR4, HOR5, HOR6, HOR7...
  - `300233-2-aforo-trafico-permanentes-csv` | https://datos.madrid.es/dataset/300233-0-aforo-trafico-permanentes/resource/3002... | status: ok
    - columns: ï»¿FDIA, FEST, FSEN, HOR1, HOR2, HOR3, HOR4, HOR5, HOR6, HOR7...
  - `300233-3-aforo-trafico-permanentes-csv` | https://datos.madrid.es/dataset/300233-0-aforo-trafico-permanentes/resource/3002... | status: ok
    - columns: ï»¿FDIA, FEST, FSEN, HOR1, HOR2, HOR3, HOR4, HOR5, HOR6, HOR7...
  - `300233-4-aforo-trafico-permanentes-csv` | https://datos.madrid.es/dataset/300233-0-aforo-trafico-permanentes/resource/3002... | status: ok
    - columns: FDIA, FEST, FSEN, HOR1, HOR2, HOR3, HOR4, HOR5, HOR6, HOR7...

## traffic_history

- CSV resources found: 0 (readable: 0)


## street_directory

- CSV resources found: 6 (readable: 6)

  - `213605-1-callejero-oficial-madrid-csv` | https://datos.madrid.es/dataset/213605-0-callejero-oficial-madrid/resource/21360... | status: ok
    - columns: COD_VIA, VIA_SQC, VIA_CLASE, VIA_PAR, VIA_NOMBRE, VIA_NOMBRE_ACENTOS, COD_NDP, CLASE_NDP, NÚMERO, CALIFICADOR...
  - `213605-2-callejero-oficial-madrid-csv` | https://datos.madrid.es/dataset/213605-0-callejero-oficial-madrid/resource/21360... | status: ok
    - columns: COD_VIA, VIA_SQC, VIA_CLASE, VIA_PAR, VIA_NOMBRE, VIA_NOMBRE_ACENTOS, FECHA_DE_ALTA, FECHA_DE_BAJA, VIA_ESTADO...
  - `213605-4-callejero-oficial-madrid-csv` | https://datos.madrid.es/dataset/213605-0-callejero-oficial-madrid/resource/21360... | status: ok
    - columns: COD_VIA, VIA_CLASE, VIA_PAR, VIA_NOMBRE, VIA_NOMBRE_ACENTOS, CLASE_APP, NUMERO, CALIFICADOR, TIPO_NDP, COD_NDP...
  - `213605-3-callejero-oficial-madrid-csv` | https://datos.madrid.es/dataset/213605-0-callejero-oficial-madrid/resource/21360... | status: ok
    - columns: COD_VIA, VIA_CLASE, VIA_PAR, VIA_NOMBRE, VIA_NOMBRE_ACENTOS, DISTRITO, BARRIO, IMPAR_MIN, IMPAR_MAX, PAR_MIN...
  - `213605-0-callejero-oficial-madrid-csv` | https://datos.madrid.es/dataset/213605-0-callejero-oficial-madrid/resource/21360... | status: ok
    - columns: COD_VIA, VIA_CLASE, VIA_PAR, VIA_NOMBRE, VIA_NOMBRE_ACENTOS, DISTRITO, IMPAR_MIN, IMPAR_MAX, PAR_MIN, PAR_MAX...

## Pending verifications

- Confirm full uniqueness of `num_expediente` across all years (Phase 2).
- Quantify valid UTM coordinates in the full accidents dataset.
- Confirm granularity of traffic history (point vs. segment).
- Confirm whether street directory includes segments or only vials.
