# Source Audit Report

## URL availability

See `01_url_availability.json` for HTTP status and content-type of each source.

## Schema samples

### accidents_2019

- **Columns**: num_expediente;fecha;hora;localizacion;numero;cod_distrito;distrito;tipo_accidente;estado_meteorológico;tipo_vehiculo;tipo_persona;rango_edad;sexo;cod_lesividad;lesividad;coordenada_x_utm;coordenada_y_utm;positiva_alcohol;positiva_droga

- **Dtypes (sample)**: {'num_expediente;fecha;hora;localizacion;numero;cod_distrito;distrito;tipo_accidente;estado_meteorológico;tipo_vehiculo;tipo_persona;rango_edad;sexo;cod_lesividad;lesividad;coordenada_x_utm;coordenada_y_utm;positiva_alcohol;positiva_droga': 'str'}

- **Null counts (sample of 5 rows)**: {'num_expediente;fecha;hora;localizacion;numero;cod_distrito;distrito;tipo_accidente;estado_meteorológico;tipo_vehiculo;tipo_persona;rango_edad;sexo;cod_lesividad;lesividad;coordenada_x_utm;coordenada_y_utm;positiva_alcohol;positiva_droga': 0}

### accidents_2024
- **ERROR**: 404 Client Error: Not Found for url: https://datos.madrid.es/egob/catalogo/300228-33-accidentes-trafico-detalle.csv

### permanent_counts
- **ERROR**: 404 Client Error: Not Found for url: https://datos.madrid.es/egob/catalogo/300233-0-aforo-trafico-permanentes.csv

### traffic_history
- **ERROR**: 404 Client Error: Not Found for url: https://datos.madrid.es/egob/catalogo/208627-0-transporte-ptomedida-historico.csv

### street_directory
- **ERROR**: 'utf-8' codec can't decode byte 0xed in position 12: invalid continuation byte

## Pending verifications

- Confirm that `num_expediente` is unique per accident across 2019–2025.
- Check the proportion of valid UTM coordinates in the accidents dataset.
- Confirm whether 208627 provides point-level or segment-level traffic data.
- Confirm the structure of the street directory and whether it includes segments.
- Locate and verify the non-permanent counts dataset (300209).
