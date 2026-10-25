# Phase 8 pre-check · Administrative geometries

## Local files

| path | type | n_rows | CRS | geom | name cols |
| --- | --- | ---: | --- | --- | --- |
| `data\processed\accidents_clean.parquet` | parquet (pandas) | 137328 | n/a | n/a | ['cod_distrito', 'distrito'] |
| `data\processed\accidents_with_edges.parquet` | parquet (pandas) | 137295 | n/a | n/a | [] |
| `data\processed\accidents_with_network_features.parquet` | parquet (pandas) | 137295 | n/a | n/a | ['edge_name', 'nombre_via'] |
| `data\processed\accidents_with_segment.parquet` | parquet (pandas) | 137295 | n/a | n/a | [] |
| `data\processed\baseline_test_predictions.parquet` | parquet (pandas) | 42386 | n/a | n/a | [] |
| `data\processed\hdbscan_clusters_all.parquet` | parquet (pandas) | 74250 | n/a | n/a | ['distrito'] |
| `data\processed\hdbscan_clusters_injured.parquet` | parquet (pandas) | 35107 | n/a | n/a | ['distrito'] |
| `data\processed\hdbscan_persistence_all.parquet` | parquet (pandas) | 137295 | n/a | n/a | [] |
| `data\processed\hdbscan_persistence_injured.parquet` | parquet (pandas) | 62460 | n/a | n/a | [] |
| `data\processed\imd_points.parquet` | parquet (pandas) | 1961 | n/a | n/a | ['name', 'Nombre'] |
| `data\processed\network_callejero.parquet` | parquet (pandas) | 9408 | n/a | str | [] |
| `data\processed\network_kde_b.parquet` | parquet (pandas) | 95437 | n/a | n/a | [] |
| `data\processed\network_kde_c.parquet` | parquet (pandas) | 95437 | n/a | n/a | [] |
| `data\processed\network_kde_final.parquet` | parquet (pandas) | 95437 | n/a | n/a | [] |
| `data\processed\network_osm.parquet` | parquet (pandas) | 95437 | n/a | str | ['name'] |
| `data\processed\network_segments.parquet` | parquet (pandas) | 3928 | n/a | str | ['name'] |
| `data\processed\persons_clean.parquet` | parquet (pandas) | 322317 | n/a | n/a | ['cod_distrito', 'distrito', 'cod_lesividad'] |
| `data\processed\segments_with_imd.parquet` | parquet (pandas) | 3928 | n/a | str | ['name'] |
| `data\processed\splits.parquet` | parquet (pandas) | 137328 | n/a | n/a | [] |

## Callejero (network_callejero.parquet)

- rows: **9,408**
- columns: `['TOP_ID', 'TOP_ID_COM', 'TOP_ID_TER', 'TVIA_ID', 'TOP_DT_FCH', 'TOP_DT_F_1', 'CV_TX_DENO', 'DEN_TX_PAR', 'DEN_TX_NOM', 'geometry_wkt']`
- district-like: `[]`
- neighbourhood-like: `[]`

## CKAN candidates

### Query: `distritos municipales`

- **Distritos municipales de Madrid** (`300497-0-distritos-municipales-madrid`) — 7 resources, formats: ['CSV', 'KML', 'TXT', 'XLSX', 'ZIP']
    - `CSV`: https://sigma.madrid.es/hosted/rest/services/CARTOGRAFIA/LIMITES_ADMINISTRATIVOS/MapServer
    - `XLSX`: https://geoportal.madrid.es/fsdescargas/IDEAM_WBGEOPORTAL/LIMITES_ADMINISTRATIVOS/Distritos/Distritos.xlsx
    - `ZIP`: https://geoportal.madrid.es/fsdescargas/IDEAM_WBGEOPORTAL/LIMITES_ADMINISTRATIVOS/Distritos/Distritos.zip
    - `CSV`: https://geoportal.madrid.es/fsdescargas/IDEAM_WBGEOPORTAL/LIMITES_ADMINISTRATIVOS/Distritos/Distritos.csv
- **Juntas Municipales de Distrito** (`206688-0-juntas-distrito`) — 6 resources, formats: ['CSV', 'GEO', 'JSON', 'PDF', 'RDF', 'XML']
    - `CSV`: https://datos.madrid.es/dataset/206688-0-juntas-distrito/resource/206688-0-juntas-distrito-csv/download/206688-0-juntas-distrito.csv
    - `JSON`: https://datos.madrid.es/dataset/206688-0-juntas-distrito/resource/206688-1-juntas-distrito-json/download/206688-0-juntas-distrito.json
    - `GEO`: https://datos.madrid.es/dataset/206688-0-juntas-distrito/resource/206688-2-juntas-distrito-geo/download/206688-0-juntas-distrito.geo
    - `RDF`: https://datos.madrid.es/dataset/206688-0-juntas-distrito/resource/206688-3-juntas-distrito-rdf/download/206688-0-juntas-distrito.rdf
- **Barrios municipales de Madrid** (`300496-0-barrios-madrid`) — 6 resources, formats: ['CSV', 'KML', 'TXT', 'XLSX', 'ZIP']
    - `KML`: https://geoportal.madrid.es/fsdescargas/IDEAM_WBGEOPORTAL/LIMITES_ADMINISTRATIVOS/Barrios/Barrios.kml
    - `XLSX`: https://geoportal.madrid.es/fsdescargas/IDEAM_WBGEOPORTAL/LIMITES_ADMINISTRATIVOS/Barrios/Barrios.xlsx
    - `TXT`: https://geoportal.madrid.es/fsdescargas/IDEAM_WBGEOPORTAL/LIMITES_ADMINISTRATIVOS/Barrios/Barrios.txt
    - `ZIP`: https://geoportal.madrid.es/fsdescargas/IDEAM_WBGEOPORTAL/LIMITES_ADMINISTRATIVOS/Barrios/Barrios.zip
- **Cartografía municipal por distritos a escala 1:1000, formato SHP, ETRS89** (`213565-0-cartografia-distritos-1-1000`) — 23 resources, formats: ['CSV', 'ZIP']
    - `ZIP`: https://datos.madrid.es/dataset/213565-0-cartografia-distritos-1-1000/resource/213565-0-cartografia-distritos-1-1000-zip/download/213565-0-cartografia-distritos-1-1000-zip.zip
    - `CSV`: https://datos.madrid.es/dataset/213565-0-cartografia-distritos-1-1000/resource/213565-1-cartografia-distritos-1-1000-csv/download/213565-1-cartografia-distritos-1-1000-csv.csv
    - `ZIP`: https://datos.madrid.es/dataset/213565-0-cartografia-distritos-1-1000/resource/213565-2-cartografia-distritos-1-1000-zip/download/213565-2-cartografia-distritos-1-1000-zip.zip
    - `ZIP`: https://datos.madrid.es/dataset/213565-0-cartografia-distritos-1-1000/resource/213565-3-cartografia-distritos-1-1000-zip/download/213565-3-cartografia-distritos-1-1000-zip.zip
- **Cartografía municipal por distritos a escala 1:5000, formato SHP, ETRS89** (`213583-0-cartografia-distritos-1-5000`) — 23 resources, formats: ['CSV', 'ZIP']
    - `ZIP`: https://datos.madrid.es/dataset/213583-0-cartografia-distritos-1-5000/resource/213583-0-cartografia-distritos-1-5000-zip/download/213583-0-cartografia-distritos-1-5000-zip.zip
    - `ZIP`: https://datos.madrid.es/dataset/213583-0-cartografia-distritos-1-5000/resource/213583-1-cartografia-distritos-1-5000-zip/download/213583-1-cartografia-distritos-1-5000-zip.zip
    - `ZIP`: https://datos.madrid.es/dataset/213583-0-cartografia-distritos-1-5000/resource/213583-2-cartografia-distritos-1-5000-zip/download/213583-2-cartografia-distritos-1-5000-zip.zip
    - `ZIP`: https://datos.madrid.es/dataset/213583-0-cartografia-distritos-1-5000/resource/213583-3-cartografia-distritos-1-5000-zip/download/213583-3-cartografia-distritos-1-5000-zip.zip

### Query: `barrios municipales`

- **Barrios municipales de Madrid** (`300496-0-barrios-madrid`) — 6 resources, formats: ['CSV', 'KML', 'TXT', 'XLSX', 'ZIP']
    - `KML`: https://geoportal.madrid.es/fsdescargas/IDEAM_WBGEOPORTAL/LIMITES_ADMINISTRATIVOS/Barrios/Barrios.kml
    - `XLSX`: https://geoportal.madrid.es/fsdescargas/IDEAM_WBGEOPORTAL/LIMITES_ADMINISTRATIVOS/Barrios/Barrios.xlsx
    - `TXT`: https://geoportal.madrid.es/fsdescargas/IDEAM_WBGEOPORTAL/LIMITES_ADMINISTRATIVOS/Barrios/Barrios.txt
    - `ZIP`: https://geoportal.madrid.es/fsdescargas/IDEAM_WBGEOPORTAL/LIMITES_ADMINISTRATIVOS/Barrios/Barrios.zip
- **Panel de indicadores de distritos y barrios de Madrid. Estudio sociodemográfico** (`300087-0-indicadores-distritos`) — 23 resources, formats: ['CSV', 'PDF', 'XLS', 'XLSX']
    - `CSV`: https://datos.madrid.es/dataset/300087-0-indicadores-distritos/resource/300087-0-indicadores-distritos-csv/download/acumulado.csv
    - `XLSX`: https://datos.madrid.es/dataset/300087-0-indicadores-distritos/resource/300087-1-indicadores-distritos-xlsx/download/estructura_conjunto_datos_panel_indicadores_distritos_barrios-documento-tecnico.xlsx
    - `XLSX`: https://datos.madrid.es/dataset/300087-0-indicadores-distritos/resource/300087-2-indicadores-distritos-xlsx/download/300087-2-indicadores-distritos-xlsx.xlsx
    - `XLS`: https://datos.madrid.es/dataset/300087-0-indicadores-distritos/resource/300087-3-indicadores-distritos-xls/download/300087-3-indicadores-distritos-xls.xls
- **Límites administrativos actuales** (`900012-0-limites-administrativos-mapas`) — 8 resources, formats: ['JSON', 'ZIP']
    - `ZIP`: https://geoportal.madrid.es/fsdescargas/IDEAM_WBGEOPORTAL/LIMITES_ADMINISTRATIVOS/Termino_Municipal/Termino_municipal.zip
    - `ZIP`: https://geoportal.madrid.es/fsdescargas/IDEAM_WBGEOPORTAL/LIMITES_ADMINISTRATIVOS/Seccionado/Secciones_Censales.zip
    - `ZIP`: https://geoportal.madrid.es/fsdescargas/IDEAM_WBGEOPORTAL/LIMITES_ADMINISTRATIVOS/Distritos/Distritos.zip
    - `JSON`: https://geoportal.madrid.es/fsdescargas/IDEAM_WBGEOPORTAL/LIMITES_ADMINISTRATIVOS/Distritos/TopoJSON/Distritos.json
- **Mercados municipales** (`200967-0-mercados`) — 6 resources, formats: ['CSV', 'GEO', 'JSON', 'PDF', 'RDF', 'XML']
    - `JSON`: https://datos.madrid.es/dataset/200967-0-mercados/resource/200967-0-mercados-json/download/200967-0-mercados.json
    - `GEO`: https://datos.madrid.es/dataset/200967-0-mercados/resource/200967-1-mercados-geo/download/200967-0-mercados.geo
    - `XML`: https://datos.madrid.es/dataset/200967-0-mercados/resource/200967-2-mercados-xml/download/200967-0-mercados.xml
    - `RDF`: https://datos.madrid.es/dataset/200967-0-mercados/resource/200967-3-mercados-rdf/download/200967-0-mercados.rdf
- **Padrón municipal** (`200076-0-padron`) — 4 resources, formats: ['CSV', 'JSON', 'PDF', 'XML']
    - `XML`: https://datos.madrid.es/dataset/200076-0-padron/resource/200076-0-padron-xml/download/200076_20261006_220058.xml
    - `JSON`: https://datos.madrid.es/dataset/200076-0-padron/resource/200076-1-padron-json/download/200076_20261006_220058.json
    - `CSV`: https://datos.madrid.es/dataset/200076-0-padron/resource/200076-2-padron-csv/download/200076_20261006_220058.csv
    - `PDF`: https://datos.madrid.es/dataset/200076-0-padron/resource/200076-3-padron/download/conceptos_fundamentales.pdf

### Query: `limites administrativos`

- **Límites administrativos actuales** (`900012-0-limites-administrativos-mapas`) — 8 resources, formats: ['JSON', 'ZIP']
    - `ZIP`: https://geoportal.madrid.es/fsdescargas/IDEAM_WBGEOPORTAL/LIMITES_ADMINISTRATIVOS/Termino_Municipal/Termino_municipal.zip
    - `ZIP`: https://geoportal.madrid.es/fsdescargas/IDEAM_WBGEOPORTAL/LIMITES_ADMINISTRATIVOS/Seccionado/Secciones_Censales.zip
    - `ZIP`: https://geoportal.madrid.es/fsdescargas/IDEAM_WBGEOPORTAL/LIMITES_ADMINISTRATIVOS/Distritos/Distritos.zip
    - `JSON`: https://geoportal.madrid.es/fsdescargas/IDEAM_WBGEOPORTAL/LIMITES_ADMINISTRATIVOS/Distritos/TopoJSON/Distritos.json
- **Seccionado censal** (`300724-0-seccionado-censal-mapas`) — 4 resources, formats: ['JSON', 'KML', 'XLS']
    - `XLS`: https://www.madrid.es/portales/munimadrid/es/Inicio/El-Ayuntamiento/Estadistica/Areas-de-informacion-estadistica/Territorio-climatologia-y-medio-ambiente/Territorio/Seccionado-censal/?vgnextfmt=default&vgnextoid=c13456bc06f59210VgnVCM2000000c205a0aRCRD&vgnextchannel=e59b40ebd232a210VgnVCM1000000b205a0aRCRD
    - `JSON`: https://geoportal.madrid.es/fsdescargas/IDEAM_WBGEOPORTAL/LIMITES_ADMINISTRATIVOS/Seccionado/TopoJSON/Secciones_Censales.json
    - `KML`: https://geoportal.madrid.es/fsdescargas/IDEAM_WBGEOPORTAL/LIMITES_ADMINISTRATIVOS/Seccionado/Secciones_Censales.kml
    - `XLS`: https://sigma.madrid.es/hosted/rest/services/CARTOGRAFIA/LIMITES_ADMINISTRATIVOS/MapServer
- **Término municipal de Madrid** (`300498-0-termino-municipal-madrid`) — 3 resources, formats: ['KML', 'ZIP']
    - `KML`: https://geoportal.madrid.es/fsdescargas/IDEAM_WBGEOPORTAL/LIMITES_ADMINISTRATIVOS/Termino_Municipal/Termino_Municipal.kml
    - `ZIP`: https://geoportal.madrid.es/fsdescargas/IDEAM_WBGEOPORTAL/LIMITES_ADMINISTRATIVOS/Barrios/Historicos/Divisiones_Historicas.zip
    - `ZIP`: https://geoportal.madrid.es/fsdescargas/IDEAM_WBGEOPORTAL/LIMITES_ADMINISTRATIVOS/Termino_Municipal/Termino_Municipal.zip
- **Distritos municipales de Madrid** (`300497-0-distritos-municipales-madrid`) — 7 resources, formats: ['CSV', 'KML', 'TXT', 'XLSX', 'ZIP']
    - `CSV`: https://sigma.madrid.es/hosted/rest/services/CARTOGRAFIA/LIMITES_ADMINISTRATIVOS/MapServer
    - `XLSX`: https://geoportal.madrid.es/fsdescargas/IDEAM_WBGEOPORTAL/LIMITES_ADMINISTRATIVOS/Distritos/Distritos.xlsx
    - `ZIP`: https://geoportal.madrid.es/fsdescargas/IDEAM_WBGEOPORTAL/LIMITES_ADMINISTRATIVOS/Distritos/Distritos.zip
    - `CSV`: https://geoportal.madrid.es/fsdescargas/IDEAM_WBGEOPORTAL/LIMITES_ADMINISTRATIVOS/Distritos/Distritos.csv
- **Barrios municipales de Madrid** (`300496-0-barrios-madrid`) — 6 resources, formats: ['CSV', 'KML', 'TXT', 'XLSX', 'ZIP']
    - `KML`: https://geoportal.madrid.es/fsdescargas/IDEAM_WBGEOPORTAL/LIMITES_ADMINISTRATIVOS/Barrios/Barrios.kml
    - `XLSX`: https://geoportal.madrid.es/fsdescargas/IDEAM_WBGEOPORTAL/LIMITES_ADMINISTRATIVOS/Barrios/Barrios.xlsx
    - `TXT`: https://geoportal.madrid.es/fsdescargas/IDEAM_WBGEOPORTAL/LIMITES_ADMINISTRATIVOS/Barrios/Barrios.txt
    - `ZIP`: https://geoportal.madrid.es/fsdescargas/IDEAM_WBGEOPORTAL/LIMITES_ADMINISTRATIVOS/Barrios/Barrios.zip

### Query: `division territorial`

_No hits._

## Decision checklist for Phase 8

Based on these results, decide:

- **Districts**: use local file if present, else download from the CKAN hit that lists a SHP/WFS resource.
- **Neighbourhoods**: same.
- **Street segments**: use `network_callejero.parquet` (P1=B) regardless.
- **Hex 250m**: no external geometry needed; generated from the Madrid municipal boundary (available in `network_osm` bbox and in `network_callejero`).
