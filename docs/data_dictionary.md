# Data dictionary
Column schemas for the principal parquets in `data/processed/`.
Types are PyArrow type strings.

## accidents_clean.parquet

One row per accident (severity, location, district, covariates). 137,328 rows.

| column | type |
| --- | --- |
| num_expediente | large_string |
| year | int64 |
| fecha | timestamp[us] |
| hora | large_string |
| hour | int8 |
| localizacion | large_string |
| numero | double |
| cod_distrito | double |
| distrito | large_string |
| coordenada_x_utm | double |
| coordenada_y_utm | double |
| tipo_accidente | large_string |
| estado_meteorologico | large_string |
| n_people | int64 |
| n_vehicles | int64 |
| n_conductores | int64 |
| n_pasajeros | int64 |
| n_peatones | int64 |
| has_bicycle | bool |
| has_motorcycle | bool |
| has_heavy_vehicle | bool |
| has_pedestrian | bool |
| any_alcohol_positive | bool |
| n_alcohol_positive | int64 |
| any_drug_positive | bool |
| n_fatalities | int64 |
| n_serious | int64 |
| n_minor | int64 |
| worst_lesividad_code | int64 |
| severity | large_string |
| severity_code | int8 |

## persons_clean.parquet

One row per person involved. 322,317 rows.

| column | type |
| --- | --- |
| num_expediente | large_string |
| fecha | timestamp[us] |
| hora | large_string |
| localizacion | large_string |
| numero | double |
| cod_distrito | double |
| distrito | large_string |
| tipo_accidente | large_string |
| estado_meteorologico | large_string |
| tipo_vehiculo | large_string |
| tipo_persona | large_string |
| rango_edad | large_string |
| sexo | large_string |
| cod_lesividad | double |
| lesividad | large_string |
| coordenada_x_utm | double |
| coordenada_y_utm | double |
| positiva_alcohol | large_string |
| positiva_droga | double |
| file_year | int64 |
| year | int64 |
| hour | int8 |
| is_driver | bool |
| is_passenger | bool |
| is_pedestrian | bool |
| is_bicycle | bool |
| is_motorcycle | bool |
| is_heavy_vehicle | bool |
| is_fatal | bool |
| is_serious | bool |
| is_minor | bool |
| has_known_lesividad | bool |
| alcohol_positive | bool |
| drug_positive | bool |

## splits.parquet

Train/val/test assignment by num_expediente.

| column | type |
| --- | --- |
| num_expediente | large_string |
| year | int64 |
| severity | large_string |
| severity_code | int8 |
| split | large_string |

## network_osm.parquet

OSM drive network (edges as LineStrings).

| column | type |
| --- | --- |
| osm_id | large_string |
| u | int64 |
| v | int64 |
| key | int64 |
| highway | large_string |
| name | large_string |
| maxspeed_kmh | double |
| lanes_n | double |
| oneway | bool |
| length_m | double |
| geometry | binary |
| geometry_wkt | large_string |

## moran_hex250m.parquet

LISA clusters on hex250m grid.

| column | type |
| --- | --- |
| hex_id | int64 |
| geometry | binary |
| counts_all_cluster | large_string |
| counts_all_p_adj | double |
| counts_injured_cluster | large_string |
| counts_injured_p_adj | double |
| kde_all_cluster | large_string |
| kde_all_p_adj | double |
| injury_rate_eb_cluster | large_string |
| injury_rate_eb_p_adj | double |

## exposure_hex250m_monthly.parquet

Monthly traffic exposure per hex and per year.

| column | type |
| --- | --- |
| hex_id | int64 |
| year | int64 |
| month | int64 |
| n_sensors | int64 |
| n_sensors_capped | int64 |
| intensity_day_mean_mean | double |
| intensity_day_median | double |
| intensity_day_p90 | double |
| intensity_day_sum | double |
| coverage_mean | double |
| n_urb | int64 |
| n_m30_c30 | int64 |

## panel_hex_month_n_injured.parquet

Hex-month panel with count + exposure + covariates (Phase 10).

| column | type |
| --- | --- |
| hex_id | int64 |
| year | int64 |
| month | int64 |
| intensity_day_mean_mean | double |
| intensity_day_median | double |
| intensity_day_sum | double |
| n_sensors | int64 |
| coverage_mean | double |
| n_urb | int64 |
| n_m30_c30 | int64 |
| n_all | int64 |
| n_injured | int64 |
| days_in_month | int64 |
| exposure_month | double |
| log_exposure | double |
| ym | large_string |
| pearson_resid | double |
| fitted | double |
| year_int | int64 |
| model_used | large_string |
| y_col | large_string |
