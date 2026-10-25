"""Column role metadata for the processed accident tables."""

ACCIDENT_COLUMN_ROLES = {
    "identity": ["num_expediente", "year"],
    "temporal": ["fecha", "hora", "hour"],
    "spatial": ["localizacion", "numero", "cod_distrito", "distrito",
                "coordenada_x_utm", "coordenada_y_utm"],
    "context": ["tipo_accidente", "estado_meteorologico"],
    "derived_aggregate": ["n_people", "n_vehicles", "n_conductores",
                          "n_pasajeros", "n_peatones",
                          "has_bicycle", "has_motorcycle",
                          "has_heavy_vehicle", "has_pedestrian"],
    "leakage_alcohol_drug": ["any_alcohol_positive", "n_alcohol_positive",
                             "any_drug_positive"],
    "target_raw": ["cod_lesividad_max", "n_fatalities", "n_serious", "n_minor"],
    "target": ["severity", "severity_code"],
}


def columns_with_role(role: str) -> list[str]:
    """Return the column list for a given role."""
    return list(ACCIDENT_COLUMN_ROLES.get(role, []))


SPLIT_YEARS: dict[str, tuple[int, int]] = {
    "train": (2019, 2022),
    "val": (2023, 2023),
    "test": (2024, 2025),
}


def year_to_split(year: int) -> str:
    """Map an accident year to its split. Raises on out-of-range years."""
    for name, (lo, hi) in SPLIT_YEARS.items():
        if lo <= year <= hi:
            return name
    raise ValueError(f"Year {year} does not belong to any split")