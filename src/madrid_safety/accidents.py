"""Load and harmonize Madrid traffic-accident detail files (2019-2025)."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

BASE = "https://datos.madrid.es/dataset/300228-0-accidentes-trafico-detalle/resource"
CACHE_DIR = Path("data/raw")
UA = "madrid-road-safety/0.2"

ACCIDENTS_RESOURCES: dict[int, tuple[str, str]] = {
    2019: ("300228-11-accidentes-trafico-detalle-csv",
           "300228-11-accidentes-trafico-detalle-csv.csv"),
    2020: ("300228-8-accidentes-trafico-detalle-csv",
           "300228-8-accidentes-trafico-detalle-csv.csv"),
    2021: ("300228-33-accidentes-trafico-detalle-csv",
           "300228-33-accidentes-trafico-detalle-csv.csv"),
    2022: ("300228-5-accidentes-trafico-detalle-csv",
           "300228-5-accidentes-trafico-detalle-csv.csv"),
    2023: ("300228-3-accidentes-trafico-detalle-csv",
           "300228-3-accidentes-trafico-detalle-csv.csv"),
    2024: ("300228-2-accidentes-trafico-detalle-csv",
           "300228-2-accidentes-trafico-detalle-csv.csv"),
    2025: ("300228-1-accidentes-trafico-detalle-csv",
           "accidentes_trafico_policia_2025.csv"),
}

COLUMN_ALIASES = {
    "estado_meteorolÃ³gico": "estado_meteorologico",
    "estado_meteorológico": "estado_meteorologico",
    "tipo_vehÃ­culo": "tipo_vehiculo",
}

NUMERIC_COLUMNS = (
    "numero", "cod_distrito", "cod_lesividad",
    "coordenada_x_utm", "coordenada_y_utm", "positiva_droga",
)

# From the cod_lesividad x lesividad crosstab (Phase 2 diagnostic).
# cod_lesividad 77 = "Se desconoce" is excluded from severity derivation.
LESIVIDAD_SEVERITY = {
    4:  3,  # Fallecido 24 horas                        → fatal
    3:  2,  # Ingreso superior a 24 horas                → serious
    2:  2,  # Ingreso inferior o igual a 24 horas        → serious
    1:  1,  # Atención en urgencias sin posterior ingreso→ minor
    5:  1,  # Asistencia sanitaria ambulatoria           → minor
    6:  1,  # Asistencia sanitaria inmediata             → minor
    7:  1,  # Asistencia sanitaria sólo en el lugar      → minor
    14: 0,  # Sin asistencia sanitaria                   → no injury
}
SEVERITY_LABELS = {0: "no_injury", 1: "minor", 2: "serious", 3: "fatal",
                   -1: "unknown"}


def resource_url(year: int) -> str:
    res, fname = ACCIDENTS_RESOURCES[year]
    return f"{BASE}/{res}/download/{fname}"


def _cache_path(url: str) -> Path:
    h = hashlib.sha256(url.encode()).hexdigest()[:16]
    name = url.rsplit("/", 1)[-1]
    return CACHE_DIR / f"{h}_{name}"


def read_madrid_csv_cached(url: str) -> pd.DataFrame:
    """Download once and cache in data/raw/; subsequent runs read from disk."""
    path = _cache_path(url)
    if not path.exists():
        r = requests.get(url, headers={"User-Agent": UA}, timeout=120)
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
    content = path.read_bytes()
    last_exc = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(content), sep=";", encoding=enc,
                               low_memory=False)
        except UnicodeDecodeError as exc:
            last_exc = exc
    raise last_exc  # type: ignore[misc]


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.lstrip("\ufeff").strip() for c in df.columns]
    df = df.rename(columns={k: v for k, v in COLUMN_ALIASES.items()
                            if k in df.columns})
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_accident_year(year: int) -> pd.DataFrame:
    """Download, normalize and coerce one year of accidents (one row per person)."""
    if year not in ACCIDENTS_RESOURCES:
        raise KeyError(f"No accident resource registered for {year}")
    df = read_madrid_csv_cached(resource_url(year))
    df = _normalize(df)
    df["file_year"] = year
    # Year of the accident itself (from fecha) — preferred for temporal splits.
    df["fecha"] = pd.to_datetime(df["fecha"], format="%d/%m/%Y", errors="coerce")
    df["year"] = df["fecha"].dt.year.astype("Int64")
    return df


def load_all_accidents(years: Optional[list[int]] = None,
                       restrict_to_fecha_range: bool = True) -> pd.DataFrame:
    """
    Concatenate 2019-2025 accidents.

    If restrict_to_fecha_range, drops rows whose fecha.year is outside 2019-2025
    (a handful of records that spill across year boundaries).
    """
    years = years or sorted(ACCIDENTS_RESOURCES)
    df = pd.concat([load_accident_year(y) for y in years],
                   ignore_index=True, sort=False)
    if restrict_to_fecha_range:
        mask = df["fecha"].notna() & df["year"].between(2019, 2025)
        df = df.loc[mask].copy()
    return df


def derive_severity(df: pd.DataFrame) -> pd.DataFrame:
    """
    One severity per accident using the "worst victim wins" convention.

    The mapping (LESIVIDAD_SEVERITY) is derived from the crosstab of
    cod_lesividad x lesividad, not assumed. See decisions D021.
    """
    def worst(cods: pd.Series) -> int:
        vals = cods.dropna().astype(int).tolist()
        known = [v for v in vals if v != 77]
        if not known:
            return -1 if vals else 0  # all unknown vs all NaN (no injury)
        return max(LESIVIDAD_SEVERITY.get(v, 0) for v in known)

    sev = (df.groupby(["year", "num_expediente"])["cod_lesividad"]
             .apply(worst)
             .rename("severity_code")
             .reset_index())
    sev["severity"] = sev["severity_code"].map(
        lambda c: SEVERITY_LABELS.get(int(c), "unknown")
    )
    return sev