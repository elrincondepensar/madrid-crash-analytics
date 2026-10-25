"""
Phase 2 · Iteration 2 — Build processed tables:
    data/processed/persons_clean.parquet   (one row per person)
    data/processed/accidents_clean.parquet (one row per accident)

Run from the repository root:
    python scripts/06_phase2_build_clean.py

Requires: pyarrow (for parquet). If missing:
    pip install pyarrow
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety.accidents import load_all_accidents
from madrid_safety.config import ACCIDENT_COLUMN_ROLES  # noqa: F401  (documented)
from madrid_safety.data_audit import dumps_jsonable, save_json, save_markdown_report

OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

# cod_lesividad -> severity_code (ordinal 0..3). Note the codification is
# NOT ordinal: 14 = no assistance (worst-informative value, lowest severity),
# 77 = unknown (ignored in worst-victim derivation).
LESIVIDAD_TO_SEVERITY_CODE = {
    4:  3,  # Fallecido 24h                     -> fatal
    3:  2,  # Ingreso superior a 24h            -> serious
    2:  2,  # Ingreso inferior o igual a 24h    -> serious
    1:  1,  # Atención en urgencias sin ingreso -> minor
    5:  1,  # Asistencia ambulatoria posterior  -> minor
    6:  1,  # Asistencia sanitaria inmediata    -> minor
    7:  1,  # Asistencia sanitaria en el lugar  -> minor
    14: 0,  # Sin asistencia sanitaria          -> no injury
}
SEVERITY_LABELS = {0: "no_injury", 1: "minor", 2: "serious", 3: "fatal"}
SEVERITY_CODE = {"no_injury": 0, "minor": 1, "serious": 2, "fatal": 3}


def _worst_lesividad(cods: pd.Series) -> pd.Series:
    """
    Worst-victim-wins: return (worst_severity_code, worst_lesividad_code).

    cod_lesividad 77 ("Se desconoce") is excluded; if only 77/NaN are present,
    the accident is classified as no_injury (worst_severity_code = 0).
    Tie-break: prefer the lowest cod among those with the same severity, which
    favors the more specific clinical code.
    """
    vals = [int(v) for v in cods.dropna().tolist()]
    known = [v for v in vals if v != 77]
    if not known:
        return pd.Series({"worst_severity_code": 0,
                          "worst_lesividad_code": -1})
    best = max(known, key=lambda c: (LESIVIDAD_TO_SEVERITY_CODE.get(c, 0), -c))
    return pd.Series({
        "worst_severity_code": LESIVIDAD_TO_SEVERITY_CODE.get(best, 0),
        "worst_lesividad_code": best,
    })


def build_accidents(p: pd.DataFrame) -> pd.DataFrame:
    """One row per accident with aggregate features and the target."""
    agg = (p.groupby(["year", "num_expediente"], sort=False)
             .agg(
                 fecha=("fecha", "first"),
                 hora=("hora", "first"),
                 hour=("hour", "first"),
                 localizacion=("localizacion", "first"),
                 numero=("numero", "first"),
                 cod_distrito=("cod_distrito", "first"),
                 distrito=("distrito", "first"),
                 coordenada_x_utm=("coordenada_x_utm", "first"),
                 coordenada_y_utm=("coordenada_y_utm", "first"),
                 tipo_accidente=("tipo_accidente", "first"),
                 estado_meteorologico=("estado_meteorologico", "first"),
                 n_people=("num_expediente", "size"),
                 n_vehicles=("tipo_vehiculo", "nunique"),
                 n_conductores=("is_driver", "sum"),
                 n_pasajeros=("is_passenger", "sum"),
                 n_peatones=("is_pedestrian", "sum"),
                 has_bicycle=("is_bicycle", "any"),
                 has_motorcycle=("is_motorcycle", "any"),
                 has_heavy_vehicle=("is_heavy_vehicle", "any"),
                 has_pedestrian=("is_pedestrian", "any"),
                 any_alcohol_positive=("alcohol_positive", "any"),
                 n_alcohol_positive=("alcohol_positive", "sum"),
                 any_drug_positive=("drug_positive", "any"),
                 n_fatalities=("is_fatal", "sum"),
                 n_serious=("is_serious", "sum"),
                 n_minor=("is_minor", "sum"),
             )
             .reset_index())

    # Worst-victim severity (correct, non-ordinal mapping)
    worst = (p.groupby(["year", "num_expediente"])["cod_lesividad"]
               .apply(_worst_lesividad)
               .unstack())
    agg = agg.merge(worst, on=["year", "num_expediente"], how="left")

    agg["severity"] = agg["worst_severity_code"].map(SEVERITY_LABELS)
    agg["severity_code"] = agg["worst_severity_code"].astype("int8")
    agg = agg.drop(columns=["worst_severity_code"])

    order = [
        "num_expediente", "year",
        "fecha", "hora", "hour",
        "localizacion", "numero", "cod_distrito", "distrito",
        "coordenada_x_utm", "coordenada_y_utm",
        "tipo_accidente", "estado_meteorologico",
        "n_people", "n_vehicles", "n_conductores", "n_pasajeros", "n_peatones",
        "has_bicycle", "has_motorcycle", "has_heavy_vehicle", "has_pedestrian",
        "any_alcohol_positive", "n_alcohol_positive", "any_drug_positive",
        "n_fatalities", "n_serious", "n_minor",
        "worst_lesividad_code", "severity", "severity_code",
    ]
    return agg[order]


# ---------------------------------------------------------------------------
# Persons-level cleaning
# ---------------------------------------------------------------------------
def build_persons(df: pd.DataFrame) -> pd.DataFrame:
    """One row per person with derived flags. Keeps leaky columns for audit."""
    p = df.copy()
    # hour
    h = pd.to_datetime(p["hora"], format="%H:%M:%S", errors="coerce").dt.hour
    mask = h.isna()
    if mask.any():
        h.loc[mask] = pd.to_datetime(
            p.loc[mask, "hora"], format="%H:%M", errors="coerce"
        ).dt.hour
    p["hour"] = h.astype("Int8")

    # person type flags
    p["is_driver"] = p["tipo_persona"] == "Conductor"
    p["is_passenger"] = p["tipo_persona"] == "Pasajero"
    p["is_pedestrian"] = p["tipo_persona"].str.startswith("Peatón", na=False)

    # vehicle type flags
    p["is_bicycle"] = p["tipo_vehiculo"].isin(
        ["Bicicleta", "Bicicleta EPAC (pedaleo asistido)"])
    p["is_motorcycle"] = p["tipo_vehiculo"].str.contains(
        "Motocicleta|Ciclomotor", na=False)
    p["is_heavy_vehicle"] = p["tipo_vehiculo"].str.contains(
        "Autobús|Camión|Tractocamión|Maquinaria", na=False)

    # injury flags (raw and per class)
    p["is_fatal"] = (p["cod_lesividad"] == 4)
    p["is_serious"] = p["cod_lesividad"].isin([2, 3])
    p["is_minor"] = p["cod_lesividad"].isin([1, 5, 6, 7])
    p["has_known_lesividad"] = p["cod_lesividad"].notna()

    # alcohol / drug flags (leaky: for descriptive only)
    p["alcohol_positive"] = p["positiva_alcohol"] == "S"
    p["drug_positive"] = p["positiva_droga"] == 1.0

    return p

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate(acc: pd.DataFrame, per: pd.DataFrame) -> dict:
    # Consistency checks that should always be true
    fatal_iff = ((acc["severity"] == "fatal") == (acc["n_fatalities"] >= 1))
    no_injury_matches = (
        (acc["severity"] == "no_injury")
        == (acc["worst_lesividad_code"].isin([-1, 14]))
    )
    return {
        "n_accidents": int(len(acc)),
        "n_persons": int(len(per)),
        "n_duplicate_expedientes": int(acc["num_expediente"].duplicated().sum()),
        "n_rows_year_outside_range": int((~acc["year"].between(2019, 2025)).sum()),
        "n_missing_coords": int(acc["coordenada_x_utm"].isna().sum()),
        "severity_counts": acc["severity"].value_counts().to_dict(),
        "severity_share": (acc["severity"].value_counts(normalize=True)
                           .round(4).to_dict()),
        "severity_counts_by_year": (
            acc.groupby(["year", "severity"]).size()
               .unstack(fill_value=0).to_dict(orient="index")),
        "total_fatalities": int(acc["n_fatalities"].sum()),
        "sum_n_people_equals_n_persons": bool(
            int(acc["n_people"].sum()) == int(len(per))),
        "severity_fatal_iff_n_fatalities_ge_1": bool(fatal_iff.all()),
        "severity_fatal_violations": int((~fatal_iff).sum()),
        "severity_no_injury_matches_worst_code": bool(no_injury_matches.all()),
        "severity_no_injury_violations": int((~no_injury_matches).sum()),
        "worst_lesividad_code_counts": (
            acc["worst_lesividad_code"].value_counts().to_dict()),
        "column_dtypes": {c: str(d) for c, d in acc.dtypes.items()},
        "persons_column_dtypes": {c: str(d) for c, d in per.dtypes.items()},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading accidents (from cache)...")
    df = load_all_accidents()
    print(f"  {len(df):,} person-rows, {df['num_expediente'].nunique():,} accidents.")

    print("Building persons_clean.parquet...")
    persons = build_persons(df)
    persons.to_parquet(OUT / "persons_clean.parquet", index=False)
    print(f"  -> {OUT / 'persons_clean.parquet'} ({len(persons):,} rows, "
          f"{persons.shape[1]} columns).")

    print("Building accidents_clean.parquet...")
    acc = build_accidents(persons)
    acc.to_parquet(OUT / "accidents_clean.parquet", index=False)
    print(f"  -> {OUT / 'accidents_clean.parquet'} ({len(acc):,} rows, "
          f"{acc.shape[1]} columns).")

    print("Validating...")
    checks = validate(acc, persons)
    save_json(checks, OUT / "accidents_clean_validation.json")

    sections = {
        "Validation": dumps_jsonable(
            {k: v for k, v in checks.items()
             if k not in ("column_dtypes", "persons_column_dtypes")},
            indent=2),
        "Accident columns and dtypes": dumps_jsonable(
            checks["column_dtypes"], indent=2),
        "Person columns and dtypes": dumps_jsonable(
            checks["persons_column_dtypes"], indent=2),
    }
    save_markdown_report(sections, Path("reports/06_build_clean.md"))
    print("Done. See reports/06_build_clean.md.")


if __name__ == "__main__":
    main()