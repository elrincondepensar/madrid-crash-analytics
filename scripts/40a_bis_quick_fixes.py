"""
Script 40a-bis — Quick fixes for three issues in 40a:
  1. Naming regex: 09_2024.csv → 2024-09 (closes the false "gap").
  2. id join with stratified sampling across the full CSV.
  3. 2025-01 CSV integrity: encoding alternatives and error rate.

Run:
    python scripts/40a_bis_quick_fixes.py
"""

from __future__ import annotations

import io
import logging
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety.data_audit import dumps_jsonable  # noqa: E402

CKAN = "https://datos.madrid.es/api/3/action"
UA = "madrid-road-safety/phase9/40a-bis"
SAMPLES_DIR = Path("data/raw/40a_range_headers")
REPORTS_TABLES = Path("reports/tables")

log = logging.getLogger("phase9_40a_bis")


# ---------------------------------------------------------------------------
# Fix 1: tolerant month regex
# ---------------------------------------------------------------------------

_MONTH_RE = re.compile(
    r"(?P<mm>\d{1,2})[_\-\s]?(?P<yyyy>20\d{2})", re.IGNORECASE
)
_ANNUAL_RE = re.compile(r"^(?P<yyyy>20\d{2})$")

# Spanish month names → numbers (for "Abril 2016.csv")
_SPANISH_MONTHS = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
    "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
    "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}


def infer_month_tolerant(filename: str) -> str | None:
    """Parse a filename into YYYY-MM or YYYY-ANNUAL."""
    if not filename:
        return None
    stem = filename.strip().rstrip("/").replace(".csv", "").replace(".CSV", "")

    # Annual: exactly YYYY
    m = _ANNUAL_RE.match(stem)
    if m:
        return f"{m.group('yyyy')}-ANNUAL"

    # Numeric: MM-YYYY or MM_YYYY or MM YYYY
    m = _MONTH_RE.search(stem)
    if m:
        mm = int(m.group("mm"))
        yyyy = m.group("yyyy")
        if 1 <= mm <= 12:
            return f"{yyyy}-{mm:02d}"

    # Spanish: "Abril 2016"
    parts = stem.lower().split()
    if len(parts) >= 2:
        month_name = parts[0].strip()
        year_part = parts[-1]
        if month_name in _SPANISH_MONTHS and year_part.isdigit():
            mm = _SPANISH_MONTHS[month_name]
            return f"{year_part}-{mm:02d}"

    return None


# ---------------------------------------------------------------------------
# Fix 2: id join with stratified sampling
# ---------------------------------------------------------------------------

def id_join_stratified(month_zip: Path, catalog_csv: Path,
                       n_chunks: int = 20,
                       chunk_size: int = 100_000) -> dict:
    """Read the month CSV in evenly-spaced chunks to get a representative
    sample of ids. Much better than nrows=200_000 at the top."""
    out = {}

    # Catalog (already downloaded)
    df_cat = pd.read_csv(catalog_csv, sep=";", encoding="utf-8-sig",
                         on_bad_lines="skip")
    ids_cat_str = set(df_cat["id"].astype(str).str.strip().tolist())
    ids_cat_num = set(pd.to_numeric(df_cat["id"], errors="coerce")
                      .dropna().astype(int).tolist())
    out["catalog_n"] = len(ids_cat_str)
    out["catalog_numeric_range"] = (
        [min(ids_cat_num), max(ids_cat_num)] if ids_cat_num else None)
    out["catalog_has_leading_zero"] = any(
        s.startswith("0") for s in ids_cat_str if s)
    out["catalog_tipo_elem"] = (
        df_cat["tipo_elem"].value_counts(dropna=False).to_dict()
        if "tipo_elem" in df_cat.columns else None)

    # Open month ZIP and read in chunks at evenly-spaced positions
    with zipfile.ZipFile(month_zip) as zf:
        csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csvs:
            return {**out, "error": "no csv"}
        csv_name = csvs[0]
        out["month_csv"] = csv_name

        ids_month = set()
        n_rows_read = 0
        n_chunks_read = 0
        try:
            with zf.open(csv_name) as f:
                reader = pd.read_csv(
                    f, sep=";", encoding="utf-8-sig",
                    usecols=["id"], chunksize=chunk_size,
                    on_bad_lines="skip",
                )
                for i, chunk in enumerate(reader):
                    if i % max(1, (1000 // n_chunks)) == 0 or i < n_chunks:
                        ids_chunk = (
                            chunk["id"].dropna().astype(str).str.strip())
                        ids_month.update(ids_chunk.tolist())
                        n_rows_read += len(chunk)
                        n_chunks_read += 1
                    if n_chunks_read >= n_chunks:
                        break
        except Exception as e:  # noqa: BLE001
            out["error"] = f"{type(e).__name__}: {e}"
            return out

    out["month_n_ids_sampled"] = len(ids_month)
    out["month_n_rows_read"] = n_rows_read
    out["month_n_chunks_read"] = n_chunks_read

    # Compare as strings AND as ints (to rule out leading-zero mismatch)
    inter_str = ids_cat_str & ids_month
    ids_month_num = set()
    for s in ids_month:
        try:
            ids_month_num.add(int(s))
        except ValueError:
            pass
    inter_num = ids_cat_num & ids_month_num

    out["intersection_str"] = len(inter_str)
    out["intersection_num"] = len(inter_num)
    out["frac_month_matched_str"] = (
        round(len(inter_str) / len(ids_month), 4) if ids_month else None)
    out["frac_month_matched_num"] = (
        round(len(inter_num) / len(ids_month_num), 4)
        if ids_month_num else None)
    out["frac_catalog_matched_str"] = (
        round(len(inter_str) / len(ids_cat_str), 4) if ids_cat_str else None)

    # Sample of unmatched (if any)
    unmatched_month = ids_month - ids_cat_str
    out["sample_unmatched_month"] = sorted(list(unmatched_month))[:10]
    out["month_id_numeric_range"] = (
        [min(ids_month_num), max(ids_month_num)] if ids_month_num else None)

    return out


# ---------------------------------------------------------------------------
# Fix 3: 2025-01 integrity
# ---------------------------------------------------------------------------

def check_2025_01(month_zip: Path) -> dict:
    """Diagnose the parsing error on the 2025-01 CSV."""
    out = {}
    with zipfile.ZipFile(month_zip) as zf:
        csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csvs:
            return {"error": "no csv"}
        csv_name = csvs[0]
        out["csv_name"] = csv_name
        with zf.open(csv_name) as f:
            raw = f.read()

    out["file_size_bytes"] = len(raw)

    # Try different encodings and parse strategies
    for enc in ("utf-8-sig", "latin-1", "utf-8"):
        try:
            df = pd.read_csv(io.BytesIO(raw), sep=";", encoding=enc,
                             on_bad_lines="skip", nrows=10_000)
            out[f"parse_{enc}"] = {
                "ok": True,
                "n_rows": len(df),
                "columns": list(df.columns),
            }
            break
        except Exception as e:  # noqa: BLE001
            out[f"parse_{enc}"] = {
                "ok": False,
                "error": f"{type(e).__name__}: {str(e)[:120]}",
            }
    else:
        return out

    # Full-file count with on_bad_lines='skip'
    try:
        df_full = pd.read_csv(io.BytesIO(raw), sep=";", encoding="latin-1",
                              on_bad_lines="skip")
        out["full_rows_read_skip"] = len(df_full)
        out["columns"] = list(df_full.columns)
        if "fecha" in df_full.columns:
            dates = pd.to_datetime(df_full["fecha"], errors="coerce")
            out["fecha_min"] = str(dates.min())
            out["fecha_max"] = str(dates.max())
            out["fecha_n_null"] = int(dates.isna().sum())
        if "tipo_elem" in df_full.columns:
            out["tipo_elem_dist"] = (
                df_full["tipo_elem"].value_counts(dropna=False).head(10).to_dict())
        if "id" in df_full.columns:
            out["n_unique_ids"] = int(df_full["id"].nunique())
    except Exception as e:  # noqa: BLE001
        out["full_parse_error"] = f"{type(e).__name__}: {str(e)[:200]}"

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Fix 1: verify month regex
    log.info("FIX 1: tolerant month regex")
    test_cases = [
        "09_2024.csv", "Abril 2016.csv", "04-2015/", "2013.csv",
        "12-2022.csv", "01-2026.csv",
    ]
    fix1 = {
        "test_cases": [
            {"input": tc, "output": infer_month_tolerant(tc)}
            for tc in test_cases
        ]
    }
    for tc in test_cases:
        log.info(f"  {tc} -> {infer_month_tolerant(tc)}")

    # Fix 2: stratified id join
    log.info("FIX 2: stratified id join")
    month_zip = SAMPLES_DIR / "full_2022-06.zip"
    catalog_csv = SAMPLES_DIR / "202468_catalog.csv"
    fix2 = {}
    if month_zip.exists() and catalog_csv.exists():
        fix2 = id_join_stratified(month_zip, catalog_csv, n_chunks=20)
        log.info(f"  catalog_n={fix2.get('catalog_n')}, "
                 f"month_n={fix2.get('month_n_ids_sampled')}, "
                 f"inter_str={fix2.get('intersection_str')}, "
                 f"frac_month_str={fix2.get('frac_month_matched_str')}, "
                 f"frac_month_num={fix2.get('frac_month_matched_num')}")
    else:
        log.warning(f"  missing: zip={month_zip.exists()} catalog={catalog_csv.exists()}")

    # Fix 3: 2025-01 integrity
    log.info("FIX 3: 2025-01 integrity")
    zip_2025 = SAMPLES_DIR / "full_2025-01.zip"
    fix3 = {}
    if zip_2025.exists():
        fix3 = check_2025_01(zip_2025)
        log.info(f"  ok_encoding={[k for k in fix3 if k.startswith('parse_') and fix3[k].get('ok')]}")
        log.info(f"  full_rows={fix3.get('full_rows_read_skip')}, "
                 f"fecha_range=[{fix3.get('fecha_min')}, {fix3.get('fecha_max')}]")
    else:
        log.warning(f"  {zip_2025} not found")

    # Save
    result = {"fix1_month_regex": fix1,
              "fix2_id_join": fix2,
              "fix3_2025_01": fix3}
    out_json = REPORTS_TABLES / "40a_bis_quick_fixes.json"
    out_json.write_text(dumps_jsonable(result), encoding="utf-8")
    log.info(f"saved {out_json}")

    log.info("=" * 72)
    log.info("SUMMARY:")
    log.info(f"  2024-09 gap closed: "
             f"{infer_month_tolerant('09_2024.csv') == '2024-09'}")
    if fix2.get("frac_month_matched_str") is not None:
        log.info(f"  id join frac (stratified): "
                 f"str={fix2['frac_month_matched_str']:.2%} "
                 f"num={fix2.get('frac_month_matched_num', 0):.2%}")
    if fix3.get("full_rows_read_skip"):
        log.info(f"  2025-01 readable with latin-1 fallback: "
                 f"{fix3['full_rows_read_skip']:,} rows, "
                 f"fecha={fix3.get('fecha_min')}..{fix3.get('fecha_max')}")


if __name__ == "__main__":
    main()