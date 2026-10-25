"""
Script 39 — Phase 9 iter 1: Audit of exposure data sources.

Answers the question: what traffic exposure can we build, at what
spatial/temporal granularity, from the available sources?

Four sources:
  1. 300233 — Permanent count stations (aforos). 60 stations, monthly CSVs.
  2. 208627 — Historical traffic. 162 ZIPs, 15-min granularity. THE
     critical question: is `tipo_elem` M30-only, or city-wide?
  3. 202468 — Measurement point locations. 307 resources, extended inventory.
  4. 203962 — IMD annual by segment. Static 2012-2013 (already characterized).

Outputs:
  - data/raw/39_samples/*.{csv,zip}    (samples for inspection)
  - reports/tables/39_exposure_audit.json
  - reports/39_exposure_audit.md
"""

from __future__ import annotations

import io
import logging
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety.data_audit import dumps_jsonable  # noqa: E402


CKAN = "https://datos.madrid.es/api/3/action"
TIMEOUT = 60
UA = "madrid-road-safety/phase9/0.1"

SAMPLES_DIR = Path("data/raw/39_samples")
REPORTS_TABLES = Path("reports/tables")
REPORTS = Path("reports")

SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_TABLES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("phase9_exposure")


# ---------------------------------------------------------------------------
# CKAN helpers
# ---------------------------------------------------------------------------

def ckan_package_show(dataset_id: str) -> dict | None:
    try:
        r = requests.get(f"{CKAN}/package_show",
                         params={"id": dataset_id},
                         headers={"User-Agent": UA}, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        if not data.get("success"):
            return None
        return data["result"]
    except Exception as e:  # noqa: BLE001
        log.warning(f"  ckan_package_show({dataset_id}) failed: {e}")
        return None


def head_size(url: str) -> int | None:
    try:
        r = requests.head(url, headers={"User-Agent": UA},
                          timeout=20, allow_redirects=True)
        s = r.headers.get("Content-Length")
        return int(s) if s else None
    except Exception:  # noqa: BLE001
        return None


def download_sample(url: str, dest: Path,
                    max_size_mb: int = 200,
                    timeout: int = 180) -> bytes | None:
    if dest.exists() and dest.stat().st_size > 0:
        log.info(f"    cached: {dest.name}")
        return dest.read_bytes()

    size = head_size(url)
    if size and size > max_size_mb * 1024 * 1024:
        log.warning(f"    skip: size {size / 1e6:.1f} MB > {max_size_mb} MB")
        return None

    try:
        r = requests.get(url, headers={"User-Agent": UA},
                         timeout=timeout, stream=True)
        r.raise_for_status()
        content = r.content
        dest.write_bytes(content)
        log.info(f"    downloaded {len(content) / 1e6:.1f} MB -> {dest.name}")
        return content
    except Exception as e:  # noqa: BLE001
        log.warning(f"    download failed: {e}")
        return None


def read_csv_robust(content: bytes, max_rows: int = 5000) -> tuple:
    """Try multiple encodings and separators."""
    for enc in ("utf-8-sig", "latin-1", "utf-8"):
        for sep in (";", ",", "\t", "|"):
            try:
                df = pd.read_csv(io.BytesIO(content),
                                 sep=sep, encoding=enc,
                                 nrows=max_rows, on_bad_lines="skip")
                if df.shape[1] >= 2:
                    return df, enc, sep
            except Exception:  # noqa: BLE001
                continue
    return None, None, None


# ---------------------------------------------------------------------------
# 1. 300233 — Permanent count stations
# ---------------------------------------------------------------------------

def audit_300233() -> dict:
    log.info("[300233] Auditing permanent count stations")
    result = {"dataset_id": "300233-0-aforo-trafico-permanentes"}

    pkg = ckan_package_show(result["dataset_id"])
    if not pkg:
        return {**result, "error": "CKAN package_show failed"}

    resources = pkg.get("resources", [])
    result["n_resources"] = len(resources)
    result["resource_formats"] = sorted(set(
        r.get("format") for r in resources if r.get("format")
    ))
    log.info(f"  {len(resources)} resources")

    # Metadata files (110: 60 stations; 112: 120 rows)
    metadata = {}
    for suffix in ("110", "112"):
        matches = [r for r in resources
                   if suffix in (r.get("name") or "")]
        if not matches:
            continue
        url = matches[0].get("url")
        if not url:
            continue
        content = download_sample(url, SAMPLES_DIR / f"300233_{suffix}.csv",
                                  max_size_mb=10)
        if not content:
            continue
        df, enc, sep = read_csv_robust(content, max_rows=200)
        if df is None:
            continue
        metadata[suffix] = {
            "n_rows_sampled": len(df),
            "columns": list(df.columns),
            "encoding": enc, "sep": sep,
            "sample_rows": df.head(3).to_dict(orient="records"),
        }
    result["metadata_files"] = metadata

    # Sample one monthly file
    monthly_candidates = [
        r for r in resources
        if (r.get("format") or "").upper() == "CSV"
        and "110" not in (r.get("name") or "")
        and "112" not in (r.get("name") or "")
    ]
    if monthly_candidates:
        r = monthly_candidates[0]
        url = r.get("url")
        name = r.get("name", "monthly")
        log.info(f"  sampling monthly: {name}")
        content = download_sample(url, SAMPLES_DIR / f"300233_monthly.csv",
                                  max_size_mb=30)
        if content:
            df, enc, sep = read_csv_robust(content, max_rows=1000)
            if df is not None:
                sample = {
                    "resource_name": name,
                    "n_rows_sampled": len(df),
                    "columns": list(df.columns),
                    "encoding": enc, "sep": sep,
                    "sample_rows": df.head(2).to_dict(orient="records"),
                }
                if "FSEN" in df.columns:
                    sample["fsen_values"] = (
                        df["FSEN"].value_counts(dropna=False).head(10)
                        .to_dict()
                    )
                if "FDIA" in df.columns:
                    sample["n_unique_dates"] = int(df["FDIA"].nunique())
                if "FEST" in df.columns:
                    sample["n_unique_stations"] = int(
                        df["FEST"].nunique(dropna=True))
                result["monthly_sample"] = sample

    return result


# ---------------------------------------------------------------------------
# 2. 208627 — Historical traffic (CRITICAL)
# ---------------------------------------------------------------------------

def audit_208627() -> dict:
    log.info("[208627] Auditing historical traffic ZIPs (critical)")
    result = {"dataset_id": "208627-0-transporte-ptomedida-historico"}

    pkg = ckan_package_show(result["dataset_id"])
    if not pkg:
        return {**result, "error": "CKAN package_show failed"}

    resources = pkg.get("resources", [])
    result["n_resources"] = len(resources)

    formats = {}
    for r in resources:
        f = r.get("format") or "unknown"
        formats[f] = formats.get(f, 0) + 1
    result["resource_formats"] = formats
    result["first_10_resource_names"] = [
        r.get("name") for r in resources[:10]
    ]

    # Sample 3 ZIPs (index 0, mid, last) for representative coverage
    if len(resources) >= 3:
        idxs = sorted(set([0, len(resources) // 2, len(resources) - 1]))
    else:
        idxs = list(range(len(resources)))

    samples = []
    for i, idx in enumerate(idxs):
        r = resources[idx]
        if (r.get("format") or "").upper() != "ZIP":
            continue
        url = r.get("url")
        if not url:
            continue
        log.info(f"  sampling ZIP {i+1}/{len(idxs)}: {r.get('name')}")
        dest = SAMPLES_DIR / f"208627_sample_{i+1}.zip"
        content = download_sample(url, dest, max_size_mb=300, timeout=240)
        if not content:
            samples.append({"resource_name": r.get("name"),
                            "url": url, "error": "download_failed"})
            continue

        try:
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                csv_files = [n for n in zf.namelist()
                             if n.lower().endswith(".csv")]
                if not csv_files:
                    samples.append({
                        "resource_name": r.get("name"), "url": url,
                        "error": "no_csv_inside",
                        "zip_contents": zf.namelist()[:5],
                    })
                    continue
                csv_name = csv_files[0]
                with zf.open(csv_name) as f:
                    csv_content = f.read()
                df, enc, sep = read_csv_robust(csv_content, max_rows=5000)
                if df is None:
                    samples.append({
                        "resource_name": r.get("name"), "url": url,
                        "error": "csv_parse_failed", "csv_name": csv_name,
                    })
                    continue

                sample = {
                    "resource_name": r.get("name"), "url": url,
                    "csv_name": csv_name,
                    "n_rows_sampled": len(df),
                    "columns": list(df.columns),
                    "encoding": enc, "sep": sep,
                    "sample_rows": df.head(2).to_dict(orient="records"),
                }

                if "tipo_elem" in df.columns:
                    tipo_counts = df["tipo_elem"].value_counts(dropna=False)
                    sample["tipo_elem_distribution_top20"] = {
                        str(k): int(v) for k, v in tipo_counts.head(20).items()
                    }
                    sample["n_unique_tipo_elem"] = int(
                        df["tipo_elem"].nunique(dropna=True))
                if "id" in df.columns:
                    sample["n_unique_ids"] = int(
                        df["id"].nunique(dropna=True))
                    if "tipo_elem" in df.columns:
                        # Cross-tab: id prefix × tipo_elem
                        df["_id_prefix"] = (
                            df["id"].astype(str).str[:4])
                        cross = pd.crosstab(
                            df["_id_prefix"], df["tipo_elem"])
                        sample["id_prefix_vs_tipo_elem"] = (
                            cross.head(10).to_dict()
                        )
                if "fecha" in df.columns:
                    try:
                        dates = pd.to_datetime(df["fecha"],
                                               errors="coerce")
                        if dates.notna().any():
                            sample["date_range"] = {
                                "min": str(dates.min()),
                                "max": str(dates.max()),
                            }
                    except Exception:  # noqa: BLE001
                        pass
                samples.append(sample)
        except Exception as e:  # noqa: BLE001
            samples.append({"resource_name": r.get("name"),
                            "url": url,
                            "error": f"zip_parse_failed: {type(e).__name__}: {e}"})

    result["samples"] = samples
    return result


# ---------------------------------------------------------------------------
# 3. 202468 — Measurement point locations
# ---------------------------------------------------------------------------

def audit_202468() -> dict:
    log.info("[202468] Auditing measurement points")
    result = {"dataset_id": "202468-0-ubicacion-puntos-medida"}

    pkg = ckan_package_show(result["dataset_id"])
    if not pkg:
        return {**result, "error": "CKAN package_show failed"}

    resources = pkg.get("resources", [])
    result["n_resources"] = len(resources)

    formats = {}
    for r in resources:
        f = r.get("format") or "unknown"
        formats[f] = formats.get(f, 0) + 1
    result["resource_formats"] = formats

    csvs = [r for r in resources
            if (r.get("format") or "").upper() == "CSV"]
    if not csvs:
        return {**result, "error": "no CSV resources"}

    r = csvs[0]
    url = r.get("url")
    log.info(f"  sampling CSV: {r.get('name')}")
    content = download_sample(url, SAMPLES_DIR / "202468_sample.csv",
                              max_size_mb=20)
    if not content:
        return {**result, "error": "download_failed"}

    df, enc, sep = read_csv_robust(content, max_rows=500)
    if df is None:
        result["error"] = "csv_parse_failed"
        result["head_bytes"] = content[:500].decode("latin-1",
                                                       errors="replace")
        return result

    sample = {
        "resource_name": r.get("name"),
        "n_rows_sampled": len(df),
        "columns": list(df.columns),
        "encoding": enc, "sep": sep,
        "sample_rows": df.head(3).to_dict(orient="records"),
    }
    if "tipo_elem" in df.columns:
        sample["tipo_elem_values"] = (
            df["tipo_elem"].value_counts(dropna=False).head(15).to_dict()
        )
    result["sample"] = sample
    return result


# ---------------------------------------------------------------------------
# 4. 203962 — IMD (already characterized)
# ---------------------------------------------------------------------------

def audit_203962() -> dict:
    log.info("[203962] Confirming static IMD")
    parquet = Path("data/processed/imd_points.parquet")
    if not parquet.exists():
        return {"error": "imd_points.parquet missing"}
    df = pd.read_parquet(parquet)
    return {
        "n_points": len(df),
        "columns": list(df.columns),
        "year_counts": (df["AÑO"].value_counts(dropna=False).to_dict()
                        if "AÑO" in df.columns else None),
        "imd_fixed_stats": (
            {"median": float(df["TOTAL_fixed"].median()),
             "max": float(df["TOTAL_fixed"].max()),
             "n_null": int(df["TOTAL_fixed"].isna().sum())}
            if "TOTAL_fixed" in df.columns else None
        ),
        "note": ("Static snapshot 2012-2013. Usable as segment-level "
                 "time-invariant covariate, NOT as temporal offset."),
    }


# ---------------------------------------------------------------------------
# Decision matrix
# ---------------------------------------------------------------------------

def build_decision_matrix(audits: dict) -> dict:
    log.info("Building decision matrix ...")
    m = {}

    # 300233
    a = audits.get("300233", {})
    if a.get("error"):
        m["300233"] = {"status": "ERROR", "error": a["error"]}
    else:
        has_monthly = "monthly_sample" in a
        m["300233"] = {
            "status": "USABLE" if has_monthly else "PARTIAL",
            "n_resources": a.get("n_resources"),
            "has_monthly_sample": has_monthly,
            "fsen_values": a.get("monthly_sample", {}).get("fsen_values"),
            "recommendation": (
                "Permanent count stations. 60 physical stations, "
                "monthly data 2018-2025. Limited spatial coverage "
                "(arterial roads only). Suitable as Poisson offset for "
                "hex250m cells containing stations."),
        }

    # 208627 (critical)
    a = audits.get("208627", {})
    if a.get("error"):
        m["208627"] = {"status": "ERROR", "error": a["error"]}
    else:
        tipo_elems = set()
        for s in a.get("samples", []):
            if "tipo_elem_distribution_top20" in s:
                tipo_elems.update(s["tipo_elem_distribution_top20"].keys())
        if not tipo_elems:
            status = "NO_TIPO_ELEM"
            recommendation = (
                "No samples with tipo_elem. Cannot determine coverage. "
                "Needs deeper audit in iter 2.")
        elif tipo_elems <= {"M30", "M30_", "m30"}:
            status = "M30_ONLY"
            recommendation = (
                "M30-ONLY. Historical cannot serve as city-wide exposure. "
                "Use 300233 as primary. Historical as M30-specific offset "
                "for hexes intersecting M-30.")
        else:
            status = "CITY_WIDE"
            recommendation = (
                f"CITY-WIDE COVERAGE: {len(tipo_elems)} distinct tipo_elem "
                f"values ({sorted(tipo_elems)[:8]}...). "
                f"15-min granularity, city-wide. GOLD MINE. "
                f"Primary temporal exposure.")
        m["208627"] = {
            "status": status,
            "n_resources": a.get("n_resources"),
            "tipo_elem_values_observed": sorted(tipo_elems),
            "n_samples_analyzed": len(a.get("samples", [])),
            "recommendation": recommendation,
        }

    # 202468
    a = audits.get("202468", {})
    if a.get("error"):
        m["202468"] = {"status": "ERROR", "error": a["error"]}
    else:
        sample = a.get("sample", {})
        has_coords = any(k in sample.get("columns", [])
                         for k in ("utm_x", "utm_y", "latitud", "longitud",
                                   "lon", "lat"))
        m["202468"] = {
            "status": "USABLE_LOCATIONS" if has_coords else "PARTIAL",
            "n_resources": a.get("n_resources"),
            "has_coordinates": has_coords,
            "recommendation": (
                "Extended point inventory. Extends spatial coverage beyond "
                "300233's 60 stations. Useful as secondary exposure or as "
                "additional offset network. Detailed assessment in iter 2."),
        }

    # 203962
    a = audits.get("203962", {})
    m["203962"] = {
        "status": "STATIC_ONLY",
        "n_points": a.get("n_points"),
        "recommendation": (
            "Static 2012-2013 snapshot. Usable ONLY as time-invariant "
            "covariate in Phase 10, not as temporal offset."),
    }

    # Overall
    if m.get("208627", {}).get("status") == "CITY_WIDE":
        overall = (
            "PRIMARY: 208627 (city-wide, 15-min). "
            "SECONDARY: 300233 (60 stations, 2018-2025). "
            "STATIC CONTROL: 203962 IMD.")
    elif m.get("208627", {}).get("status") == "M30_ONLY":
        overall = (
            "PRIMARY: 300233 (60 stations, 2018-2025). "
            "M30-specific offset: 208627. "
            "STATIC CONTROL: 203962 IMD. "
            "LIMITATION: no city-wide fine-grained exposure.")
    else:
        overall = (
            "PRIMARY: 300233 (60 stations, 2018-2025). "
            "STATIC CONTROL: 203962 IMD. "
            "208627 status uncertain.")

    m["overall_recommendation"] = overall
    return m


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def render_markdown(audits: dict, matrix: dict) -> str:
    lines = ["# Phase 9 · Iter 1 — Exposure sources audit\n"]
    lines.append(
        "**Question:** What traffic exposure can we build, at what "
        "spatial/temporal granularity, from the available sources?\n"
    )

    lines.append("## Sources audited\n")
    lines.append("| source | dataset | n_resources |\n")
    lines.append("| --- | --- | ---: |\n")
    for src, name in (
        ("300233", "Aforo tráfico permanentes"),
        ("208627", "Transporte ptomedida histórico"),
        ("202468", "Ubicación puntos de medida"),
        ("203962", "Tráfico IMD anual"),
    ):
        n = audits.get(src, {}).get("n_resources", "n/a")
        lines.append(f"| `{src}` | {name} | {n} |\n")
    lines.append("\n")

    lines.append("## Decision matrix\n")
    lines.append("| source | status | recommendation |\n")
    lines.append("| --- | --- | --- |\n")
    for src in ("300233", "208627", "202468", "203962"):
        s = matrix.get(src, {})
        lines.append(
            f"| `{src}` | **{s.get('status', 'n/a')}** | "
            f"{s.get('recommendation', '')} |\n"
        )
    lines.append("\n")

    lines.append("## 208627 (critical) — tipo_elem evidence\n")
    a = audits.get("208627", {})
    for s in a.get("samples", []):
        name = s.get("resource_name", "?")
        if "error" in s:
            lines.append(f"- `{name}`: ERROR {s['error']}\n")
            continue
        dist = s.get("tipo_elem_distribution_top20", {})
        n_types = s.get("n_unique_tipo_elem", 0)
        lines.append(
            f"- `{name}`: {n_types} unique tipo_elem — "
            f"top values: {list(dist.items())[:5]}\n"
        )
    lines.append("\n")

    lines.append("## Overall recommendation\n")
    lines.append(f"{matrix.get('overall_recommendation', 'n/a')}\n")

    return "".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    audits = {}
    audits["300233"] = audit_300233()
    audits["208627"] = audit_208627()
    audits["202468"] = audit_202468()
    audits["203962"] = audit_203962()

    matrix = build_decision_matrix(audits)

    out_json = REPORTS_TABLES / "39_exposure_audit.json"
    out_json.write_text(dumps_jsonable({"audits": audits, "matrix": matrix}),
                        encoding="utf-8")
    log.info(f"saved {out_json}")

    out_md = REPORTS / "39_exposure_audit.md"
    out_md.write_text(render_markdown(audits, matrix), encoding="utf-8")
    log.info(f"saved {out_md}")

    log.info("=" * 72)
    for src in ("300233", "208627", "202468", "203962"):
        log.info(f"[{src}] status={matrix.get(src, {}).get('status')}")
    log.info(f"Overall: {matrix.get('overall_recommendation')}")


if __name__ == "__main__":
    main()