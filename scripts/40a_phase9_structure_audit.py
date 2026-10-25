"""
Script 40a — Phase 9 iter 2a: Structural audit of 208627.

Resolves four open questions before committing to a full download:

  1. Resource → month mapping for all 161 ZIPs. Uses HTTP Range requests
     to read only the ZIP local file header (30 + filename bytes) instead
     of the full 60-100 MB archives. ~3 min for 161 resources.

  2. ZIP idx 80 (784 MB, 10× the size of others). Reads its local file
     header via Range to identify the inner CSV name.

  3. id join between 202468 (500 points, URB + other) and 208627
     (4487 IDs in a month). The 208627 `id` column and 202468 `id`
     column must be compatible for hex aggregation to work.

  4. Internal date range check on 3 ZIPs (2019, 2022, 2025) to verify
     the inferred month matches the actual content. Also resolves the
     `09-2026` anomaly from iter 1b.

Outputs:
  - reports/tables/40a_structure_audit.json
  - reports/40a_structure_audit.md
  - data/raw/40a_range_headers/*.bin  (cached local file headers)
"""

from __future__ import annotations

import io
import logging
import struct
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety.data_audit import dumps_jsonable  # noqa: E402

CKAN = "https://datos.madrid.es/api/3/action"
TIMEOUT = 60
UA = "madrid-road-safety/phase9/40a"

SAMPLES_DIR = Path("data/raw/40a_range_headers")
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("phase9_40a")


# ---------------------------------------------------------------------------
# CKAN
# ---------------------------------------------------------------------------

def ckan_pkg(dataset_id: str) -> dict | None:
    try:
        r = requests.get(f"{CKAN}/package_show",
                         params={"id": dataset_id},
                         headers={"User-Agent": UA}, timeout=TIMEOUT)
        r.raise_for_status()
        d = r.json()
        return d["result"] if d.get("success") else None
    except Exception as e:  # noqa: BLE001
        log.warning(f"  ckan_pkg({dataset_id}) failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Range request to read ZIP local file header
# ---------------------------------------------------------------------------

def read_zip_first_entry_name(url: str,
                              cache_name: str | None = None,
                              n_bytes: int = 1024) -> dict:
    """Fetch only the first N bytes of a remote ZIP and parse the local
    file header to extract the filename of the first entry.

    Returns {'zip_size_bytes', 'filename', 'compressed_size_in_header',
             'local_header_ok', 'raw_first_4_bytes'}.
    """
    out = {"url_tail": url.split("/")[-1]}

    # Cache
    if cache_name is None:
        cache_name = "hdr_" + out["url_tail"].replace("/", "_")[:80]
    cache_path = SAMPLES_DIR / cache_name

    if cache_path.exists() and cache_path.stat().st_size > 0:
        raw = cache_path.read_bytes()
        # Also try to fetch Content-Length via HEAD for size
        try:
            rh = requests.head(url, headers={"User-Agent": UA},
                               timeout=20, allow_redirects=True)
            size = rh.headers.get("Content-Length")
            out["zip_size_bytes"] = int(size) if size else None
        except Exception:  # noqa: BLE001
            out["zip_size_bytes"] = None
    else:
        try:
            r = requests.get(url,
                             headers={"User-Agent": UA,
                                      "Range": f"bytes=0-{n_bytes-1}"},
                             timeout=30, stream=False)
            r.raise_for_status()
            raw = r.content
            # Content-Length of full resource may be in Content-Range
            cr = r.headers.get("Content-Range", "")
            if "/" in cr:
                total = cr.split("/")[-1]
                out["zip_size_bytes"] = int(total) if total.isdigit() else None
            else:
                cl = r.headers.get("Content-Length")
                out["zip_size_bytes"] = int(cl) if cl and int(cl) > n_bytes else None
            cache_path.write_bytes(raw)
        except Exception as e:  # noqa: BLE001
            out["error"] = f"{type(e).__name__}: {e}"
            return out

    out["raw_first_4_bytes"] = raw[:4].hex()
    out["n_bytes_read"] = len(raw)

    if len(raw) < 30:
        out["local_header_ok"] = False
        return out

    sig = struct.unpack("<I", raw[:4])[0]
    if sig != 0x04034b50:  # PK\x03\x04
        out["local_header_ok"] = False
        return out

    out["local_header_ok"] = True
    # Parse header fields
    # (2) version, (2) flags, (2) method, (2) modtime, (2) moddate,
    # (4) crc, (4) comp_size, (4) uncomp_size, (2) name_len, (2) extra_len
    try:
        fields = struct.unpack("<2s2s2s2s2s4s4s4s2s2s", raw[4:30])
        name_len = struct.unpack("<H", fields[8])[0]
        comp_size = struct.unpack("<I", fields[6])[0]
        out["filename"] = raw[30:30 + name_len].decode(
            "utf-8", errors="replace")
        out["compressed_size_in_header"] = comp_size
    except Exception as e:  # noqa: BLE001
        out["parse_error"] = str(e)

    return out


def infer_month(filename: str) -> str | None:
    """Map 'MM-YYYY.csv' → 'YYYY-MM'. Also handle 'YYYY.csv' for annual."""
    if not filename:
        return None
    stem = filename.replace(".csv", "").replace(".CSV", "")
    parts = stem.split("-")
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        mm, yyyy = parts
        if 1 <= int(mm) <= 12 and 2010 <= int(yyyy) <= 2030:
            return f"{yyyy}-{int(mm):02d}"
    if stem.isdigit() and 2010 <= int(stem) <= 2030:
        return f"{stem}-ANNUAL"
    return None


# ---------------------------------------------------------------------------
# Full ZIP sampling (validate month vs content)
# ---------------------------------------------------------------------------

def fetch_and_validate_month(url: str, expected_month: str | None,
                             cache_path: Path) -> dict:
    """Download full ZIP, open inner CSV, check date range."""
    if cache_path.exists() and cache_path.stat().st_size > 0:
        content = cache_path.read_bytes()
    else:
        try:
            r = requests.get(url, headers={"User-Agent": UA},
                             timeout=240, stream=True)
            r.raise_for_status()
            content = r.content
            cache_path.write_bytes(content)
        except Exception as e:  # noqa: BLE001
            return {"error": f"download_failed: {e}"}

    out = {"zip_size_bytes": len(content)}
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            csvs = [n for n in zf.namelist()
                    if n.lower().endswith(".csv")]
            if not csvs:
                out["error"] = "no_csv_inside"
                return out
            csv_name = csvs[0]
            out["csv_inner_name"] = csv_name
            out["inferred_month_from_name"] = infer_month(csv_name)
            out["expected_month"] = expected_month
            out["month_matches"] = (
                out["inferred_month_from_name"] == expected_month
                if expected_month else None
            )

            # Read first 100k rows for date stats
            with zf.open(csv_name) as f:
                head = f.read(2_000_000)  # 2 MB
            df = pd.read_csv(io.BytesIO(head), sep=";",
                             encoding="utf-8-sig",
                             nrows=50_000, on_bad_lines="skip")
            if "fecha" in df.columns:
                dates = pd.to_datetime(df["fecha"], errors="coerce")
                valid = dates.dropna()
                if len(valid) > 0:
                    out["fecha_min_sample"] = str(valid.min())
                    out["fecha_max_sample"] = str(valid.max())
                    out["n_fecha_sample"] = len(valid)
            out["columns"] = list(df.columns)
            out["sample_tipo_elem"] = (
                df["tipo_elem"].value_counts(dropna=False).to_dict()
                if "tipo_elem" in df.columns else None
            )
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"

    return out


# ---------------------------------------------------------------------------
# id join check
# ---------------------------------------------------------------------------

def check_id_join(zip_208627_url: str,
                  sample_202468_path: Path,
                  cache_208627: Path) -> dict:
    """Compare IDs between a full 208627 month and the 202468 catalogue."""
    out = {}

    # --- 202468: download full CSV ---
    if not sample_202468_path.exists():
        try:
            r = requests.get(
                "https://datos.madrid.es/dataset/202468-0-intensidad-trafico/"
                "resource/202468-0-intensidad-trafico-csv/download/"
                "202468-0-intensidad-trafico-csv.csv",
                headers={"User-Agent": UA}, timeout=120)
            r.raise_for_status()
            sample_202468_path.write_bytes(r.content)
        except Exception as e:  # noqa: BLE001
            out["catalog_error"] = str(e)
            return out

    try:
        df_cat = pd.read_csv(sample_202468_path, sep=";",
                             encoding="utf-8-sig",
                             on_bad_lines="skip")
        out["catalog_n_rows"] = len(df_cat)
        out["catalog_columns"] = list(df_cat.columns)
        ids_cat = set(df_cat["id"].dropna().astype(str).tolist()) \
            if "id" in df_cat.columns else set()
        out["catalog_n_unique_ids"] = len(ids_cat)
        out["catalog_tipo_elem"] = (
            df_cat["tipo_elem"].value_counts(dropna=False).to_dict()
            if "tipo_elem" in df_cat.columns else None
        )
        out["catalog_id_min"] = min(ids_cat) if ids_cat else None
        out["catalog_id_max"] = max(ids_cat) if ids_cat else None
    except Exception as e:  # noqa: BLE001
        out["catalog_parse_error"] = str(e)
        return out

    # --- 208627: full month ---
    if not cache_208627.exists():
        try:
            r = requests.get(zip_208627_url,
                             headers={"User-Agent": UA},
                             timeout=240, stream=True)
            r.raise_for_status()
            cache_208627.write_bytes(r.content)
        except Exception as e:  # noqa: BLE001
            out["month_error"] = str(e)
            return out

    try:
        with zipfile.ZipFile(cache_208627) as zf:
            csvs = [n for n in zf.namelist()
                    if n.lower().endswith(".csv")]
            if not csvs:
                out["month_error"] = "no csv inside"
                return out
            csv_name = csvs[0]
            out["month_csv"] = csv_name
            with zf.open(csv_name) as f:
                df_month = pd.read_csv(
                    f, sep=";", encoding="utf-8-sig",
                    usecols=["id"] if False else None,
                    on_bad_lines="skip",
                    nrows=200_000,  # 200k is enough for unique id coverage
                )
        ids_month = set(df_month["id"].dropna().astype(str).tolist()) \
            if "id" in df_month.columns else set()
        out["month_n_unique_ids_sample"] = len(ids_month)
        out["month_id_min"] = min(ids_month) if ids_month else None
        out["month_id_max"] = max(ids_month) if ids_month else None
    except Exception as e:  # noqa: BLE001
        out["month_parse_error"] = str(e)
        return out

    # --- Intersection ---
    intersect = ids_cat & ids_month
    out["n_intersection"] = len(intersect)
    out["frac_catalog_matched"] = (
        round(len(intersect) / len(ids_cat), 4) if ids_cat else None
    )
    out["frac_month_matched"] = (
        round(len(intersect) / len(ids_month), 4) if ids_month else None
    )
    # Sample of IDs from each side for inspection
    out["sample_catalog_ids"] = sorted(list(ids_cat))[:10]
    out["sample_month_ids"] = sorted(list(ids_month))[:10]

    return out


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def render_md(result: dict) -> str:
    lines = ["# Phase 9 · Iter 2a — Structural audit of 208627\n"]

    lines.append("## 1. Resource → month mapping (all 161 ZIPs)\n")
    m = result.get("resource_map", {})
    lines.append(f"- Attempted: {m.get('n_attempted', 0)}\n")
    lines.append(f"- Successfully parsed: {m.get('n_parsed', 0)}\n")
    lines.append(f"- Failed: {m.get('n_failed', 0)}\n")

    coverage = m.get("coverage_by_year", {})
    if coverage:
        lines.append("\n### Coverage by year (inferred)\n")
        lines.append("| year | months present | complete? |\n")
        lines.append("| --- | --- | --- |\n")
        for y in sorted(coverage.keys()):
            months = coverage[y]
            n = len(months)
            lines.append(
                f"| {y} | {sorted(months)} | "
                f"{'FULL' if n == 12 else f'PARTIAL ({n}/12)'} |\n"
            )
        lines.append("\n")

    gaps = m.get("gaps_2019_2025", [])
    if gaps:
        lines.append(f"\n**Gaps in 2019-2025:** {gaps}\n")
    else:
        lines.append("\n**No gaps in 2019-2025.** Full monthly coverage.\n")

    # non-standard entries
    anom = m.get("anomalous_entries", [])
    if anom:
        lines.append(f"\n**Anomalous entries:** {anom}\n")

    lines.append("\n## 2. ZIP idx 80 (784 MB)\n")
    z = result.get("idx_80", {})
    lines.append(f"- inner csv name: `{z.get('filename', 'n/a')}`\n")
    lines.append(f"- inferred month: `{z.get('inferred_month_from_name', 'n/a')}`\n")
    lines.append(f"- zip size: {z.get('zip_size_bytes', 'n/a')}\n")

    lines.append("\n## 3. Internal date validation (3 ZIPs)\n")
    for v in result.get("date_validation", []):
        lines.append(
            f"- `{v.get('csv_inner_name', v.get('error', '?'))}` "
            f"inferred={v.get('inferred_month_from_name')} "
            f"content_fecha=[{v.get('fecha_min_sample','?')}, "
            f"{v.get('fecha_max_sample','?')}] "
            f"match={v.get('month_matches')}\n"
        )

    lines.append("\n## 4. id join check (202468 vs 208627)\n")
    j = result.get("id_join", {})
    lines.append(f"- catalog unique ids: {j.get('catalog_n_unique_ids')}\n")
    lines.append(f"- catalog id range: [{j.get('catalog_id_min')}, "
                 f"{j.get('catalog_id_max')}]\n")
    lines.append(f"- month sample unique ids: {j.get('month_n_unique_ids_sample')}\n")
    lines.append(f"- month sample id range: [{j.get('month_id_min')}, "
                 f"{j.get('month_id_max')}]\n")
    lines.append(f"- intersection: {j.get('n_intersection')}\n")
    lines.append(f"- frac catalog matched: {j.get('frac_catalog_matched')}\n")
    lines.append(f"- frac month matched: {j.get('frac_month_matched')}\n")

    lines.append("\n## Decision matrix\n")
    lines.append(result.get("decision_text", ""))
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

    pkg = ckan_pkg("208627-0-transporte-ptomedida-historico")
    if not pkg:
        raise SystemExit("cannot fetch 208627 CKAN metadata")

    resources = [r for r in pkg["resources"]
                 if (r.get("format") or "").upper() == "ZIP"]
    log.info(f"208627 has {len(resources)} ZIP resources")

    # ------------------------------------------------------------------
    # 1. Read local file headers via Range requests
    # ------------------------------------------------------------------
    log.info("=" * 72)
    log.info("STEP 1/4: reading local file headers for all ZIPs")
    resource_map = []
    t0 = time.time()
    for i, r in enumerate(resources):
        url = r["url"]
        cache_name = f"hdr_{i:03d}.bin"
        hdr = read_zip_first_entry_name(url, cache_name=cache_name)
        hdr["resource_index"] = i
        hdr["resource_name"] = r.get("name")
        hdr["inferred_month"] = infer_month(hdr.get("filename", ""))
        resource_map.append(hdr)
        if (i + 1) % 20 == 0:
            log.info(f"  {i + 1}/{len(resources)} headers read "
                     f"({time.time() - t0:.0f}s)")

    n_parsed = sum(1 for x in resource_map
                   if x.get("local_header_ok") and x.get("filename"))
    n_failed = len(resource_map) - n_parsed

    coverage_by_year = {}
    anomalous = []
    for x in resource_map:
        m = x.get("inferred_month")
        if not m:
            anomalous.append({
                "index": x["resource_index"],
                "filename": x.get("filename"),
                "size": x.get("zip_size_bytes"),
            })
            continue
        parts = m.split("-")
        if len(parts) == 2 and parts[1] == "ANNUAL":
            anomalous.append({
                "index": x["resource_index"],
                "filename": x.get("filename"),
                "note": "ANNUAL aggregate",
            })
            continue
        year, month = parts
        coverage_by_year.setdefault(year, []).append(month)

    gaps_2019_2025 = []
    for y in ("2019", "2020", "2021", "2022", "2023", "2024", "2025"):
        present = set(coverage_by_year.get(y, []))
        all_months = {f"{m:02d}" for m in range(1, 13)}
        missing = sorted(all_months - present)
        if missing:
            gaps_2019_2025.append({"year": y, "missing_months": missing})

    step1 = {
        "n_attempted": len(resources),
        "n_parsed": n_parsed,
        "n_failed": n_failed,
        "coverage_by_year": {
            y: sorted(set(ms)) for y, ms in coverage_by_year.items()
        },
        "gaps_2019_2025": gaps_2019_2025,
        "anomalous_entries": anomalous,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    log.info(f"  coverage: {sorted(coverage_by_year.keys())}")

    # ------------------------------------------------------------------
    # 2. Inspect idx 80 (the 784 MB anomaly)
    # ------------------------------------------------------------------
    log.info("=" * 72)
    log.info("STEP 2/4: inspecting the 784 MB ZIP (idx 80)")
    idx_80_url = next((r["url"] for i, r in enumerate(resources) if i == 80),
                      None)
    idx_80 = {}
    if idx_80_url:
        hdr = read_zip_first_entry_name(idx_80_url,
                                        cache_name="hdr_080.bin")
        hdr["inferred_month_from_name"] = infer_month(hdr.get("filename", ""))
        idx_80 = hdr

    # ------------------------------------------------------------------
    # 3. Internal date validation on 3 sample months
    # ------------------------------------------------------------------
    log.info("=" * 72)
    log.info("STEP 3/4: internal date validation on 3 ZIPs")
    # Pick first month of 2019, 2022, 2025
    targets = {"2019-01": None, "2022-06": None, "2025-01": None}
    for x in resource_map:
        m = x.get("inferred_month")
        if m in targets and targets[m] is None:
            targets[m] = x["resource_index"]
    # If 2019-01 missing, fall back to first month of 2019
    if targets["2019-01"] is None:
        for x in resource_map:
            if str(x.get("inferred_month", "")).startswith("2019"):
                targets["2019-01"] = x["resource_index"]
                break
    # If 2022-06 missing, first of 2022
    if targets["2022-06"] is None:
        for x in resource_map:
            if str(x.get("inferred_month", "")).startswith("2022"):
                targets["2022-06"] = x["resource_index"]
                break
    # If 2025-01 missing, first of 2025
    if targets["2025-01"] is None:
        for x in resource_map:
            if str(x.get("inferred_month", "")).startswith("2025"):
                targets["2025-01"] = x["resource_index"]
                break

    date_validation = []
    for label, idx in targets.items():
        if idx is None:
            date_validation.append({"target": label,
                                    "error": "no resource found"})
            continue
        url = resources[idx]["url"]
        cache = SAMPLES_DIR / f"full_{label}.zip"
        log.info(f"  downloading {label} (idx {idx}) ...")
        v = fetch_and_validate_month(url, label, cache)
        v["target"] = label
        date_validation.append(v)

    # ------------------------------------------------------------------
    # 4. id join check (202468 vs 208627)
    # ------------------------------------------------------------------
    log.info("=" * 72)
    log.info("STEP 4/4: id join check (202468 vs 208627)")
    # Use the 2022-06 download (already cached) for month side
    month_zip_path = SAMPLES_DIR / "full_2022-06.zip"
    month_zip_url = next(
        (r["url"] for x, r in enumerate(resources)
         if x == targets.get("2022-06")), resources[0]["url"])
    id_join = check_id_join(
        zip_208627_url=month_zip_url,
        sample_202468_path=SAMPLES_DIR / "202468_catalog.csv",
        cache_208627=month_zip_path,
    )

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------
    decision_lines = []
    if n_parsed < 100:
        decision_lines.append(
            f"**WARNING:** only {n_parsed}/{len(resources)} headers parsed. "
            "Range requests may not be supported consistently; "
            "resource-to-month mapping is unreliable.")
    else:
        decision_lines.append(
            f"Resource mapping: {n_parsed}/{len(resources)} parsed "
            f"({n_failed} failed).")

    if not gaps_2019_2025:
        decision_lines.append(
            "**Full monthly coverage 2019-2025 confirmed.** "
            "Ready to build exposure aggregates.")
    else:
        years_with_gaps = [g["year"] for g in gaps_2019_2025]
        decision_lines.append(
            f"Gaps detected in {years_with_gaps}. "
            "Exposure aggregates will skip missing months.")

    if id_join.get("frac_catalog_matched", 0) is not None and \
       id_join.get("frac_catalog_matched", 0) > 0.9:
        decision_lines.append(
            f"**id join OK:** {id_join['frac_catalog_matched']:.0%} of "
            f"catalog IDs matched in 208627. Can map IDs → UTM coordinates.")
    elif id_join.get("frac_catalog_matched") is not None:
        decision_lines.append(
            f"**id join WARNING:** only "
            f"{id_join['frac_catalog_matched']:.0%} of catalog IDs matched. "
            f"May need fuzzy join or additional sources.")
    else:
        decision_lines.append("id join could not be evaluated.")

    if idx_80.get("local_header_ok"):
        decision_lines.append(
            f"**ZIP idx 80:** inner csv `{idx_80.get('filename')}`, "
            f"size {idx_80.get('zip_size_bytes', 0) / 1e6:.0f} MB.")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    result = {
        "resource_map": step1,
        "idx_80": idx_80,
        "date_validation": date_validation,
        "id_join": id_join,
        "decision_text": " ".join(decision_lines),
        "raw_resource_map": resource_map,  # full listing
    }

    out_json = Path("reports/tables/40a_structure_audit.json")
    out_md = Path("reports/40a_structure_audit.md")
    out_json.write_text(dumps_jsonable(result), encoding="utf-8")
    out_md.write_text(render_md(result), encoding="utf-8")
    log.info(f"saved {out_json}")
    log.info(f"saved {out_md}")

    log.info("=" * 72)
    for line in decision_lines:
        log.info(line)


if __name__ == "__main__":
    main()