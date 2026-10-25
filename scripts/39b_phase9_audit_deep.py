"""
Script 39b — Phase 9 iter 1b: Deep audit of exposure sources.

Resolves four open questions from iter 1:

  1. 202468 (measurement points): recover the correct CKAN ID,
     audit schema, confirm whether it expands city-wide coverage.

  2. 208627 (historical): map all 162 resources to months,
     sample 5 ZIPs across the resource list, count UNIQUE IDs
     in a full-month CSV (not truncated), find whether any ZIP
     contains tipo_elem other than C30/M30.

  3. 205865 (IMD laborables): the forgotten dataset. Sample CSV,
     check schema, evaluate as possible city-wide alternative.

  4. 300233 (permanent stations): confirm exact temporal coverage
     (how many months, date range), and count total station×submeasure
     points in the network.

Outputs:
  - data/raw/39b_samples/*
  - reports/tables/39b_exposure_audit_deep.json
  - reports/39b_exposure_audit_deep.md
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
UA = "madrid-road-safety/phase9b/0.1"

SAMPLES_DIR = Path("data/raw/39b_samples")
REPORTS_TABLES = Path("reports/tables")
REPORTS = Path("reports")

SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_TABLES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("phase9b")


# ---------------------------------------------------------------------------
# CKAN helpers
# ---------------------------------------------------------------------------

def ckan_pkg_show(dataset_id: str) -> dict | None:
    try:
        r = requests.get(f"{CKAN}/package_show",
                         params={"id": dataset_id},
                         headers={"User-Agent": UA}, timeout=TIMEOUT)
        r.raise_for_status()
        d = r.json()
        return d["result"] if d.get("success") else None
    except Exception:  # noqa: BLE001
        return None


def ckan_search(q: str, rows: int = 10) -> list[dict]:
    try:
        r = requests.get(f"{CKAN}/package_search",
                         params={"q": q, "rows": rows},
                         headers={"User-Agent": UA}, timeout=TIMEOUT)
        r.raise_for_status()
        d = r.json()
        if not d.get("success"):
            return []
        return d["result"].get("results", [])
    except Exception:  # noqa: BLE001
        return []


def download(url: str, dest: Path, max_mb: int = 300,
             timeout: int = 240) -> bytes | None:
    if dest.exists() and dest.stat().st_size > 0:
        return dest.read_bytes()
    try:
        r = requests.get(url, headers={"User-Agent": UA},
                         timeout=timeout, stream=True)
        r.raise_for_status()
        # sanity size
        size = int(r.headers.get("Content-Length", 0))
        if size and size > max_mb * 1024 * 1024:
            log.warning(f"  skip {dest.name}: {size / 1e6:.0f} MB > {max_mb} MB")
            return None
        content = r.content
        dest.write_bytes(content)
        log.info(f"  downloaded {len(content) / 1e6:.1f} MB -> {dest.name}")
        return content
    except Exception as e:  # noqa: BLE001
        log.warning(f"  download failed: {e}")
        return None


def read_csv_multi(content: bytes, max_rows: int = 5000) -> tuple:
    for enc in ("utf-8-sig", "latin-1", "utf-8"):
        for sep in (";", ",", "\t", "|"):
            try:
                df = pd.read_csv(io.BytesIO(content), sep=sep, encoding=enc,
                                 nrows=max_rows, on_bad_lines="skip")
                if df.shape[1] >= 2:
                    return df, enc, sep
            except Exception:  # noqa: BLE001
                continue
    return None, None, None


# ---------------------------------------------------------------------------
# 1. 202468 — Recover
# ---------------------------------------------------------------------------

def recover_202468() -> dict:
    log.info("[202468] Recovering dataset")
    result = {"dataset_id_attempts": []}

    candidate_ids = [
        "202468-0-ubicacion-puntos-medida",
        "202468-0-ubicacion-puntos-de-medida",
        "202468-0-puntos-medida",
        "202468",
    ]
    queries = [
        "ubicacion puntos medida",
        "puntos de medida trafico",
        "202468",
    ]

    found = None
    for cid in candidate_ids:
        pkg = ckan_pkg_show(cid)
        result["dataset_id_attempts"].append({
            "id": cid, "found": pkg is not None,
            "n_resources": len(pkg.get("resources", [])) if pkg else 0,
        })
        if pkg:
            found = pkg
            result["dataset_id"] = cid
            break

    if not found:
        # Search CKAN
        for q in queries:
            hits = ckan_search(q, rows=5)
            result["dataset_id_attempts"].append({
                "query": q, "n_hits": len(hits),
                "titles": [h.get("title") for h in hits],
                "names": [h.get("name") for h in hits],
            })
            if hits:
                found = hits[0]
                result["dataset_id"] = found.get("name")
                break

    if not found:
        result["error"] = "not found via any query"
        return result

    resources = found.get("resources", [])
    result["n_resources"] = len(resources)
    result["title"] = found.get("title")
    result["resource_formats"] = sorted(set(
        r.get("format") for r in resources if r.get("format")
    ))

    csvs = [r for r in resources
            if (r.get("format") or "").upper() == "CSV"]
    if csvs:
        r = csvs[0]
        content = download(r["url"],
                           SAMPLES_DIR / "202468_sample.csv",
                           max_mb=30)
        if content:
            df, enc, sep = read_csv_multi(content, max_rows=500)
            if df is not None:
                result["sample"] = {
                    "resource_name": r.get("name"),
                    "columns": list(df.columns),
                    "n_rows_sampled": len(df),
                    "encoding": enc, "sep": sep,
                    "sample_rows": df.head(3).to_dict(orient="records"),
                }
                if "tipo_elem" in df.columns:
                    result["sample"]["tipo_elem_values"] = (
                        df["tipo_elem"].value_counts().head(20).to_dict()
                    )
            else:
                result["sample"] = {"error": "csv_parse_failed",
                                    "head": content[:300].decode("latin-1",
                                                                  errors="replace")}
    return result


# ---------------------------------------------------------------------------
# 2. 208627 — Map resources to dates, deep sample
# ---------------------------------------------------------------------------

def map_208627_resources() -> dict:
    log.info("[208627] Mapping resources to dates")
    pkg = ckan_pkg_show("208627-0-transporte-ptomedida-historico")
    if not pkg:
        return {"error": "package_show failed"}

    resources = [r for r in pkg.get("resources", [])
                 if (r.get("format") or "").upper() == "ZIP"]
    log.info(f"  {len(resources)} ZIP resources")

    # Try to infer date from resource name if present
    listing = []
    for i, r in enumerate(resources):
        name = r.get("name") or ""
        desc = r.get("description") or ""
        listing.append({
            "index": i,
            "name": name,
            "description": desc[:80],
            "url_tail": (r.get("url") or "").split("/")[-1],
        })
    return {
        "n_zip_resources": len(resources),
        "resources_listing": listing,
        "first_5": listing[:5],
        "last_5": listing[-5:],
    }


def sample_208627_across(n_samples: int = 5) -> dict:
    """Sample n ZIPs evenly across the resource list, identify month."""
    log.info(f"[208627] Sampling {n_samples} ZIPs across the resource list")
    pkg = ckan_pkg_show("208627-0-transporte-ptomedida-historico")
    if not pkg:
        return {"error": "package_show failed"}

    resources = [r for r in pkg.get("resources", [])
                 if (r.get("format") or "").upper() == "ZIP"]
    if not resources:
        return {"error": "no ZIP resources"}

    n = len(resources)
    idxs = sorted(set([int(i * (n - 1) / (n_samples - 1))
                       for i in range(n_samples)]))

    samples = []
    for i, idx in enumerate(idxs):
        r = resources[idx]
        log.info(f"  [{i+1}/{len(idxs)}] resource index {idx}: {r.get('name')}")
        dest = SAMPLES_DIR / f"208627_deep_{i+1}.zip"
        content = download(r["url"], dest, max_mb=300, timeout=240)
        if not content:
            samples.append({"index": idx, "name": r.get("name"),
                            "error": "download_failed"})
            continue

        try:
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                csvs = [n for n in zf.namelist()
                        if n.lower().endswith(".csv")]
                if not csvs:
                    samples.append({"index": idx, "name": r.get("name"),
                                    "error": "no_csv",
                                    "zip_contents": zf.namelist()[:3]})
                    continue
                csv_name = csvs[0]
                with zf.open(csv_name) as f:
                    csv_bytes = f.read()

                # Parse the CSV month from its filename
                month_inferred = csv_name.replace(".csv", "")

                # Full month: count unique ids using chunks to avoid RAM blowup
                n_rows_total = 0
                id_set = set()
                tipo_counter = {}
                try:
                    for chunk in pd.read_csv(
                        io.BytesIO(csv_bytes), sep=";",
                        encoding="utf-8-sig",
                        chunksize=100_000,
                        usecols=["id", "tipo_elem"] if False else None,
                        on_bad_lines="skip",
                    ):
                        n_rows_total += len(chunk)
                        if "id" in chunk.columns:
                            id_set.update(
                                chunk["id"].dropna().astype(str).unique())
                        if "tipo_elem" in chunk.columns:
                            for k, v in (chunk["tipo_elem"]
                                         .value_counts().items()):
                                tipo_counter[k] = tipo_counter.get(k, 0) + int(v)
                except Exception as e:  # noqa: BLE001
                    samples.append({
                        "index": idx, "name": r.get("name"),
                        "csv_name": csv_name,
                        "error": f"full_month_parse_failed: {e}",
                    })
                    continue

                samples.append({
                    "index": idx, "name": r.get("name"),
                    "csv_name": csv_name,
                    "month_inferred": month_inferred,
                    "n_rows_total": n_rows_total,
                    "n_unique_ids": len(id_set),
                    "id_min": min(id_set) if id_set else None,
                    "id_max": max(id_set) if id_set else None,
                    "tipo_elem_counts": tipo_counter,
                    "tipo_elem_unique": sorted(tipo_counter.keys()),
                })
        except Exception as e:  # noqa: BLE001
            samples.append({"index": idx, "name": r.get("name"),
                            "error": f"zip_error: {type(e).__name__}: {e}"})

    return {"n_samples": len(samples), "samples": samples}


# ---------------------------------------------------------------------------
# 3. 205865 — Forgotten dataset
# ---------------------------------------------------------------------------

def audit_205865() -> dict:
    log.info("[205865] Auditing 'intensidad media laborables'")
    result = {"dataset_id": "205865-0-intensidad-media-laborables"}

    pkg = ckan_pkg_show(result["dataset_id"])
    if not pkg:
        # Try search
        hits = ckan_search("intensidad media laborables", rows=5)
        if hits:
            pkg = hits[0]
            result["dataset_id"] = pkg.get("name")
        else:
            result["error"] = "not found"
            return result

    resources = pkg.get("resources", [])
    result["title"] = pkg.get("title")
    result["n_resources"] = len(resources)
    result["resource_formats"] = sorted(set(
        r.get("format") for r in resources if r.get("format")
    ))

    csvs = [r for r in resources
            if (r.get("format") or "").upper() == "CSV"]
    if not csvs:
        result["error"] = "no CSV resources"
        return result

    r = csvs[0]
    content = download(r["url"], SAMPLES_DIR / "205865_sample.csv",
                       max_mb=20)
    if not content:
        result["error"] = "download_failed"
        return result

    df, enc, sep = read_csv_multi(content, max_rows=500)
    if df is None:
        result["error"] = "csv_parse_failed"
        result["head"] = content[:300].decode("latin-1", errors="replace")
        return result

    result["sample"] = {
        "resource_name": r.get("name"),
        "columns": list(df.columns),
        "n_rows_sampled": len(df),
        "encoding": enc, "sep": sep,
        "sample_rows": df.head(3).to_dict(orient="records"),
    }
    return result


# ---------------------------------------------------------------------------
# 4. 300233 — Temporal coverage
# ---------------------------------------------------------------------------

def audit_300233_coverage() -> dict:
    log.info("[300233] Confirming temporal coverage and station count")
    pkg = ckan_pkg_show("300233-0-aforo-trafico-permanentes")
    if not pkg:
        return {"error": "package_show failed"}

    resources = pkg.get("resources", [])
    monthly = [r for r in resources
               if (r.get("format") or "").upper() == "CSV"
               and "110" not in (r.get("name") or "")
               and "112" not in (r.get("name") or "")]

    # Try to infer year-month from resource names
    dates = []
    for r in monthly:
        name = r.get("name") or ""
        # Resource names look like "300233-N-..." — no date in name.
        # But the download URL often contains YYYY-MM or similar.
        url = r.get("url") or ""
        tail = url.split("/")[-1]
        dates.append(tail)

    return {
        "n_monthly_resources": len(monthly),
        "sample_url_tails": dates[:10],
        "note": ("Resource names carry no date; URL structure may. "
                 "Full coverage requires downloading each monthly file "
                 "and checking FDIA. Not done in this audit."),
    }


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

def build_updated_matrix(audits: dict) -> dict:
    m = {}

    # 202468
    a = audits.get("202468", {})
    if a.get("sample") and "columns" in a.get("sample", {}):
        has_coords = any(c in a["sample"]["columns"]
                         for c in ("utm_x", "utm_y", "latitud", "longitud"))
        m["202468"] = {
            "status": "USABLE" if has_coords else "PARTIAL",
            "dataset_id": a.get("dataset_id"),
            "n_resources": a.get("n_resources"),
            "sample_columns": a["sample"]["columns"],
            "sample_tipo_elem": a["sample"].get("tipo_elem_values"),
        }
    else:
        m["202468"] = {"status": "UNRESOLVED",
                       "error": a.get("error", "no sample")}

    # 208627
    a = audits.get("208627_deep", {})
    if "samples" in a:
        # Aggregate tipo_elem across all samples
        tipo_union = set()
        unique_ids_per_sample = []
        for s in a["samples"]:
            if "tipo_elem_unique" in s:
                tipo_union.update(s["tipo_elem_unique"])
                unique_ids_per_sample.append(s.get("n_unique_ids"))
        # Remove C30/M30 synonyms for status decision
        non_m30 = tipo_union - {"C30", "M30", "m30", "C30_", "M30_"}
        if not non_m30:
            status = "M30_ONLY"
        else:
            status = "MULTI_NETWORK"
        m["208627"] = {
            "status": status,
            "tipo_elem_union": sorted(tipo_union),
            "n_non_m30_values": len(non_m30),
            "unique_ids_per_sample": unique_ids_per_sample,
            "n_samples": len(a["samples"]),
        }
    else:
        m["208627"] = {"status": "UNKNOWN"}

    # 205865
    a = audits.get("205865", {})
    if a.get("sample") and "columns" in a.get("sample", {}):
        m["205865"] = {
            "status": "USABLE",
            "dataset_id": a.get("dataset_id"),
            "columns": a["sample"]["columns"],
            "n_rows_sampled": a["sample"]["n_rows_sampled"],
        }
    else:
        m["205865"] = {"status": "UNKNOWN",
                       "error": a.get("error")}

    # 300233 coverage (partial info)
    a = audits.get("300233_coverage", {})
    m["300233"] = {
        "status": "USABLE",
        "n_monthly_resources": a.get("n_monthly_resources"),
        "note": a.get("note"),
    }

    # Overall recommendation update
    if m.get("208627", {}).get("status") == "M30_ONLY":
        if m.get("205865", {}).get("status") == "USABLE":
            overall = (
                "Exposure city-wide from 205865 (to verify scope). "
                "208627 is M-30 ONLY, not city-wide. "
                "300233 remains 60 stations. 203962 static-only."
            )
        else:
            overall = (
                "CITY-WIDE FINE-GRAINED EXPOSURE NOT AVAILABLE. "
                "208627 is M-30 only. 300233 has 60 stations. "
                "Phase 10 must use exposure proxy: IMD static + "
                "network length per cell, or restrict to station-containing cells."
            )
    else:
        overall = "Reassess after 205865 inspection."

    m["overall_recommendation"] = overall
    return m


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def render_md(audits: dict, matrix: dict) -> str:
    lines = ["# Phase 9 · Iter 1b — Deep exposure audit\n"]
    lines.append("Resolves four open questions from iter 1.\n\n")

    # 208627
    lines.append("## 208627 — resource-to-month mapping and full-month ID count\n")
    deep = audits.get("208627_deep", {})
    for s in deep.get("samples", []):
        if "error" in s:
            lines.append(f"- idx {s['index']} `{s['name']}`: ERROR {s['error']}\n")
            continue
        lines.append(
            f"- idx {s['index']} `{s.get('csv_name')}` "
            f"(month inferred): {s.get('n_rows_total'):,} rows, "
            f"**{s.get('n_unique_ids')}** unique ids "
            f"(min={s.get('id_min')}, max={s.get('id_max')}), "
            f"tipo_elem={s.get('tipo_elem_unique')}\n"
        )
    lines.append("\n")

    # 202468
    lines.append("## 202468 — recovered?\n")
    r = audits.get("202468", {})
    lines.append(f"- dataset_id: `{r.get('dataset_id', 'n/a')}`\n")
    lines.append(f"- title: `{r.get('title', 'n/a')}`\n")
    if "sample" in r:
        s = r["sample"]
        lines.append(f"- columns: {s.get('columns')}\n")
        lines.append(f"- tipo_elem values: {s.get('tipo_elem_values')}\n")
    else:
        lines.append(f"- error: {r.get('error')}\n")
    lines.append("\n")

    # 205865
    lines.append("## 205865 — forgotten dataset\n")
    r = audits.get("205865", {})
    if "sample" in r:
        s = r["sample"]
        lines.append(f"- title: `{r.get('title')}`\n")
        lines.append(f"- columns: {s.get('columns')}\n")
        lines.append(f"- sample rows: {s.get('sample_rows')}\n")
    else:
        lines.append(f"- error: {r.get('error', 'unknown')}\n")
    lines.append("\n")

    # Matrix
    lines.append("## Updated decision matrix\n")
    lines.append("| source | status | key facts |\n")
    lines.append("| --- | --- | --- |\n")
    for src in ("208627", "205865", "202468", "300233"):
        s = matrix.get(src, {})
        lines.append(f"| `{src}` | **{s.get('status')}** | "
                     f"{_short(s)} |\n")
    lines.append("\n")

    lines.append("## Overall\n")
    lines.append(f"{matrix.get('overall_recommendation')}\n")
    return "".join(lines)


def _short(s: dict) -> str:
    if "tipo_elem_union" in s:
        return (f"tipo_elem={s['tipo_elem_union']}, "
                f"ids_per_sample={s.get('unique_ids_per_sample')}")
    if "sample_columns" in s:
        return f"columns={s['sample_columns']}"
    if "columns" in s:
        return f"columns={s['columns']}"
    return s.get("note", s.get("error", ""))


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

    log.info("=" * 72)
    audits["202468"] = recover_202468()

    log.info("=" * 72)
    audits["208627_deep"] = sample_208627_across(n_samples=5)

    log.info("=" * 72)
    audits["205865"] = audit_205865()

    log.info("=" * 72)
    audits["300233_coverage"] = audit_300233_coverage()

    matrix = build_updated_matrix(audits)

    out_json = REPORTS_TABLES / "39b_exposure_audit_deep.json"
    out_json.write_text(
        dumps_jsonable({"audits": audits, "matrix": matrix}),
        encoding="utf-8",
    )
    log.info(f"saved {out_json}")

    out_md = REPORTS / "39b_exposure_audit_deep.md"
    out_md.write_text(render_md(audits, matrix), encoding="utf-8")
    log.info(f"saved {out_md}")

    log.info("=" * 72)
    for k in ("208627", "205865", "202468", "300233"):
        log.info(f"[{k}] {matrix.get(k, {}).get('status')}")
    log.info(f"Overall: {matrix.get('overall_recommendation')}")


if __name__ == "__main__":
    main()