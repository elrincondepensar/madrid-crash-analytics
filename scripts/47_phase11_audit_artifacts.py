"""
Script 47 — Phase 11 iter 3: Artifact inventory and cleanup recommendations.

v2 fixes:
  - Added MANUAL_PRODUCERS map for parquets whose producer cannot be
    inferred from the filename prefix (they have no numeric prefix, or
    use f-string dynamic names like moran_{unit}.parquet). Verified
    against mtime + script logs.
  - Producer resolution now checks MANUAL_PRODUCERS first, then falls
    back to the numeric-prefix heuristic for JSONs/tables.

Scans:
  - data/processed/*.parquet (top-level only; skips known intermediates)
  - reports/tables/*.json
  - reports/figures/final/*.png
  - scripts/*.py

Outputs:
  reports/47_artifacts_inventory.json
  reports/47_artifacts_inventory.md
"""

from __future__ import annotations

import logging
import re
import sys
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety.data_audit import dumps_jsonable  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
TABLES = ROOT / "reports" / "tables"
FIGURES = ROOT / "reports" / "figures" / "final"
SCRIPTS = ROOT / "scripts"
OUT_MD = ROOT / "reports" / "47_artifacts_inventory.md"
OUT_JSON = ROOT / "reports" / "tables" / "47_artifacts_inventory.json"

INTERMEDIATE_DIRS = {
    "admin", "exposure_by_id", "40a_range_headers", "samples",
    "39_samples", "39b_samples", "_callejero_viales",
}

SCRIPT_RE = re.compile(r"^(\d+)([a-z]?)_")

# Manual archive list (based on Phase 10 decisions, see D110)
KNOWN_ARCHIVE = {
    "41b_poisson_diagnostics.py": "Bug de fittedvalues (devuelve X·β, no exp(X·β))",
    "41c_phase10_refit.py": "Superseded by 41d (residuales corregidos)",
    "44_phase10_diagnostics_v2.py": "Bootstrap con solo 23 hexes (diseño roto)",
    "44b_phase10_bootstrap_clean.py": "Baseline-shift no corregido; conclusión es 44c",
    "44c_bootstrap_centered.py": "Diagnóstico final: mapa FE inestable, descartado",
}

# Manual producer map for parquets whose producer cannot be inferred
# from the filename prefix. Verified against mtime + script logs.
MANUAL_PRODUCERS = {
    "accidents_clean.parquet": "scripts\\06_phase2_build_clean.py",
    "persons_clean.parquet": "scripts\\06_phase2_build_clean.py",
    "splits.parquet": "scripts\\08_phase3_create_splits.py",
    "baseline_test_predictions.parquet": "scripts\\09_phase3_baselines.py",
    "network_segments.parquet": "scripts\\14_phase4_extract_network.py",
    "accidents_with_segment.parquet": "scripts\\15_phase4_verify_and_join.py",
    "segments_with_imd.parquet": "scripts\\16_phase4_fix_and_diagnose.py",
    "imd_points.parquet": "scripts\\16_phase4_fix_and_diagnose.py",
    "network_callejero.parquet": "scripts\\17c_phase4_fix_callejero.py",
    "network_osm.parquet": "scripts\\18b_phase4_fix_osm.py",
    "accidents_with_network_features.parquet": "scripts\\19b_phase4_refresh_features.py",
    "hdbscan_clusters_all.parquet": "scripts\\31_phase6_hdbscan.py",
    "hdbscan_clusters_injured.parquet": "scripts\\31_phase6_hdbscan.py",
    "hdbscan_persistence_all.parquet": "scripts\\32_phase6_persistence.py",
    "hdbscan_persistence_injured.parquet": "scripts\\32_phase6_persistence.py",
    "network_kde_c.parquet": "scripts\\34_phase7_kde.py",
    "accidents_with_edges.parquet": "scripts\\36_phase7_compare_b.py",
    "network_kde_b.parquet": "scripts\\36_phase7_compare_b.py",
    "network_kde_final.parquet": "scripts\\36b_kde_compare.py",
    "moran_hex250m.parquet": "scripts\\37_phase8_moran.py",
    "moran_barrios.parquet": "scripts\\37_phase8_moran.py",
    "moran_distritos.parquet": "scripts\\37_phase8_moran.py",
    "moran_callejero.parquet": "scripts\\37_phase8_moran.py",
    "exposure_hex250m_monthly.parquet": "scripts\\40d_fix_units.py",
    "panel_hex_month_n_injured.parquet": "scripts\\41_phase10_poisson_baseline.py",
    "hex_risk_fe.parquet": "scripts\\42_phase10_fixed_effects.py",
    "hex_risk_fe_flags.parquet": "scripts\\43_phase10_fe_diagnostics.py",
    "variance_by_year.parquet": "scripts\\44_phase10_diagnostics_v2.py",
    "hex_risk_bootstrap.parquet": "scripts\\44_phase10_diagnostics_v2.py",
    "hex_risk_bootstrap_v2.parquet": "scripts\\44b_phase10_bootstrap_clean.py",
}

log = logging.getLogger("phase11_47")


# ---------------------------------------------------------------------------
# Scanners
# ---------------------------------------------------------------------------

def scan_data_processed() -> list[dict]:
    rows = []
    for p in sorted(DATA.iterdir()):
        if p.name in INTERMEDIATE_DIRS:
            continue
        if p.is_dir():
            files = [f for f in p.rglob("*") if f.is_file()]
            rows.append({
                "path": str(p.relative_to(ROOT)),
                "type": "directory",
                "n_files": len(files),
                "size_kb": round(
                    sum(f.stat().st_size for f in files) / 1024, 1),
                "mtime": datetime.fromtimestamp(p.stat().st_mtime).isoformat()[:19],
            })
            continue
        if p.suffix != ".parquet":
            continue
        try:
            meta = pq.read_metadata(p)
            n_rows, n_cols = meta.num_rows, meta.num_columns
        except Exception:  # noqa: BLE001
            n_rows = n_cols = None
        rows.append({
            "path": str(p.relative_to(ROOT)),
            "type": "parquet",
            "size_kb": round(p.stat().st_size / 1024, 1),
            "n_rows": n_rows,
            "n_cols": n_cols,
            "mtime": datetime.fromtimestamp(p.stat().st_mtime).isoformat()[:19],
        })
    return rows


def scan_tables() -> list[dict]:
    return [{
        "path": str(p.relative_to(ROOT)),
        "size_kb": round(p.stat().st_size / 1024, 1),
        "mtime": datetime.fromtimestamp(p.stat().st_mtime).isoformat()[:19],
    } for p in sorted(TABLES.glob("*.json"))]


def scan_figures() -> list[dict]:
    return [{
        "path": str(p.relative_to(ROOT)),
        "size_kb": round(p.stat().st_size / 1024, 1),
        "mtime": datetime.fromtimestamp(p.stat().st_mtime).isoformat()[:19],
    } for p in sorted(FIGURES.glob("*.png"))]


def scan_scripts() -> list[dict]:
    rows = []
    for p in sorted(SCRIPTS.glob("*.py")):
        m = SCRIPT_RE.match(p.name)
        prefix = m.group(1) if m else None
        letter = (m.group(2) or "") if m else ""
        rows.append({
            "path": str(p.relative_to(ROOT)),
            "name": p.name,
            "prefix": prefix,
            "iter": letter,
            "kind": "primary" if not letter else "iteration",
            "size_kb": round(p.stat().st_size / 1024, 1),
            "mtime": datetime.fromtimestamp(p.stat().st_mtime).isoformat()[:19],
        })
    return rows


# ---------------------------------------------------------------------------
# Cross-reference: producer / consumer
# ---------------------------------------------------------------------------

def read_all_scripts(scripts: list[dict]) -> dict[str, str]:
    contents = {}
    for s in scripts:
        try:
            contents[s["path"]] = (ROOT / s["path"]).read_text(
                encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            contents[s["path"]] = ""
    return contents


def find_producers_consumers(artifacts: list[dict],
                              script_contents: dict) -> dict:
    """For each artifact, list scripts that mention its basename.

    Producer resolution:
      1. MANUAL_PRODUCERS map (parquets without numeric prefix, or with
         dynamic f-string names like moran_{unit}.parquet).
      2. Numeric-prefix heuristic (JSONs like 37_moran.json → script 37).
    """
    out = {}
    for a in artifacts:
        basename = Path(a["path"]).name
        prefix = None
        m = re.match(r"^(\d+)[a-z]?_", basename)
        if m:
            prefix = m.group(1)

        mentioned_in = []
        for s_path, content in script_contents.items():
            if basename in content:
                mentioned_in.append(s_path)

        producer = None
        if basename in MANUAL_PRODUCERS:
            producer = MANUAL_PRODUCERS[basename]
            if producer not in mentioned_in:
                mentioned_in.append(producer)
        elif prefix:
            for s_path in mentioned_in:
                s_name = Path(s_path).name
                sm = re.match(r"^(\d+)[a-z]?_", s_name)
                if sm and sm.group(1) == prefix:
                    producer = s_path
                    break

        consumers = [s for s in mentioned_in if s != producer]
        out[a["path"]] = {
            "basename": basename,
            "prefix": prefix,
            "n_mentions": len(mentioned_in),
            "producer": producer,
            "consumers": consumers,
        }
    return out


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

def build_recommendations(artifacts: dict, scripts: list[dict]) -> dict:
    """Classify each artifact as keep/review/archive/orphan."""
    rec = {"keep": [], "review": [], "archive": [], "orphan_parquets": []}

    # Scripts
    for s in scripts:
        name = s["name"]
        if name in KNOWN_ARCHIVE:
            rec["archive"].append({
                "path": s["path"],
                "reason": KNOWN_ARCHIVE[name],
            })
            continue
        producers = [a for a in artifacts.values()
                     if a["producer"] == s["path"]]
        if not producers:
            rec["review"].append({
                "path": s["path"],
                "reason": "no outputs identified (audit, maps, or refactor)",
            })
        else:
            rec["keep"].append({"path": s["path"], "type": "script"})

    # Parquets
    for path, a in artifacts.items():
        if not path.endswith(".parquet"):
            continue
        if a["producer"] is None:
            rec["orphan_parquets"].append({
                "path": path,
                "reason": "no script produces this file",
            })
        elif not a["consumers"]:
            rec["review"].append({
                "path": path,
                "reason": f"only producer {a['producer']} (no consumers)",
            })
        else:
            rec["keep"].append({"path": path, "type": "parquet"})

    # JSON tables
    for path, a in artifacts.items():
        if not path.endswith(".json"):
            continue
        if a["producer"] is None:
            rec["orphan_parquets"].append({
                "path": path,
                "reason": "no producer script",
            })
        else:
            rec["keep"].append({"path": path, "type": "table"})

    # Figures: all output-only
    for path in artifacts:
        if path.endswith(".png"):
            rec["keep"].append({"path": path, "type": "figure"})

    return rec


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

def render_md(results: dict) -> str:
    md = ["# Phase 11 · Iter 3 — Artifacts inventory\n\n"]

    md.append("## Resumen\n\n")
    n_scripts = len(results["scripts"])
    n_parquets = sum(1 for a in results["data_processed"]
                     if a.get("type") == "parquet")
    n_tables = len(results["tables"])
    n_figures = len(results["figures"])
    md.append(f"- **Scripts**: {n_scripts}\n")
    md.append(f"- **Parquets** (top-level en data/processed/): {n_parquets}\n")
    md.append(f"- **Tablas JSON** (reports/tables/): {n_tables}\n")
    md.append(f"- **Figuras PNG** (reports/figures/final/): {n_figures}\n\n")

    rec = results["recommendations"]

    md.append("## Scripts por archivar\n\n")
    if rec["archive"]:
        md.append("| script | razón |\n")
        md.append("| --- | --- |\n")
        for item in rec["archive"]:
            md.append(f"| `{item['path']}` | {item['reason']} |\n")
    else:
        md.append("_Ninguno_\n")
    md.append("\n")

    md.append("## Parquets huérfanos\n\n")
    if rec["orphan_parquets"]:
        md.append("| path | razón |\n")
        md.append("| --- | --- |\n")
        for item in rec["orphan_parquets"]:
            md.append(f"| `{item['path']}` | {item['reason']} |\n")
    else:
        md.append("_Ninguno_\n")
    md.append("\n")

    md.append("## Ficheros a revisar\n\n")
    if rec["review"]:
        md.append("| path | razón |\n")
        md.append("| --- | --- |\n")
        for item in rec["review"]:
            md.append(f"| `{item['path']}` | {item['reason']} |\n")
    else:
        md.append("_Ninguno_\n")
    md.append("\n")

    md.append("## Scripts / artefactos a mantener\n\n")
    md.append(f"- Scripts: {sum(1 for r in rec['keep'] if r.get('type') == 'script')}\n")
    md.append(f"- Parquets: {sum(1 for r in rec['keep'] if r.get('type') == 'parquet')}\n")
    md.append(f"- Tablas JSON: {sum(1 for r in rec['keep'] if r.get('type') == 'table')}\n")
    md.append(f"- Figuras PNG: {sum(1 for r in rec['keep'] if r.get('type') == 'figure')}\n\n")

    md.append("## Parquets top-level (resumen)\n\n")
    md.append("| path | n_rows | n_cols | size_kb | producer |\n")
    md.append("| --- | ---: | ---: | ---: | --- |\n")
    for a in results["data_processed"]:
        if a.get("type") != "parquet":
            continue
        refs = results["artifact_refs"].get(a["path"], {})
        prod = refs.get("producer")
        prod_short = Path(prod).name if prod else "_orphan_"
        md.append(f"| `{Path(a['path']).name}` | "
                  f"{a.get('n_rows', 'n/a')} | "
                  f"{a.get('n_cols', 'n/a')} | "
                  f"{a.get('size_kb', 'n/a')} | `{prod_short}` |\n")
    md.append("\n")

    md.append("## Próximos pasos (script 48 — refactor)\n\n")
    md.append(
        "1. **Mover scripts archivables** a `scripts/archive/`:\n"
        "   - Los 5 scripts marcados en `archive/`.\n"
        "2. **Mover JSONs huérfanos** a `reports/tables/archive/`.\n"
        "3. **Archivar los parquets del FE descartado**:\n"
        "   - `hex_risk_fe.parquet`, `hex_risk_fe_flags.parquet`,\n"
        "     `hex_risk_bootstrap.parquet`, `hex_risk_bootstrap_v2.parquet`,\n"
        "     `variance_by_year.parquet`.\n"
        "4. **Tests unitarios** (`tests/`) para funciones críticas.\n"
        "5. **environment.yml** con versiones fijadas.\n"
    )
    return "".join(md)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    log.info("Scanning data/processed/ ...")
    data_processed = scan_data_processed()
    log.info(f"  {sum(1 for a in data_processed if a.get('type') == 'parquet')} "
             f"parquets top-level")

    log.info("Scanning reports/tables/ ...")
    tables = scan_tables()
    log.info(f"  {len(tables)} JSON tables")

    log.info("Scanning reports/figures/final/ ...")
    figures = scan_figures()
    log.info(f"  {len(figures)} PNG figures")

    log.info("Scanning scripts/ ...")
    scripts = scan_scripts()
    log.info(f"  {len(scripts)} scripts "
             f"({sum(1 for s in scripts if s['kind'] == 'primary')} primary, "
             f"{sum(1 for s in scripts if s['kind'] == 'iteration')} iteration)")

    log.info("Reading all scripts for cross-reference ...")
    script_contents = read_all_scripts(scripts)

    all_artifacts = data_processed + tables + figures
    log.info("Cross-referencing artifacts → producers/consumers ...")
    refs = find_producers_consumers(all_artifacts, script_contents)

    n_parquet_orphan = sum(
        1 for path, a in refs.items()
        if path.endswith(".parquet") and a["producer"] is None
    )
    log.info(f"  parquets with producer: "
             f"{sum(1 for a in data_processed if a.get('type') == 'parquet') - n_parquet_orphan}")

    log.info("Building recommendations ...")
    rec = build_recommendations(refs, scripts)

    results = {
        "data_processed": data_processed,
        "tables": tables,
        "figures": figures,
        "scripts": scripts,
        "artifact_refs": refs,
        "recommendations": rec,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(dumps_jsonable(results), encoding="utf-8")
    log.info(f"saved {OUT_JSON}")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(render_md(results), encoding="utf-8")
    log.info(f"saved {OUT_MD}")

    log.info("=" * 72)
    log.info(f"Archive: {len(rec['archive'])} scripts")
    log.info(f"Orphans: {len(rec['orphan_parquets'])}")
    log.info(f"Review: {len(rec['review'])}")
    log.info(f"Keep: {len(rec['keep'])}")


if __name__ == "__main__":
    main()