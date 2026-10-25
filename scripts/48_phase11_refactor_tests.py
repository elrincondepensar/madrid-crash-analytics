"""
Script 48 — Phase 11 iter 4: Physical archive + tests + environment.

Operations:

1. ARCHIVE (idempotent — safe to run twice):
   - scripts/archive/         ← 5 deprecated scripts (41b, 41c, 44, 44b, 44c)
   - reports/tables/archive/  ← 6 historical orphan JSONs
   - data/processed/archive/  ← 5 parquets from the discarded FE (D110)
   - scripts/archive/README.md explaining each archive decision.

2. TESTS:
   - tests/conftest.py                    — minimal setup
   - tests/test_io.py                     — 4 unit tests on madrid_safety.io
   - tests/test_data_audit.py             — 2 unit tests on dumps_jsonable
   Only tests on functions already in src/ are written (verifiable).
   Inline functions are left with a TODO for future refactor.

3. ENVIRONMENT:
   - environment.yml with pinned versions.

Does NOT run pytest (user does it after inspection).

Outputs:
  reports/tables/48_refactor_log.json
  reports/48_refactor_log.md
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from madrid_safety.data_audit import dumps_jsonable  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
TABLES = ROOT / "reports" / "tables"
DATA = ROOT / "data" / "processed"
TESTS = ROOT / "tests"

SCRIPTS_ARCHIVE = SCRIPTS / "archive"
TABLES_ARCHIVE = TABLES / "archive"
DATA_ARCHIVE = DATA / "archive"

OUT_JSON = TABLES / "48_refactor_log.json"
OUT_MD = ROOT / "reports" / "48_refactor_log.md"

log = logging.getLogger("phase11_48")


# ---------------------------------------------------------------------------
# Archive lists
# ---------------------------------------------------------------------------

SCRIPTS_TO_ARCHIVE = {
    "41b_poisson_diagnostics.py":
        "Bug de fittedvalues (devuelve X·β, no exp(X·β)). Reemplazado por 41c/41d.",
    "41c_phase10_refit.py":
        "Refit NB2. Superado por 41d (residuales corregidos).",
    "44_phase10_diagnostics_v2.py":
        "Bootstrap FE con solo 23 hexes (diseño roto). Superado por 44b.",
    "44b_phase10_bootstrap_clean.py":
        "Bootstrap limpio (400 hexes × 20 iter) pero sin corregir baseline-shift.",
    "44c_bootstrap_centered.py":
        "Diagnóstico final: mapa FE inestable, descartado en D110.",
}

TABLES_TO_ARCHIVE = {
    "01_schema_samples.json":
        "Muestra de schema de iter 1, previa a la consolidación en 01_v2_audit.",
    "01_url_availability.json":
        "Auditoría de URLs de iter 1, superada por 01_v2_audit.",
    "17_callejero_load_error.json":
        "Error de carga del SHP callejero (primera iteración, antes de 17b).",
    "17_callejero_meta.json":
        "Metadata previa a 17b_callejero_structure.",
    "17_imd_fixed.json":
        "Chequeo IMD previo a 17b_imd_check.",
    "28_opcion_c_bart.json":
        "Ejecución temprana de Opción C con BART antes de descartar BART (D096).",
}

PARQUETS_TO_ARCHIVE = {
    "hex_risk_fe.parquet":
        "Coeficientes FE por hex. Descartado por rank-deficiency (D110).",
    "hex_risk_fe_flags.parquet":
        "Flags de calidad del FE. Descartado con el mapa (D110).",
    "hex_risk_bootstrap.parquet":
        "Bootstrap FE diseño roto (23 hexes). Descartado.",
    "hex_risk_bootstrap_v2.parquet":
        "Bootstrap FE limpio pero conclusión es inestabilidad (44c). Descartado.",
    "variance_by_year.parquet":
        "Descomposición de varianza por año. Documentada en D110, parquet no reutilizable.",
}


# ---------------------------------------------------------------------------
# Archive operations
# ---------------------------------------------------------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_move(src: Path, dst: Path) -> dict:
    """Move a file, idempotent. Returns status dict."""
    if not src.exists():
        return {"src": str(src.relative_to(ROOT)),
                "status": "not_found"}
    if dst.exists():
        return {"src": str(src.relative_to(ROOT)),
                "dst": str(dst.relative_to(ROOT)),
                "status": "already_archived"}
    shutil.move(str(src), str(dst))
    return {"src": str(src.relative_to(ROOT)),
            "dst": str(dst.relative_to(ROOT)),
            "status": "moved"}


def archive_files(mapping: dict, src_dir: Path, dst_dir: Path,
                  kind: str) -> list[dict]:
    ensure_dir(dst_dir)
    rows = []
    for fname, reason in mapping.items():
        src = src_dir / fname
        dst = dst_dir / fname
        r = safe_move(src, dst)
        r["kind"] = kind
        r["reason"] = reason
        rows.append(r)
        log.info(f"  [{r['status']:17s}] {fname}")
    return rows


def write_archive_readme() -> None:
    """Human-readable explanation of what's in each archive/ subdir."""
    # scripts/archive/README.md
    sc_md = ["# Archived scripts\n\n",
             "These scripts are kept for reference only. They either have "
             "known bugs or have been superseded by later iterations.\n\n",
             "| script | reason |\n| --- | --- |\n"]
    for name, reason in SCRIPTS_TO_ARCHIVE.items():
        sc_md.append(f"| `{name}` | {reason} |\n")
    (SCRIPTS_ARCHIVE / "README.md").write_text("".join(sc_md),
                                                 encoding="utf-8")

    # reports/tables/archive/README.md
    tb_md = ["# Archived tables\n\n",
             "Historical audit JSONs from failed or superseded iterations. "
             "Not consumed by any active script.\n\n",
             "| table | reason |\n| --- | --- |\n"]
    for name, reason in TABLES_TO_ARCHIVE.items():
        tb_md.append(f"| `{name}` | {reason} |\n")
    (TABLES_ARCHIVE / "README.md").write_text("".join(tb_md),
                                                encoding="utf-8")

    # data/processed/archive/README.md
    dt_md = ["# Archived parquets\n\n",
             "Outputs from the discarded fixed-effects analysis (D110). "
             "The FE map was shown to be unstable via bootstrap; the "
             "parquets are not consumed by any active script.\n\n",
             "| parquet | reason |\n| --- | --- |\n"]
    for name, reason in PARQUETS_TO_ARCHIVE.items():
        dt_md.append(f"| `{name}` | {reason} |\n")
    (DATA_ARCHIVE / "README.md").write_text("".join(dt_md),
                                              encoding="utf-8")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

CONFTEST_PY = '''"""Pytest configuration: add src/ to sys.path."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
'''


TEST_IO_PY = '''"""Tests for madrid_safety.io (geometry loading helpers)."""
import geopandas as gpd
from shapely.geometry import Point, Polygon

from madrid_safety import io as ms_io


def test_detect_crs_utm():
    """Madrid UTM bounds should be detected as EPSG:25830."""
    g = Point(440000, 4475000)
    assert ms_io.detect_crs_from_bounds([g]) == "EPSG:25830"


def test_detect_crs_wgs84():
    """WGS84 (lon/lat) bounds should be detected as EPSG:4326."""
    g = Point(-3.7, 40.4)
    assert ms_io.detect_crs_from_bounds([g]) == "EPSG:4326"


def test_detect_crs_empty_defaults_4326():
    """Empty iterable defaults to EPSG:4326."""
    assert ms_io.detect_crs_from_bounds([]) == "EPSG:4326"


def test_repair_geometries_fixes_bowtie():
    """A self-intersecting bowtie polygon is repaired to a valid geometry."""
    invalid = Polygon([(0, 0), (1, 1), (1, 0), (0, 1)])
    assert not invalid.is_valid
    gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[invalid],
                            crs="EPSG:25830")
    out = ms_io.repair_geometries(gdf, label="test")
    assert len(out) == 1
    assert out.geometry.is_valid.all()
'''


TEST_DATA_AUDIT_PY = '''"""Tests for madrid_safety.data_audit."""
import json
import numpy as np

from madrid_safety.data_audit import dumps_jsonable


def test_dumps_jsonable_handles_numpy_scalars():
    """np.int64 / np.float64 should serialize cleanly."""
    payload = {"a": np.int64(5), "b": np.float64(3.14)}
    parsed = json.loads(dumps_jsonable(payload))
    assert parsed["a"] == 5
    assert abs(parsed["b"] - 3.14) < 1e-9


def test_dumps_jsonable_handles_nan():
    """np.nan and float('nan') should become null in JSON."""
    payload = {"x": np.nan, "y": float("nan")}
    parsed = json.loads(dumps_jsonable(payload))
    assert parsed["x"] is None
    assert parsed["y"] is None
'''


def write_tests() -> list[dict]:
    ensure_dir(TESTS)
    files = {
        "conftest.py": CONFTEST_PY,
        "test_io.py": TEST_IO_PY,
        "test_data_audit.py": TEST_DATA_AUDIT_PY,
    }
    rows = []
    for name, content in files.items():
        p = TESTS / name
        if p.exists():
            rows.append({"path": str(p.relative_to(ROOT)),
                         "status": "already_exists"})
            continue
        p.write_text(content, encoding="utf-8")
        rows.append({"path": str(p.relative_to(ROOT)),
                     "status": "created"})
        log.info(f"  created tests/{name}")
    return rows


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

ENVIRONMENT_YML = '''# Conda environment for madrid-road-safety.
# Reproduce with: conda env create -f environment.yml
name: sendai
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - numpy>=1.26,<2.0
  - pandas>=2.0,<3.0
  - scipy>=1.11
  - scikit-learn>=1.7,<2.0
  - statsmodels>=0.14
  - pyarrow>=15.0
  - geopandas>=0.14
  - shapely>=2.0
  - pyproj>=3.6
  - libpysal>=4.10
  - esda>=2.5
  - hdbscan>=0.8.33
  - lightgbm>=4.0
  - catboost>=1.2
  - requests>=2.31
  - lxml>=5.0
  - openpyxl>=3.1
  - osmnx>=1.9
  - matplotlib>=3.8
  - joblib>=1.3
  - pytest>=8.0
  # Optional / experimental (installed but not critical for core pipeline):
  - pymc>=5.10
  - pymc-bart>=0.6
  - imblearn>=0.12
'''


def write_environment() -> dict:
    p = ROOT / "environment.yml"
    if p.exists():
        return {"path": str(p.relative_to(ROOT)),
                "status": "already_exists"}
    p.write_text(ENVIRONMENT_YML, encoding="utf-8")
    log.info("  created environment.yml")
    return {"path": str(p.relative_to(ROOT)), "status": "created"}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def render_md(result: dict) -> str:
    md = ["# Phase 11 · Iter 4 — Refactor log\n\n"]
    md.append(f"_Executed: {result['executed_at']}_\n\n")

    md.append("## Archive — scripts\n\n")
    md.append("| src | status |\n| --- | --- |\n")
    for r in result["archived"]["scripts"]:
        md.append(f"| `{r['src']}` | {r['status']} |\n")
    md.append("\n")

    md.append("## Archive — tables\n\n")
    md.append("| src | status |\n| --- | --- |\n")
    for r in result["archived"]["tables"]:
        md.append(f"| `{r['src']}` | {r['status']} |\n")
    md.append("\n")

    md.append("## Archive — parquets\n\n")
    md.append("| src | status |\n| --- | --- |\n")
    for r in result["archived"]["parquets"]:
        md.append(f"| `{r['src']}` | {r['status']} |\n")
    md.append("\n")

    md.append("## Tests creados\n\n")
    for r in result["tests"]:
        md.append(f"- `{r['path']}`: {r['status']}\n")
    md.append("\n")

    md.append("## Environment\n\n")
    md.append(f"- `{result['environment']['path']}`: "
              f"{result['environment']['status']}\n\n")

    md.append("## Verificación manual (comandos)\n\n")
    md.append("```bash\n")
    md.append("conda activate sendai\n")
    md.append("pytest tests/ -v\n")
    md.append("```\n\n")

    md.append("## Notas\n\n")
    md.append(
        "- Los `scripts/archive/README.md`, `reports/tables/archive/README.md` "
        "y `data/processed/archive/README.md` documentan la razón de cada "
        "archivo.\n"
        "- Los scripts archivados siguen siendo válidos como referencia. "
        "Están fuera del flujo activo pero son legibles.\n"
        "- Los 5 parquets archivados corresponden al FE descartado en D110. "
        "Los mapas de Fase 11 NO los usan (usan `moran_*.parquet` y N4 "
        "residuals).\n"
        "- Los tests solo cubren funciones que ya están en `src/`. Para "
        "las funciones inline en scripts (`_worst_lesividad`, "
        "`parse_number_with_range_sanity`, `year_to_split`, etc.), el "
        "test se pospone a Fase 12 o a un refactor futuro de extracción "
        "a `src/`.\n"
        "- `environment.yml` tiene versiones mínimas (>=), no pin exacto. "
        "Si se necesita reproducibilidad bit a bit, usar `conda env export` "
        "después de instalar.\n"
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

    log.info("=== Archive: scripts ===")
    archived_scripts = archive_files(SCRIPTS_TO_ARCHIVE, SCRIPTS,
                                       SCRIPTS_ARCHIVE, kind="script")

    log.info("=== Archive: tables ===")
    archived_tables = archive_files(TABLES_TO_ARCHIVE, TABLES,
                                      TABLES_ARCHIVE, kind="table")

    log.info("=== Archive: parquets ===")
    archived_parquets = archive_files(PARQUETS_TO_ARCHIVE, DATA,
                                        DATA_ARCHIVE, kind="parquet")

    log.info("=== Writing archive READMEs ===")
    write_archive_readme()
    log.info("  READMEs creados")

    log.info("=== Creating tests ===")
    test_rows = write_tests()

    log.info("=== Creating environment.yml ===")
    env_row = write_environment()

    result = {
        "executed_at": datetime.now().isoformat()[:19],
        "archived": {
            "scripts": archived_scripts,
            "tables": archived_tables,
            "parquets": archived_parquets,
        },
        "tests": test_rows,
        "environment": env_row,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(dumps_jsonable(result), encoding="utf-8")
    log.info(f"saved {OUT_JSON}")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(render_md(result), encoding="utf-8")
    log.info(f"saved {OUT_MD}")

    # Summary
    n_moved = sum(1 for r in archived_scripts + archived_tables
                  + archived_parquets if r["status"] == "moved")
    n_already = sum(1 for r in archived_scripts + archived_tables
                    + archived_parquets if r["status"] == "already_archived")
    n_missing = sum(1 for r in archived_scripts + archived_tables
                    + archived_parquets if r["status"] == "not_found")
    n_tests_new = sum(1 for r in test_rows if r["status"] == "created")

    log.info("=" * 72)
    log.info(f"Archived files: moved={n_moved}, already={n_already}, "
             f"missing={n_missing}")
    log.info(f"Tests created: {n_tests_new}")
    log.info(f"Environment: {env_row['status']}")
    log.info("Next: run `pytest tests/ -v` to verify the tests pass.")


if __name__ == "__main__":
    main()