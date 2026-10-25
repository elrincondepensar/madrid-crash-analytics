"""
Phase 1 – Iteration 2: discover real resources via CKAN API and parse CSVs correctly.
Run from the repository root:
    python scripts/01_audit_sources_v2.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from madrid_safety.data_audit import (
    describe_dataframe,
    save_json,
    save_markdown_report,
)

TIMEOUT = 60
USER_AGENT = "madrid-road-safety-audit/0.2"

CKAN_API = "https://datos.madrid.es/api/3/action/package_show"

DATASETS = {
    "accidents": "300228-0-accidentes-trafico-detalle",
    "permanent_counts": "300233-0-aforo-trafico-permanentes",
    "traffic_history": "208627-0-transporte-ptomedida-historico",
    "street_directory": "213605-0-callejero-oficial-madrid",
}

REPORTS_DIR = Path("reports/tables")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def get_ckan_resources(dataset_id: str) -> list[dict[str, Any]]:
    """Return the list of resources for a dataset via the CKAN API."""
    url = f"{CKAN_API}?id={dataset_id}"
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    resp.raise_for_status()
    payload = resp.json()
    if not payload.get("success"):
        raise RuntimeError(f"CKAN API error for {dataset_id}: {payload}")
    return payload["result"].get("resources", [])


def read_csv_sample(url: str, n_rows: int = 10) -> pd.DataFrame:
    """Read a small sample of a CSV with the correct separator and encoding."""
    resp = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=TIMEOUT,
        stream=True,
    )
    resp.raise_for_status()
    # Read only the first chunk to avoid downloading huge files
    chunk = next(resp.iter_content(chunk_size=2 * 1024 * 1024))
    df = pd.read_csv(
        io.BytesIO(chunk),
        sep=";",
        encoding="latin-1",
        nrows=n_rows,
        low_memory=False,
    )
    return df


def audit_dataset(name: str, dataset_id: str) -> dict[str, Any]:
    """Discover resources and audit the first readable CSV of each dataset."""
    result = {"dataset": name, "dataset_id": dataset_id, "resources": []}
    try:
        resources = get_ckan_resources(dataset_id)
    except Exception as exc:
        result["error"] = str(exc)
        return result

    csv_resources = [
        r for r in resources
        if r.get("format", "").upper() == "CSV"
        or r.get("url", "").lower().endswith(".csv")
    ]

    for res in csv_resources:
        res_info = {
            "name": res.get("name", ""),
            "url": res.get("url", ""),
            "format": res.get("format", ""),
            "last_modified": res.get("last_modified", ""),
        }
        try:
            df = read_csv_sample(res["url"])
            res_info["schema"] = describe_dataframe(df)
            # Specific check for accidents: uniqueness signal
            if name == "accidents" and "num_expediente" in df.columns:
                res_info["num_expediente_sample"] = {
                    "n_rows": len(df),
                    "n_unique": int(df["num_expediente"].nunique()),
                    "n_null": int(df["num_expediente"].isna().sum()),
                }
            res_info["status"] = "ok"
        except Exception as exc:
            res_info["status"] = "error"
            res_info["error"] = str(exc)
        result["resources"].append(res_info)

    return result


def main() -> None:
    audit_results = {}
    for name, dataset_id in DATASETS.items():
        print(f"Auditing {name} ...")
        audit_results[name] = audit_dataset(name, dataset_id)

    save_json(audit_results, REPORTS_DIR / "01_v2_audit.json")

    # Build markdown report
    lines = []
    for name, res in audit_results.items():
        lines.append(f"## {name}\n")
        if "error" in res:
            lines.append(f"- **ERROR discovering resources**: {res['error']}\n")
            continue
        ok_resources = [r for r in res["resources"] if r.get("status") == "ok"]
        lines.append(f"- CSV resources found: {len(res['resources'])} (readable: {len(ok_resources)})\n")
        for r in res["resources"][:5]:  # limit output
            lines.append(f"  - `{r['name'][:60]}` | {r['url'][:80]}... | status: {r.get('status')}")
            if r.get("status") == "ok":
                lines.append(f"    - columns: {', '.join(r['schema']['columns'][:10])}...")
                if "num_expediente_sample" in r:
                    lines.append(f"    - num_expediente (sample): {r['num_expediente_sample']}")
            elif r.get("error"):
                lines.append(f"    - error: {r['error'][:120]}")
        lines.append("")

    report_sections = {
        "Audit v2 – CKAN discovery and corrected parsing": "\n".join(lines),
        "Pending verifications": (
            "- Confirm full uniqueness of `num_expediente` across all years (Phase 2).\n"
            "- Quantify valid UTM coordinates in the full accidents dataset.\n"
            "- Confirm granularity of traffic history (point vs. segment).\n"
            "- Confirm whether street directory includes segments or only vials."
        ),
    }
    save_markdown_report(report_sections, Path("reports/01_source_audit_v2.md"))

    print("Audit v2 complete. Check reports/01_source_audit_v2.md and reports/tables/01_v2_audit.json")


if __name__ == "__main__":
    main()