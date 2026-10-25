"""
Reusable functions for auditing open data sources.
No hardcoded schemas: everything is discovered from the source.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests

TIMEOUT = 60
USER_AGENT = "madrid-road-safety-audit/0.1 (contact: data-team)"


def fetch_head(url: str, n_rows: int = 5) -> pd.DataFrame:
    """Download the first n_rows of a CSV without downloading the whole file."""
    headers = {"User-Agent": USER_AGENT}
    # Use Range request to avoid pulling large files
    resp = requests.get(url, headers=headers, stream=True, timeout=TIMEOUT)
    resp.raise_for_status()

    # Read a limited chunk to infer schema
    chunk = next(resp.iter_content(chunk_size=1024 * 1024))
    df = pd.read_csv(io.BytesIO(chunk), nrows=n_rows, low_memory=False)
    return df


def describe_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Return a serialisable summary of dtypes and null counts."""
    return {
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "null_counts": df.isna().sum().to_dict(),
        "n_rows_sampled": len(df),
    }


def check_url_availability(url: str) -> dict[str, Any]:
    """Return status code and content-type without downloading the body."""
    headers = {"User-Agent": USER_AGENT}
    try:
        resp = requests.head(url, headers=headers, timeout=TIMEOUT, allow_redirects=True)
        return {
            "url": url,
            "status_code": resp.status_code,
            "content_type": resp.headers.get("Content-Type", ""),
            "content_length": resp.headers.get("Content-Length", ""),
        }
    except Exception as exc:
        return {"url": url, "error": str(exc)}


def _to_jsonable(obj: Any) -> Any:
    """
    Recursively convert numpy/pandas scalars and pandas sentinels into plain
    Python types that json.dumps can serialize.

    Handles:
      - numpy integer / floating / bool scalars -> int / float / bool
      - pandas NA, NaT, NaN, Inf -> None
      - dict keys that are numpy scalars -> Python int/str
      - nested dict/list/tuple
    """
    if isinstance(obj, dict):
        return {
            _json_key(k): _to_jsonable(v)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if obj is pd.NA or obj is pd.NaT:
        return None
    if isinstance(obj, float) and (
        obj != obj or obj == float("inf") or obj == float("-inf")
    ):
        return None
    # numpy scalar (np.int64, np.float64, np.bool_, etc.)
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        try:
            return obj.item()
        except (AttributeError, ValueError):
            pass
    return obj


def _json_key(k: Any) -> Any:
    """Return a JSON-valid dict key: str, int, float, bool or None."""
    if isinstance(k, (str, int, float, bool)) or k is None:
        return k
    if hasattr(k, "item"):
        try:
            v = k.item()
            if isinstance(v, (str, int, float, bool)) or v is None:
                return v
        except (AttributeError, ValueError):
            pass
    return str(k)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_to_jsonable(obj), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

def dumps_jsonable(obj: Any, **kwargs) -> str:
    """
    json.dumps with automatic conversion of numpy/pandas scalars,
    pandas sentinels and non-primitive dict keys to JSON-serializable types.
    Use this instead of json.dumps anywhere the payload may carry
    numpy/pandas types (groupby results, Int64 columns, etc.).
    """
    kwargs.setdefault("ensure_ascii", False)
    return json.dumps(_to_jsonable(obj), **kwargs)

def save_markdown_report(sections: dict[str, str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Source Audit Report\n"]
    for title, body in sections.items():
        lines.append(f"## {title}\n")
        lines.append(body.strip() + "\n")
    path.write_text("\n".join(lines), encoding="utf-8")

def read_madrid_csv(url: str, nrows: int | None = None,
                    usecols: list[str] | None = None,
                    sep: str = ";") -> pd.DataFrame:
    """Read a CSV from datos.madrid.es handling BOM / latin-1 and ';' separator."""
    headers = {"User-Agent": "madrid-road-safety/0.1"}
    r = requests.get(url, headers=headers, timeout=120)
    r.raise_for_status()
    last_exc = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(
                io.BytesIO(r.content), sep=sep, encoding=enc,
                nrows=nrows, usecols=usecols, low_memory=False,
            )
        except UnicodeDecodeError as exc:
            last_exc = exc
    raise last_exc  # type: ignore[misc]