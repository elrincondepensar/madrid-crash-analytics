"""Tests for madrid_safety.data_audit."""
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
