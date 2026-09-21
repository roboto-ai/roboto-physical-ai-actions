from __future__ import annotations

from collections.abc import Iterable

import numpy as np

SCALAR_SUMMARY_KEYS = ("mean", "median", "std", "min", "max", "p05", "p95")


def scalar_summary(values: Iterable[float]) -> dict[str, float]:
    """Per-dataset rollup of a per-episode scalar across episodes."""
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {k: float("nan") for k in SCALAR_SUMMARY_KEYS}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p05": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
    }


def safe_float(x) -> float:
    """Coerce to float, replacing non-finite with NaN."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return f if np.isfinite(f) else float("nan")


def resample_to_time_bins(matrix: np.ndarray, n_bins: int) -> np.ndarray:
    """Nearest-neighbor resample a (T, D) matrix onto `n_bins` normalized-time
    samples, matching HF visualizer's cross-episode variance computation.

    HF uses `srcIdx = min(round(b * (T-1)), T-1)` per bin `b`, so episodes of
    different lengths collapse onto a common time grid without smoothing.
    """
    if matrix.ndim != 2:
        raise ValueError(f"expected 2D array, got shape {matrix.shape}")
    T, D = matrix.shape
    if T < 2:
        return np.repeat(matrix[:1], n_bins, axis=0) if T == 1 else np.zeros((n_bins, D))
    bins = np.arange(n_bins, dtype=float)
    src_idx = np.clip(np.round(bins * (T - 1) / max(1, n_bins - 1)).astype(int), 0, T - 1)
    return matrix[src_idx]


def to_json_serializable(obj):
    """Convert numpy scalars / arrays nested in metric outputs to plain Python."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_serializable(v) for v in obj]
    return obj
