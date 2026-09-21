# roboto_to_lerobot/alignment.py
# -----------------------------------------------------------------------------
# Pluggable time-alignment strategies for merging data streams onto a
# reference timeline.  Each strategy is a thin wrapper around pandas
# operations and respects a caller-supplied tolerance.
# -----------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
import pandas as pd

from .contract_utils import AlignSpec
from .logger import logger


def merge_onto_timeline(
    base_df: pd.DataFrame,
    source_df: pd.DataFrame,
    spec: AlignSpec,
    value_columns: list[str],
) -> pd.DataFrame:
    """Merge *source_df* onto *base_df* using the strategy in *spec*.

    Parameters
    ----------
    base_df:
        DataFrame whose ``"timestamp"`` column defines the output cadence.
    source_df:
        DataFrame with ``"timestamp"`` plus one or more data columns.
    spec:
        ``AlignSpec`` controlling method and tolerance.
    value_columns:
        Column names in *source_df* (after any renaming) to carry through
        the merge.  ``"timestamp"`` is always included automatically.

    Returns
    -------
    pd.DataFrame
        *base_df* with the requested *value_columns* joined in.
    """
    cols = ["timestamp"] + [c for c in value_columns if c != "timestamp"]
    src = source_df[cols].copy().sort_values("timestamp")

    tolerance_ns: int | None = None
    if spec.tolerance_ms is not None:
        tolerance_ns = int(spec.tolerance_ms * 1_000_000)  # ms → ns

    dispatch = {
        "hold": _merge_hold,
        "nearest": _merge_nearest,
        "linear": _merge_linear,
        "none": _merge_none,
    }

    fn = dispatch.get(spec.method)
    if fn is None:
        raise ValueError(f"Unknown alignment method: '{spec.method}'")

    logger.debug(
        "merge_onto_timeline: method=%s, tolerance_ms=%s, columns=%s",
        spec.method,
        spec.tolerance_ms,
        value_columns,
    )
    return fn(base_df, src, value_columns, tolerance_ns)


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------


def _merge_hold(
    base: pd.DataFrame,
    src: pd.DataFrame,
    value_columns: list[str],
    tolerance_ns: int | None,
) -> pd.DataFrame:
    """Backward as-of join (last observation carried forward)."""
    kw = {"on": "timestamp", "direction": "backward"}
    if tolerance_ns is not None:
        kw["tolerance"] = tolerance_ns  # int64 nanoseconds for int64 timestamp column
    return pd.merge_asof(base, src, **kw)


def _merge_nearest(
    base: pd.DataFrame,
    src: pd.DataFrame,
    value_columns: list[str],
    tolerance_ns: int | None,
) -> pd.DataFrame:
    """Nearest-neighbour join (closest sample in either direction)."""
    kw = {"on": "timestamp", "direction": "nearest"}
    if tolerance_ns is not None:
        kw["tolerance"] = tolerance_ns  # int64 nanoseconds for int64 timestamp column
    return pd.merge_asof(base, src, **kw)


def _merge_linear(
    base: pd.DataFrame,
    src: pd.DataFrame,
    value_columns: list[str],
    tolerance_ns: int | None,
) -> pd.DataFrame:
    """Linear interpolation between the two bracketing source samples.

    1. Backward as-of join  → value *before* each reference timestamp.
    2. Forward  as-of join  → value *after*  each reference timestamp.
    3. Lerp between the two using the fractional time position.

    If only one side is available (start/end of stream), that side's value
    is used as-is (i.e. degrades to hold).  Tolerance is applied to *both*
    sides: if either bracket is farther than ``tolerance_ns`` the result is
    NaN.
    """
    data_cols = [c for c in value_columns if c != "timestamp"]

    # Suffixed copies so we can join both sides without collision
    bw_rename = {c: f"{c}__bw" for c in data_cols}
    fw_rename = {c: f"{c}__fw" for c in data_cols}

    src_bw = src.rename(columns=bw_rename)
    src_fw = src.rename(columns=fw_rename)

    # Also carry the source timestamp so we can compute the lerp fraction
    src_bw = src_bw.copy()
    src_bw["__ts_bw"] = src_bw["timestamp"]
    src_fw = src_fw.copy()
    src_fw["__ts_fw"] = src_fw["timestamp"]

    kw_bw = {"on": "timestamp", "direction": "backward"}
    kw_fw = {"on": "timestamp", "direction": "forward"}
    if tolerance_ns is not None:
        # int64 nanoseconds for int64 timestamp column
        kw_bw["tolerance"] = tolerance_ns
        kw_fw["tolerance"] = tolerance_ns

    merged = pd.merge_asof(base, src_bw, **kw_bw)
    merged = pd.merge_asof(merged, src_fw, **kw_fw)

    # Compute lerp fraction: 0.0 at backward sample, 1.0 at forward sample
    span = (merged["__ts_fw"] - merged["__ts_bw"]).astype(float)
    frac = np.where(
        span == 0,
        0.0,
        (merged["timestamp"].astype(float) - merged["__ts_bw"].astype(float)) / span,
    )

    for col in data_cols:
        bw_col = f"{col}__bw"
        fw_col = f"{col}__fw"
        bw_vals = merged[bw_col]
        fw_vals = merged[fw_col]

        # Where both sides are present, lerp; otherwise fall back to whichever exists
        both_ok = bw_vals.notna() & fw_vals.notna()
        merged[col] = np.where(
            both_ok,
            bw_vals * (1 - frac) + fw_vals * frac,
            np.where(bw_vals.notna(), bw_vals, fw_vals),
        )

        merged = merged.drop(columns=[bw_col, fw_col])

    merged = merged.drop(columns=["__ts_bw", "__ts_fw"])
    return merged


def _merge_none(
    base: pd.DataFrame,
    src: pd.DataFrame,
    value_columns: list[str],
    tolerance_ns: int | None,
) -> pd.DataFrame:
    """Exact-match only – no interpolation at all."""
    return base.merge(src, on="timestamp", how="left")

