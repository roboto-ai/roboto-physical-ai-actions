"""Converter-oracle parity tests for the live runtime's StreamBuffer.

The byte-parity north star is that the live runtime selects the *same*
sample the offline converter would. The converter aligns streams with
pandas ``merge_asof`` (``alignment._merge_hold`` / ``_merge_nearest``); the
runtime aligns with ``StreamBuffer.sample``. These tests feed identical
``(timestamp, value)`` streams and reference grids through both paths and
assert they pick the same value at every grid point.

This is the cross-check the ``test_stream_buffer`` unit tests cannot give:
those hand-code the expected value, so a divergence that is wrong in *both*
the implementation and the expectation sails through. Here the converter's
real ``merge_asof`` is the oracle, so the equidistant-tie and tolerance
boundary semantics are pinned to the converter rather than to our beliefs
about it. (This is what would have caught the nearest tie-break break.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from roboto_to_lerobot.alignment import merge_onto_timeline
from roboto_to_lerobot.contract_utils import AlignSpec
from roboto_to_lerobot.runtime.stream_buffer import StreamBuffer


def _converter_pick(base_ts, src_ts, src_val, method, tolerance_ms):
    """Value the converter joins onto each grid point (NaN where unmatched)."""
    base = pd.DataFrame({"timestamp": np.asarray(base_ts, dtype=np.int64)})
    src = pd.DataFrame(
        {
            "timestamp": np.asarray(src_ts, dtype=np.int64),
            "v": np.asarray(src_val, dtype=np.float64),
        }
    )
    merged = merge_onto_timeline(
        base, src, AlignSpec(method=method, tolerance_ms=tolerance_ms), ["v"]
    )
    return merged["v"].to_numpy()


def _runtime_pick(base_ts, src_ts, src_val, method, tolerance_ns, capacity):
    """Value StreamBuffer samples at each grid point (NaN where it returns None).

    All source samples are pushed before sampling so the buffer sees the same
    full stream merge_asof does — this compares the *alignment semantics*, not
    the online/causal arrival behaviour exercised elsewhere.
    """
    buf = StreamBuffer(method=method, tolerance_ns=tolerance_ns, capacity=capacity)
    for ts, v in zip(src_ts, src_val, strict=True):
        buf.push(int(ts), float(v))
    out = []
    for now in base_ts:
        sampled = buf.sample(int(now))
        out.append(np.nan if sampled is None else float(sampled))
    return np.asarray(out, dtype=np.float64)


def _assert_same(conv, rt, base_ts):
    assert conv.shape == rt.shape
    for i, ts in enumerate(base_ts):
        if np.isnan(conv[i]):
            assert np.isnan(rt[i]), f"grid[{i}]={ts}: converter=NaN runtime={rt[i]}"
        else:
            assert conv[i] == rt[i], (
                f"grid[{i}]={ts}: converter={conv[i]} runtime={rt[i]}"
            )


@pytest.mark.parametrize("method", ["hold", "nearest"])
@pytest.mark.parametrize("seed", range(50))
def test_stream_buffer_matches_merge_asof(method, seed):
    rng = np.random.default_rng(seed)

    n_src = int(rng.integers(1, 30))
    src_ts = np.unique(np.sort(rng.integers(0, 2_000_000_000, size=n_src)))
    src_val = rng.normal(size=len(src_ts))

    n_base = int(rng.integers(1, 30))
    # Grid spans a little before and after the stream so misses on both ends,
    # and tolerance-bounded gaps, are exercised.
    base_ts = np.sort(rng.integers(-200_000_000, 2_200_000_000, size=n_base))

    tolerance_ms = int(rng.choice([20, 50, 100, 500, 5000]))
    tolerance_ns = tolerance_ms * 1_000_000

    conv = _converter_pick(base_ts, src_ts, src_val, method, tolerance_ms)
    rt = _runtime_pick(
        base_ts, src_ts, src_val, method, tolerance_ns, capacity=len(src_ts) + 1
    )
    _assert_same(conv, rt, base_ts)


def test_oracle_nearest_equidistant_tie_picks_earlier():
    # Grid point exactly between two samples: merge_asof(nearest) breaks the
    # tie toward the earlier timestamp; the runtime must agree.
    base_ts = [150_000_000]
    src_ts = [100_000_000, 200_000_000]
    src_val = [1.0, 2.0]
    conv = _converter_pick(base_ts, src_ts, src_val, "nearest", 1000)
    rt = _runtime_pick(base_ts, src_ts, src_val, "nearest", 1000 * 1_000_000, capacity=8)
    assert conv[0] == rt[0] == 1.0


@pytest.mark.parametrize("method", ["hold", "nearest"])
def test_oracle_exact_tolerance_boundary(method):
    # A gap exactly equal to the tolerance is a match on both sides (inclusive);
    # one nanosecond more is a miss on both sides.
    src_ts = [100_000_000]
    src_val = [7.0]
    tol_ms = 50
    tol_ns = tol_ms * 1_000_000
    at_boundary = [src_ts[0] + tol_ns]
    past_boundary = [src_ts[0] + tol_ns + 1]

    conv_at = _converter_pick(at_boundary, src_ts, src_val, method, tol_ms)
    rt_at = _runtime_pick(at_boundary, src_ts, src_val, method, tol_ns, capacity=4)
    _assert_same(conv_at, rt_at, at_boundary)
    assert conv_at[0] == 7.0  # the gap == tolerance still matches

    conv_past = _converter_pick(past_boundary, src_ts, src_val, method, tol_ms)
    rt_past = _runtime_pick(past_boundary, src_ts, src_val, method, tol_ns, capacity=4)
    _assert_same(conv_past, rt_past, past_boundary)
    assert np.isnan(conv_past[0])  # one ns past tolerance is a miss
