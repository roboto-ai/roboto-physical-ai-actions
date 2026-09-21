"""Unit tests for :mod:`runtime.stream_buffer`.

The buffer mirrors converter alignment semantics in event-driven form;
these tests pin the boundary cases that the live adapter relies on
(stale-stream detection, clock rewinds, overflow behaviour) so a future
refactor that breaks them fails loudly rather than feeding silently
wrong observations to a policy.
"""

from __future__ import annotations

import pytest
from roboto_to_lerobot.runtime.stream_buffer import (
    DEFAULT_CAPACITY,
    StreamBuffer,
    auto_bound_tolerance_ns,
)

_NS_PER_MS = 1_000_000


# ----------------------------------------------------------------------------
# auto_bound_tolerance_ns
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fps, expected_ns",
    [
        # 2/fps drives the bound for slow streams; 50 ms floor for fast ones.
        (10, 200_000_000),   # 2/10 = 200 ms > 50 ms floor
        (30, 66_666_666),    # 2/30 ≈ 66.67 ms > 50 ms floor
        (100, 50_000_000),   # 2/100 = 20 ms < 50 ms floor → floor wins
    ],
)
def test_auto_bound_tolerance_none_uses_floor_or_two_over_fps(fps, expected_ns):
    assert auto_bound_tolerance_ns(None, fps) == expected_ns


def test_auto_bound_tolerance_positive_converts_ms_to_ns():
    assert auto_bound_tolerance_ns(25, fps=30) == 25 * _NS_PER_MS


def test_auto_bound_tolerance_rejects_negative_ms():
    with pytest.raises(ValueError):
        auto_bound_tolerance_ns(-1, fps=30)


def test_auto_bound_tolerance_rejects_zero_ms():
    # 0 is no longer a valid AlignSpec.tolerance_ms value (None is the
    # unlimited sentinel now) — a stray 0 must fail loudly, not silently
    # get treated as "unlimited" the way it used to.
    with pytest.raises(ValueError):
        auto_bound_tolerance_ns(0, fps=30)


def test_auto_bound_tolerance_requires_positive_fps_only_when_unlimited():
    # fps is only consulted when tolerance_ms is None
    assert auto_bound_tolerance_ns(10, fps=0) == 10 * _NS_PER_MS
    with pytest.raises(ValueError):
        auto_bound_tolerance_ns(None, fps=0)


# ----------------------------------------------------------------------------
# Constructor refusals
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["linear", "none"])
def test_init_refuses_deferred_methods(method):
    with pytest.raises(ValueError) as exc:
        StreamBuffer(method=method, tolerance_ns=10 * _NS_PER_MS)
    assert method in str(exc.value)


def test_init_refuses_unknown_method():
    with pytest.raises(ValueError):
        StreamBuffer(method="bogus", tolerance_ns=10 * _NS_PER_MS)


def test_init_refuses_zero_tolerance():
    # Callers MUST pass a positive tolerance — auto_bound_tolerance_ns is
    # the bridge from contract tolerance_ms=None (unlimited).
    with pytest.raises(ValueError):
        StreamBuffer(method="hold", tolerance_ns=0)


def test_init_refuses_zero_capacity():
    with pytest.raises(ValueError):
        StreamBuffer(method="hold", tolerance_ns=10 * _NS_PER_MS, capacity=0)


# ----------------------------------------------------------------------------
# hold semantics
# ----------------------------------------------------------------------------


def test_hold_returns_none_on_empty_buffer():
    buf = StreamBuffer(method="hold", tolerance_ns=50 * _NS_PER_MS)
    assert buf.sample(now_ns=1_000_000) is None


def test_hold_returns_last_value_within_tolerance():
    buf = StreamBuffer(method="hold", tolerance_ns=50 * _NS_PER_MS)
    buf.push(100_000_000, "a")
    buf.push(200_000_000, "b")
    # now is 220 ms (20 ms after newest push) → still within 50 ms tolerance
    assert buf.sample(now_ns=220_000_000) == "b"


def test_hold_returns_none_when_newest_push_is_older_than_tolerance():
    buf = StreamBuffer(method="hold", tolerance_ns=50 * _NS_PER_MS)
    buf.push(100_000_000, "a")
    # 200 ms after the push → outside 50 ms tolerance
    assert buf.sample(now_ns=300_000_000) is None


def test_hold_ignores_future_samples():
    # If a sample arrived from the future (e.g. driver clock skew), hold
    # must not jump forward — it must use the most-recent past sample.
    buf = StreamBuffer(method="hold", tolerance_ns=100 * _NS_PER_MS)
    buf.push(100_000_000, "past")
    buf.push(500_000_000, "future")
    assert buf.sample(now_ns=150_000_000) == "past"


def test_hold_returns_none_when_only_future_samples_exist():
    buf = StreamBuffer(method="hold", tolerance_ns=100 * _NS_PER_MS)
    buf.push(500_000_000, "future")
    assert buf.sample(now_ns=100_000_000) is None


# ----------------------------------------------------------------------------
# nearest semantics
# ----------------------------------------------------------------------------


def test_nearest_picks_closer_of_before_and_after():
    buf = StreamBuffer(method="nearest", tolerance_ns=100 * _NS_PER_MS)
    buf.push(100_000_000, "before")  # 50 ms before now
    buf.push(170_000_000, "after")   # 20 ms after now
    assert buf.sample(now_ns=150_000_000) == "after"


def test_nearest_picks_before_when_closer():
    buf = StreamBuffer(method="nearest", tolerance_ns=100 * _NS_PER_MS)
    buf.push(140_000_000, "before")  # 10 ms before now
    buf.push(200_000_000, "after")   # 50 ms after now
    assert buf.sample(now_ns=150_000_000) == "before"


def test_nearest_equidistant_tie_prefers_earlier_timestamp():
    # Two samples exactly equidistant from now. pandas
    # merge_asof(direction="nearest") — the converter's alignment — breaks
    # such a tie toward the earlier timestamp. Push in timestamp-descending
    # order so a push-order tie-break would pick the later one and fail.
    buf = StreamBuffer(method="nearest", tolerance_ns=100 * _NS_PER_MS)
    buf.push(200_000_000, "after")   # 50 ms after now, pushed first
    buf.push(100_000_000, "before")  # 50 ms before now, pushed second
    assert buf.sample(now_ns=150_000_000) == "before"


def test_nearest_respects_tolerance_on_either_side():
    buf = StreamBuffer(method="nearest", tolerance_ns=10 * _NS_PER_MS)
    buf.push(50_000_000, "before")   # 100 ms away
    buf.push(250_000_000, "after")   # 100 ms away
    assert buf.sample(now_ns=150_000_000) is None


def test_nearest_empty_returns_none():
    buf = StreamBuffer(method="nearest", tolerance_ns=10 * _NS_PER_MS)
    assert buf.sample(now_ns=0) is None


# ----------------------------------------------------------------------------
# Clock-reset / out-of-order pushes
# ----------------------------------------------------------------------------


def test_clock_reset_does_not_corrupt_subsequent_samples():
    # Simulate a ReplayAdapter bag rewind: ts goes backwards, then forwards
    # again. Sampling at "now" past the rewind should pick the right value
    # by timestamp, not by push order.
    buf = StreamBuffer(method="hold", tolerance_ns=200 * _NS_PER_MS)
    buf.push(500_000_000, "old")   # ts=500ms
    buf.push(100_000_000, "rewind")  # bag restarted, ts=100ms
    buf.push(150_000_000, "rewind2")  # ts=150ms
    # now=160ms — hold should return the most recent push with ts<=now,
    # which is "rewind2" at 150ms, not "old" at 500ms (which is in the future).
    assert buf.sample(now_ns=160_000_000) == "rewind2"


# ----------------------------------------------------------------------------
# Capacity overflow
# ----------------------------------------------------------------------------


def test_capacity_overflow_drops_oldest():
    buf = StreamBuffer(method="hold", tolerance_ns=1_000_000_000, capacity=3)
    for i in range(5):
        buf.push((i + 1) * 100_000_000, f"v{i}")
    assert len(buf) == 3
    # Oldest two ("v0", "v1") were dropped; the buffer holds v2, v3, v4.
    # At now=500ms (= ts of v4), hold returns v4.
    assert buf.sample(now_ns=500_000_000) == "v4"
    # The hold lookup back to v2's ts must still succeed since v2 is still resident.
    assert buf.sample(now_ns=300_000_000) == "v2"


def test_default_capacity_constant_is_reasonable():
    # Pin the documented default so a refactor that quietly shrinks it
    # to e.g. 2 surfaces as a test diff rather than a runtime regression.
    assert DEFAULT_CAPACITY == 64
