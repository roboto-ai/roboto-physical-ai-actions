"""Single-stream sample buffer for the live runtime.

Mirrors the converter's :func:`alignment._merge_hold` and
:func:`alignment._merge_nearest` semantics in event-driven form: ROS
callbacks ``push`` ``(ts_ns, value)`` tuples; the tick loop calls
``sample(now_ns)`` to materialise an observation. ``linear`` and ``none``
are not implemented and refuse at ``__init__`` so a
contract author sees the limitation at adapter boot rather than mid-episode.

Tolerance divergence from the converter
---------------------------------------
``AlignSpec.tolerance_ms is None`` means "unlimited carry-forward" offline
(see ``AlignSpec.tolerance_ms`` in ``contract_utils``). Contract loading
already resolves an omitted ``tolerance_ms``/``align`` block to a bounded
default using this same formula, so ``None`` only reaches the live runtime
when a contract explicitly sets ``tolerance_ms: null``. In the live
runtime, unlimited is unsafe regardless — a stalled stream would silently
feed the policy stale observations forever.
This module exposes :func:`auto_bound_tolerance_ns` which maps the
converter's "unlimited" (``None``) value to ``max(2/fps, 50 ms)``. The
choice is intentionally bounded but generous: at 30 fps, a stream that
misses two expected ticks is still served; longer gaps return ``None`` so
the adapter falls back to its ``safety_behavior`` (publish_nothing today).

The :class:`StreamBuffer` itself only receives the already-bounded
``tolerance_ns`` so it stays free of contract semantics — callers
(``LiveAdapter``) are responsible for the conversion plus the one-shot
log message naming any stream that got auto-bounded.
"""

from __future__ import annotations

from collections import deque
from typing import Any

SUPPORTED_METHODS = ("hold", "nearest")
_UNSUPPORTED_METHODS = ("linear", "none")

DEFAULT_CAPACITY = 64

_AUTO_BOUND_FLOOR_NS = 50_000_000  # 50 ms floor — fast streams shouldn't be over-eager


def auto_bound_tolerance_ns(tolerance_ms: float | None, fps: float) -> int:
    """Translate a contract ``tolerance_ms`` into a live-runtime ``tolerance_ns``.

    ``AlignSpec.tolerance_ms is None`` is the converter's "unlimited
    carry-forward" value; the live runtime caps that at
    ``max(2/fps, 50 ms)`` so a stalled stream can't silently feed
    last-known values forever. A positive value converts directly (ms →
    ns) so explicit contract tolerances pass through unchanged. Contract
    loading rejects ``tolerance_ms: 0`` and auto-resolves an omitted
    tolerance to this same bounded default, so by the time a value reaches
    here it is always ``None`` or strictly positive.

    Args:
        tolerance_ms: as read from ``AlignSpec.tolerance_ms``.
        fps: the contract's nominal sample rate; only consulted when the
            tolerance needs bounding (``tolerance_ms is None``).

    Returns:
        Effective tolerance in nanoseconds.
    """
    if tolerance_ms is not None:
        if tolerance_ms <= 0:
            raise ValueError(
                f"tolerance_ms must be None (unlimited) or > 0, got {tolerance_ms}"
            )
        return int(tolerance_ms * 1_000_000)
    if fps <= 0:
        raise ValueError(f"fps must be positive when tolerance_ms is None, got {fps}")
    return max(int(2e9 / fps), _AUTO_BOUND_FLOOR_NS)


class StreamBuffer:
    """Bounded ring of ``(ts_ns, value)`` tuples with hold/nearest sampling.

    A live ROS subscription owns one buffer per observation stream and
    drops messages into it on each callback. The tick loop calls
    :meth:`sample` to look up the value the policy should see. Tolerance
    is enforced on each sample so a stalled stream returns ``None``
    rather than feeding stale data.

    Args:
        method: one of :data:`SUPPORTED_METHODS`. ``linear`` and ``none``
            raise ``ValueError`` here — they are not implemented.
        tolerance_ns: maximum acceptable distance (in nanoseconds)
            between ``now_ns`` and the chosen sample's ``ts_ns``. Must be
            ``> 0``; callers convert the contract ``tolerance_ms`` via
            :func:`auto_bound_tolerance_ns`.
        capacity: ring size. Defaults to :data:`DEFAULT_CAPACITY`. When
            the ring is full, the oldest entry is silently dropped — a
            slow consumer leaking memory is worse than a missed sample.
    """

    __slots__ = ("_buffer", "_method", "_tolerance_ns")

    def __init__(
        self,
        method: str,
        tolerance_ns: int,
        capacity: int = DEFAULT_CAPACITY,
    ) -> None:
        if method in _UNSUPPORTED_METHODS:
            raise ValueError(
                f"StreamBuffer method '{method}' is not supported. "
                f"Supported methods: {SUPPORTED_METHODS}. "
                f"linear/none are not implemented."
            )
        if method not in SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown StreamBuffer method '{method}'. "
                f"Supported: {SUPPORTED_METHODS}."
            )
        if tolerance_ns <= 0:
            raise ValueError(
                f"tolerance_ns must be > 0, got {tolerance_ns}. "
                f"Use auto_bound_tolerance_ns() to translate from contract tolerance_ms."
            )
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")

        self._method = method
        self._tolerance_ns = tolerance_ns
        self._buffer: deque[tuple[int, Any]] = deque(maxlen=capacity)

    @property
    def method(self) -> str:
        return self._method

    @property
    def tolerance_ns(self) -> int:
        return self._tolerance_ns

    def __len__(self) -> int:
        return len(self._buffer)

    def push(self, ts_ns: int, value: Any) -> None:
        """Append a new sample. Capacity overflow silently drops the oldest."""
        self._buffer.append((ts_ns, value))

    def sample(self, now_ns: int) -> Any | None:
        """Return the value for ``now_ns`` under the buffer's policy.

        Returns ``None`` when the buffer is empty or when the best
        candidate is farther than ``tolerance_ns`` from ``now_ns``.
        """
        if not self._buffer:
            return None
        if self._method == "hold":
            return self._sample_hold(now_ns)
        # method == "nearest" — guarded at __init__
        return self._sample_nearest(now_ns)

    def _sample_hold(self, now_ns: int) -> Any | None:
        # Scan all entries rather than relying on push order so that a
        # bag rewind in ReplayAdapter (or a real ROS clock jump) doesn't
        # corrupt subsequent samples. With capacity 64 this is trivially
        # fast and avoids brittle insertion-order invariants.
        best_ts: int | None = None
        best_value: Any = None
        for ts_ns, value in self._buffer:
            if ts_ns > now_ns:
                continue
            if best_ts is None or ts_ns > best_ts:
                best_ts = ts_ns
                best_value = value
        if best_ts is None:
            return None
        if now_ns - best_ts > self._tolerance_ns:
            return None
        return best_value

    def _sample_nearest(self, now_ns: int) -> Any | None:
        best_dist: int | None = None
        best_ts: int | None = None
        best_value: Any = None
        for ts_ns, value in self._buffer:
            dist = abs(ts_ns - now_ns)
            # On an exact equidistant tie, prefer the earlier timestamp to
            # match pandas ``merge_asof(direction="nearest")`` (the converter's
            # alignment). Deque order is push order, which is not timestamp
            # order after a bag rewind / clock jump, so the tie-break must
            # compare ts_ns rather than rely on iteration order.
            if (
                best_dist is None
                or dist < best_dist
                or (dist == best_dist and ts_ns < best_ts)
            ):
                best_dist = dist
                best_ts = ts_ns
                best_value = value
        if best_dist is None or best_dist > self._tolerance_ns:
            return None
        return best_value
