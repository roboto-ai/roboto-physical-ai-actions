"""Fetching a camera topic's range in consecutive time windows.

The SDK returns a time range's messages all at once, so one call over an
episode's range holds all of a camera's payloads in memory until they are
spooled: a minute of one 640x480 camera at 30 fps is about 100 MB as JPEG
and 1.7 GB as a raw ``Image``. Fetching the range in windows, and spooling
each window's payloads before requesting the next, bounds what a camera
fetch holds by its window, whatever the episode's length.

Windows are half-open, ``[start, end)``, as the SDK's time ranges are, so
consecutive windows return every message of the range exactly once: a
message whose log time is a window's end belongs to the next window. A range
read in windows therefore returns the same messages, in the same order, as
one call over the range.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from typing import Any

__all__ = ("fetch_in_windows", "payload_nbytes")

# The first window's length. A camera's byte rate is unknown until a window
# has been read, so the first window is bounded by time. A range whose data
# spans no more than this is read in one call with its original bounds.
FIRST_WINDOW_NS = 10 * 1_000_000_000
# The payload bytes each later window aims for, sized from the byte rate of
# the window before it. 64 MB is about 40 s of a 640x480 JPEG camera, 2 s of
# the same camera raw, or 14 minutes of it as H.264.
TARGET_WINDOW_BYTES = 64 * 1024 * 1024
# Bounds on a later window's length: at least this long, so one oversized
# message does not turn the rest of the range into tiny requests, and at most
# this many times the previous window, so a sparse window does not size the
# next one to a camera that has only just started recording.
MIN_WINDOW_NS = 100_000_000
MAX_WINDOW_GROWTH = 4
# How many camera windows one process fetches at once. A fetch holds its
# window's payloads, with the SDK's own copies of the data alongside, until
# it returns, so this times about three windows bounds the memory that the
# fetches take. A CompressedImage, Image or file-backed video fetch spools
# its window before it returns; a compressed-video probe holds each window
# after its fetch, until its decoder has read it.
CONCURRENT_WINDOWS = 4
_WINDOW_SLOTS = threading.BoundedSemaphore(CONCURRENT_WINDOWS)


def payload_nbytes(value: Any) -> int:
    """The size in bytes of a camera payload, or 0 for a value that is not a buffer (``None``, NaN)."""
    try:
        return memoryview(value).nbytes
    except TypeError:
        return 0


def fetch_in_windows[T](
    fetch: Callable[[int, int], tuple[T, int, int | None]],
    start_ns: int,
    end_ns: int,
    *,
    data_start_ns: int | None = None,
    data_end_ns: int | None = None,
) -> Iterator[T]:
    """Fetch ``[start_ns, end_ns)`` in consecutive windows and yield each window's result in order.

    Each window is fetched only when the previous one's result has been
    consumed, so a caller that spools a window's payloads before asking for
    the next holds one window at a time. At most ``CONCURRENT_WINDOWS``
    fetches run at once in a process; the slot is released before the result
    is yielded, so a caller may fetch another range while it consumes one.

    The first window is ``FIRST_WINDOW_NS`` long. Each later one is sized
    from the bytes per nanosecond the previous window returned, measured from
    its first message to its end, to hold about ``TARGET_WINDOW_BYTES``; an
    empty window leaves the length unchanged.

    ``data_start_ns`` and ``data_end_ns``, the topic's first and last log
    times if known, only place the windows: the stretch of the range before
    the topic's first message joins the first window, the stretch after its
    last message joins the last, and a range whose data spans no more than
    the first window is fetched in one call with its original bounds. Wrong
    values change how the range is split, never which messages it returns.

    Args:
        fetch: Fetches one window ``[window_start, window_end)`` and returns
            its result, the payload bytes it holds, and the log time of its
            first message (``None`` when it is empty).
        start_ns: The range's start, inclusive.
        end_ns: The range's end, exclusive.
        data_start_ns: The topic's first log time, if known.
        data_end_ns: The topic's last log time, if known.

    Yields:
        Each window's result, in time order.
    """
    data_start = start_ns if data_start_ns is None else max(start_ns, data_start_ns)
    data_end = end_ns if data_end_ns is None else min(end_ns, data_end_ns + 1)
    window_ns = FIRST_WINDOW_NS
    if data_end - data_start <= window_ns:
        with _WINDOW_SLOTS:
            result, _, _ = fetch(start_ns, end_ns)
        yield result
        return
    window_start, window_end = start_ns, data_start + window_ns
    while True:
        if window_end >= data_end:
            window_end = end_ns
        with _WINDOW_SLOTS:
            result, payload_bytes, first_ns = fetch(window_start, window_end)
        yield result
        del result  # the caller has this window; hold no more than the next one while fetching it
        if window_end == end_ns:
            return
        if payload_bytes > 0 and first_ns is not None:
            data_ns = max(1, window_end - max(window_start, first_ns))
            sized_ns = TARGET_WINDOW_BYTES * data_ns // payload_bytes
            window_ns = max(MIN_WINDOW_NS, min(sized_ns, MAX_WINDOW_GROWTH * window_ns))
        window_start, window_end = window_end, window_end + window_ns
