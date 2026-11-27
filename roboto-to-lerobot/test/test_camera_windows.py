"""Tests for reading camera topics in time windows (``camera_windows``).

A worker reads each camera's range in half-open windows and spools each
window's payloads before it requests the next. The window sizes are module
constants; these tests shrink them so a short range is read in many windows,
and check what a user would see: the episode holds the same frames as when
the range is read in one call, and a worker's memory does not grow with the
episode's length.
"""

from __future__ import annotations

import array
import math
import pickle
import tracemalloc
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from roboto.experimental.video.h264 import is_keyframe
from roboto_to_lerobot import camera_windows, event_worker
from roboto_to_lerobot.contract_utils import DataCollection
from roboto_to_lerobot.event_worker import EventTask, _worker_build_episode
from roboto_to_lerobot.main import _drain_one_episode, _DrainAccum
from roboto_to_lerobot.spool import SpoolWriter, unspool
from test_compressed_video import (
    COMPRESSED_VIDEO,
    FIRST_LOG_TIME_NS,
    FRAME_INTERVAL_NS,
    FakeVideoTopic,
    _h264,
    _messages,
)
from test_spool import COMPRESSED_IMAGE, _camera_contract

_SECOND = 1_000_000_000


class _SdkTopic:
    """A ``roboto.Topic`` stand-in that answers reads as the SDK's MCAP reader does.

    A read returns the messages whose log time is in ``[start_time, end_time)``,
    built into a DataFrame with ``pd.json_normalize`` and indexed by log time,
    as the SDK builds it. ``message(i)`` builds message ``i``'s record on every
    read, so the topic itself holds no payload. Records each range it is asked for.
    """

    topic_id = "tp_cam"

    def __init__(self, log_times: list[int], message: Callable[[int], dict]) -> None:
        self._log_times = log_times
        self._message = message
        self.start_time, self.end_time = min(log_times), max(log_times)
        self.requested_ranges: list[tuple[int, int]] = []

    def _read(self, start_time: pd.Timestamp, end_time: pd.Timestamp) -> Iterator[tuple[int, dict]]:
        self.requested_ranges.append((start_time.value, end_time.value))
        for i, log_time in enumerate(self._log_times):
            if start_time.value <= log_time < end_time.value:
                yield log_time, self._message(i)

    def get_data_as_df(self, *, start_time, end_time, message_paths_include=None) -> pd.DataFrame:
        times, records = [], []
        for log_time, record in self._read(start_time, end_time):
            times.append(log_time)
            records.append({k: v for k, v in record.items() if not message_paths_include or k in message_paths_include})
        df = pd.json_normalize(records)
        df.index = pd.to_datetime(pd.Series(times, dtype="int64"), unit="ns", utc=True)
        df.index.name = "_index"
        return df

    def get_data(self, *, start_time, end_time) -> Iterator[tuple[pd.Timestamp, dict]]:
        for log_time, record in self._read(start_time, end_time):
            yield pd.Timestamp(log_time, unit="ns"), {"image": record["data"]}


@pytest.fixture
def windows(monkeypatch) -> Callable[..., None]:
    """Set the window sizes: ``windows(first_ns, target_bytes, min_ns)``."""

    def set_windows(first_ns: int, target_bytes: int = 1, min_ns: int = 1) -> None:
        monkeypatch.setattr(camera_windows, "FIRST_WINDOW_NS", first_ns)
        monkeypatch.setattr(camera_windows, "TARGET_WINDOW_BYTES", target_bytes)
        monkeypatch.setattr(camera_windows, "MIN_WINDOW_NS", min_ns)

    return set_windows


_ONE_CALL = {"first_ns": 10**18}
_PAYLOAD_BYTES = 600
# Window sizes, with the range starting on a message and messages
# ``FRAME_INTERVAL_NS`` apart.
_WINDOW_PLANS = {
    # Every window two messages long, so every boundary is a message's log time.
    "every boundary on a message": {"first_ns": 2 * FRAME_INTERVAL_NS, "min_ns": 2 * FRAME_INTERVAL_NS},
    # Windows sized from the byte rate to hold about four messages, so most
    # boundaries fall between log times.
    "sized from the byte rate": {"first_ns": 3 * FRAME_INTERVAL_NS, "target_bytes": 4 * _PAYLOAD_BYTES},
}


# 23 messages at whole multiples of ``FRAME_INTERVAL_NS`` after the first: two
# share the log time 4 intervals after the first, and none falls 13 to 20
# intervals after it.
_STILL_LOG_TIMES = sorted(
    FIRST_LOG_TIME_NS + i * FRAME_INTERVAL_NS for i in [*range(5), 4, *range(5, 13), *range(21, 30)]
)
# The messages 6 and 7 intervals after the first, alone in one window of
# "every boundary on a message", carry no format: on pandas 3 that
# window's column alone is ``object`` rather than ``str``.
_UNLABELLED = {
    i for i, log_time in enumerate(_STILL_LOG_TIMES) if (log_time - FIRST_LOG_TIME_NS) // FRAME_INTERVAL_NS in (6, 7)
}


def _payload(i: int) -> bytes:
    return bytes([i % 251]) * _PAYLOAD_BYTES


def _compressed_image(i: int) -> dict:
    return {
        "header": {"stamp": {"sec": i, "nanosec": 7}, "frame_id": "cam"},
        "format": None if i in _UNLABELLED else "jpeg",
        "data": array.array("B", _payload(i)),
    }


def _plain(value: Any) -> Any:
    """``value`` with a spooled payload read back and any buffer as ``bytes``, so values compare by content.

    NaN becomes ``"NaN"``, so it equals NaN but not ``None``: the decoders turn
    a missing ``format`` or ``encoding`` into text with ``str()``, which gives
    ``nan`` for one and ``None`` for the other.
    """
    value = unspool(value)
    if isinstance(value, (array.array, np.ndarray, memoryview, bytearray)):
        return bytes(value)
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    return value


def _camera_rows(type_: str, spool_dir: Path) -> tuple[list, list, list, int]:
    """The camera's loaded DataFrame as (columns, null mask, values), and how many reads it took."""
    topic = _SdkTopic(_STILL_LOG_TIMES, _compressed_image)
    spool = SpoolWriter(spool_dir)
    collection = DataCollection(
        _camera_contract(type_), {"/cam": [topic]}, FIRST_LOG_TIME_NS, topic.end_time + 1, spool=spool,
    )
    df = collection.videos["observation.images.cam"]
    values = [[_plain(v) for v in row] for row in df.itertuples(index=False)]
    return list(df.columns), df.isna().to_numpy().tolist(), values, len(topic.requested_ranges)


@pytest.mark.parametrize("plan", list(_WINDOW_PLANS.values()), ids=list(_WINDOW_PLANS))
@pytest.mark.parametrize("type_", [COMPRESSED_IMAGE, "video"], ids=["jpeg", "file-backed-video"])
def test_a_camera_read_in_windows_loads_what_one_read_loads(tmp_path, windows, type_, plan):
    """Reading a camera in windows loads the same rows, values and nulls as one read of the range, with
    messages on window boundaries, two messages at one log time, and windows holding no message."""
    # Arrange
    windows(**_ONE_CALL)
    one_read = _camera_rows(type_, tmp_path)
    windows(**plan)

    # Act
    windowed = _camera_rows(type_, tmp_path)

    # Assert
    assert one_read[3] == 1
    assert windowed[3] > 4, "the range must be read in windows"
    assert windowed[:3] == one_read[:3]


def _written_compressed_video(messages: list, start_index: int, spool_dir: Path) -> tuple[list, list, int]:
    """Build an episode from a compressed-video camera in a worker and drain it; return what the writer
    receives, the dropped frames and how many reads the camera took."""
    topic = FakeVideoTopic(messages=messages, start_time=messages[0][0], end_time=messages[-1][0])
    contract = _camera_contract(COMPRESSED_VIDEO)
    start_ns, end_ns = messages[start_index][0], messages[-1][0] + 1
    step_ns = round(1e9 / contract.fps)
    task = EventTask(
        event_idx=0, event_id="ev", src_ds_id="ds", start_time_ns=start_ns, end_time_ns=end_ns, task_label="t",
        num_frames=(end_ns - start_ns) // step_ns, buffer_ns=0, action_lead_ns=0,
    )
    event_worker._WORKER_STATE.update(contracts={"ds": contract}, topics={"ds": {"/cam": [topic]}},
                                      spool_dir=str(spool_dir))
    episode = pickle.loads(pickle.dumps(_worker_build_episode(task)))
    images: list[np.ndarray] = []
    accumulator = _DrainAccum()
    _drain_one_episode(
        writer=SimpleNamespace(add_frame=lambda frame, task: images.append(frame[contract.videos[0].key]),
                               save_episode=lambda: None),
        video_specs_by_key={v.key: v for v in contract.videos}, src_ds_id="ds", event_id="ev",
        start_time_ns=start_ns, end_time_ns=end_ns, episode=episode, accumulator=accumulator,
    )
    return images, accumulator.episode_to_event[0]["dropped_frames"], len(topic.requested_ranges)


@pytest.mark.parametrize("plan", list(_WINDOW_PLANS.values()), ids=list(_WINDOW_PLANS))
@pytest.mark.parametrize("bframes", [0, 2], ids=["h264", "h264-with-b-frames"])
def test_compressed_video_read_in_windows_writes_what_one_read_writes(monkeypatch, tmp_path, windows, bframes, plan):
    """A compressed-video camera read in windows gives the writer the same frames, pixels and dropped frames
    as one read, with GOPs and the keyframe lookback before a mid-GOP start crossing window boundaries."""
    # Arrange: 8-frame GOPs, with the episode starting at message 11.
    monkeypatch.setattr(event_worker, "_WORKER_STATE", {})
    messages = _messages(list(_h264(bframes=bframes)))
    assert not is_keyframe(messages[11][2]), "the episode must start mid-GOP"
    windows(**_ONE_CALL)
    images, dropped, reads = _written_compressed_video(messages, 11, tmp_path)
    windows(**plan)

    # Act
    windowed_images, windowed_dropped, windowed_reads = _written_compressed_video(messages, 11, tmp_path)

    # Assert
    assert windowed_reads > reads, "the range must be read in windows"
    assert windowed_dropped == dropped
    assert len(windowed_images) == len(images) > 0
    for got, expected in zip(windowed_images, images, strict=True):
        np.testing.assert_array_equal(got, expected)


def _traced_worker_peak(frames: int, payload_bytes: int, spool_dir: Path) -> int:
    """tracemalloc's peak while a worker builds a ``frames``-frame JPEG-camera episode with a spool."""
    log_times = [FIRST_LOG_TIME_NS + i * FRAME_INTERVAL_NS for i in range(frames)]
    topic = _SdkTopic(log_times, lambda i: {"format": "jpeg", "data": array.array("B", bytes(payload_bytes))})
    contract = _camera_contract(COMPRESSED_IMAGE)
    event_worker._WORKER_STATE.update(contracts={"ds": contract}, topics={"ds": {"/cam": [topic]}},
                                      spool_dir=str(spool_dir))
    task = EventTask(
        event_idx=0, event_id="ev", src_ds_id="ds", start_time_ns=log_times[0], end_time_ns=log_times[-1] + 1,
        task_label="t", num_frames=frames * FRAME_INTERVAL_NS * contract.fps // _SECOND, buffer_ns=0,
        action_lead_ns=0,
    )
    tracemalloc.start()
    try:
        _worker_build_episode(task)
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_worker_memory_does_not_grow_with_the_length_of_a_spooled_camera(monkeypatch, tmp_path, windows):
    """With a spool, a worker building an episode four times as long peaks at about the same memory: it
    holds one window of camera payloads at a time, not the episode's."""
    # Arrange: windows of about 10 frames; the short episode is 30 frames.
    monkeypatch.setattr(event_worker, "_WORKER_STATE", {})
    payload_bytes = 200_000
    windows(10 * FRAME_INTERVAL_NS, target_bytes=10 * payload_bytes)

    # Act
    peak_short = _traced_worker_peak(30, payload_bytes, tmp_path)
    peak_long = _traced_worker_peak(120, payload_bytes, tmp_path)

    # Assert: holding the 90 extra frames' payloads would add 18 MB.
    assert peak_long - peak_short < 10 * payload_bytes
