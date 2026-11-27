"""Tests for spooling camera payloads to a per-episode file (``spool``).

A worker given a spool directory writes each fetched camera payload to the
episode's spool file and ships references in its place; the writer reads
each frame's bytes back just before decoding it. These tests build episodes
through the real worker and drain, with in-memory topics, and check what a
user would see: the writer receives the same frames, or the same error,
with or without a spool; the episode handed to the writer stays small however
large its camera payloads; and no spool file outlives its episode.
"""

from __future__ import annotations

import array
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pytest
from roboto_to_lerobot import event_worker
from roboto_to_lerobot.contract_utils import AlignSpec, Contract, ObservationSpec
from roboto_to_lerobot.event_worker import EventTask, _worker_build_episode
from roboto_to_lerobot.main import _drain_one_episode, _DrainAccum
from test_compressed_video import (
    COMPRESSED_VIDEO,
    FIRST_LOG_TIME_NS,
    FRAME_INTERVAL_NS,
    _encode_access_units,
    _encode_h264_access_units,
    _topic_with,
)

_FRAMES = 12
_FPS = 30


class _ImageTopic:
    """``roboto.Topic`` stand-in serving one image message per frame, indexed by log time."""

    topic_id = "tp_cam"

    def __init__(self, columns: dict[str, list[Any]]) -> None:
        times = [FIRST_LOG_TIME_NS + i * FRAME_INTERVAL_NS for i in range(len(columns["data"]))]
        self._df = pd.DataFrame(columns, index=pd.to_datetime(times))
        self.start_time, self.end_time = times[0], times[-1]

    def get_data_as_df(self, *, start_time, end_time, message_paths_include=None):
        df = self._df.loc[start_time:end_time]
        if message_paths_include:
            df = df[[c for c in df.columns if c in message_paths_include]]
        return df.copy()


class _FailingTopic(_ImageTopic):
    """A topic whose fetch fails, as one does when the service is unreachable."""

    topic_id = "tp_broken"

    def get_data_as_df(self, **_kwargs):
        raise RuntimeError("fetch failed")


def _pixels(i: int, height: int = 24, width: int = 32) -> np.ndarray:
    rng = np.random.default_rng(i)
    return rng.integers(0, 255, (height, width, 3), dtype=np.uint8)


def _encoded(ext: str, image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(ext, image)
    assert ok
    return buffer.tobytes()


def _as(source: str, data: bytes) -> Any:
    """``data`` as the SDK hands it over: MCAP ``uint8[]`` fields are ``array.array('B')``,
    Parquet ``list<uint8>`` cells are ndarrays, Parquet binary cells are ``bytes``."""
    return {"bytes": bytes, "array": lambda d: array.array("B", d),
            "ndarray": lambda d: np.frombuffer(d, dtype=np.uint8).copy()}[source](data)


def _compressed_image_topic(ext: str, source: str) -> _ImageTopic:
    data = [_as(source, _encoded(ext, _pixels(i))) for i in range(_FRAMES)]
    return _ImageTopic({"format": [ext.strip(".")] * _FRAMES, "data": data})


def _raw_image_topic(encoding: str, data: list[bytes], source: str, height: int = 24, width: int = 32) -> _ImageTopic:
    return _ImageTopic({
        "height": [height] * _FRAMES, "width": [width] * _FRAMES, "encoding": [encoding] * _FRAMES,
        "data": [_as(source, d) for d in data],
    })


def _rgb8_topic(source: str, height: int = 24, width: int = 32) -> _ImageTopic:
    return _raw_image_topic(
        "rgb8", [_pixels(i, height, width).tobytes() for i in range(_FRAMES)], source, height, width,
    )


def _png_depth_topic(source: str) -> _ImageTopic:
    """A depth camera whose pixels arrive PNG-encoded, which the decoder recognizes only in ``bytes``."""
    depth = [np.full((24, 32), 1000 + i, dtype=np.uint16) for i in range(_FRAMES)]
    return _raw_image_topic("16UC1", [_encoded(".png", d) for d in depth], source)


def _camera_contract(type_: str, resize: tuple[int, int] | None = (16, 16)) -> Contract:
    camera = ObservationSpec(
        key="observation.images.cam", topic="/cam", type=type_,
        image={"resize": list(resize)} if resize else None,
        align=AlignSpec(method="hold", tolerance_ms=None),
    )
    return Contract(
        name="t", version=1, fps=_FPS, action_lead_steps=0,
        observations=[], videos=[camera], actions=[], tasks=[], robot_type=None,
    )


def _build_episode(monkeypatch, contract: Contract, topics: dict, spool_dir: Path | None, frames: int = _FRAMES):
    """Build one episode in a worker and pass it through pickle, as the pool hands it to the writer."""
    monkeypatch.setitem(event_worker._WORKER_STATE, "contracts", {"ds": contract})
    monkeypatch.setitem(event_worker._WORKER_STATE, "topics", {"ds": topics})
    monkeypatch.setitem(event_worker._WORKER_STATE, "spool_dir", str(spool_dir) if spool_dir else None)
    step_ns = 1_000_000_000 // _FPS
    task = EventTask(
        event_idx=0, event_id="ev", src_ds_id="ds", start_time_ns=FIRST_LOG_TIME_NS,
        end_time_ns=FIRST_LOG_TIME_NS + (frames - 1) * step_ns, task_label="t", num_frames=frames,
        buffer_ns=0, action_lead_ns=0,
    )
    return pickle.loads(pickle.dumps(_worker_build_episode(task)))


def _drain(contract: Contract, episode, writer=None) -> list[dict]:
    written: list[dict] = []
    writer = writer or SimpleNamespace(
        add_frame=lambda frame, task: written.append({k: np.array(v) for k, v in frame.items()}),
        save_episode=lambda: None,
    )
    _drain_one_episode(
        writer=writer, video_specs_by_key={v.key: v for v in contract.videos}, src_ds_id="ds",
        event_id="ev", start_time_ns=0, end_time_ns=1, episode=episode, accumulator=_DrainAccum(),
    )
    return written


def _lossless_h264_topic(width: int, height: int):
    """A compressed-video camera encoded losslessly, so its payload grows with its resolution."""
    return _topic_with(_encode_access_units(
        "libx264", _FRAMES + 4, gop=5, width=width, height=height,
        options={"bf": "0", "tune": "zerolatency", "preset": "ultrafast", "qp": "0"},
    ))


IMAGE = "sensor_msgs/msg/Image"
COMPRESSED_IMAGE = "sensor_msgs/msg/CompressedImage"


@pytest.mark.parametrize(
    ("contract", "topic"),
    [
        pytest.param(_camera_contract(COMPRESSED_IMAGE), lambda: _compressed_image_topic(".jpg", "bytes"), id="jpeg"),
        pytest.param(_camera_contract(IMAGE, resize=None), lambda: _rgb8_topic("array"), id="raw-mcap"),
        pytest.param(_camera_contract(IMAGE), lambda: _png_depth_topic("bytes"), id="png-in-raw"),
        pytest.param(_camera_contract(COMPRESSED_VIDEO),
                     lambda: _topic_with(_encode_h264_access_units(_FRAMES + 4, gop=5)), id="h264"),
    ],
)
def test_the_writer_receives_the_same_frames_with_and_without_a_spool(monkeypatch, tmp_path, contract, topic):
    """Spooling changes where camera bytes wait, not what is written: each frame is the same."""
    # Arrange
    topics = {"/cam": [topic()]}

    # Act
    in_memory = _drain(contract, _build_episode(monkeypatch, contract, topics, spool_dir=None))
    spooled = _drain(contract, _build_episode(monkeypatch, contract, topics, spool_dir=tmp_path))

    # Assert
    assert len(spooled) == len(in_memory) > 0
    for got, expected in zip(spooled, in_memory, strict=True):
        assert got.keys() == expected.keys()
        for key, value in expected.items():
            assert got[key].dtype == value.dtype, key
            np.testing.assert_array_equal(got[key], value, err_msg=key)


@pytest.mark.parametrize(
    ("contract", "topic"),
    [
        # The decoder recognizes PNG only in ``bytes``, so it reads these PNG
        # depth bytes as raw 16-bit pixels and fails.
        pytest.param(_camera_contract(IMAGE), lambda: _png_depth_topic("array"), id="png-in-raw-mcap"),
        # The decoder compares an ndarray's first bytes to the PNG signature,
        # which raises on an ndarray.
        pytest.param(_camera_contract(IMAGE, resize=None), lambda: _rgb8_topic("ndarray"), id="raw-parquet-list"),
    ],
)
def test_a_camera_that_fails_without_a_spool_fails_the_same_way_with_one(monkeypatch, tmp_path, contract, topic):
    """A payload reads back from the spool as the type the SDK returned, so a camera whose decode fails on
    that type fails with the same error with or without a spool."""
    # Arrange
    topics = {"/cam": [topic()]}

    # Act
    with pytest.raises(Exception) as in_memory:
        _drain(contract, _build_episode(monkeypatch, contract, topics, spool_dir=None))
    with pytest.raises(Exception) as spooled:
        _drain(contract, _build_episode(monkeypatch, contract, topics, spool_dir=tmp_path))

    # Assert
    assert (spooled.type, str(spooled.value)) == (in_memory.type, str(in_memory.value))


@pytest.mark.parametrize(
    ("contract", "small", "large"),
    [
        # rgb8 at 8x8 and 256x256: 192 and 196,608 bytes per frame.
        pytest.param(_camera_contract(IMAGE, resize=None), lambda: _rgb8_topic("bytes", 8, 8),
                     lambda: _rgb8_topic("bytes", 256, 256), id="bytes"),
        pytest.param(_camera_contract(IMAGE, resize=None), lambda: _rgb8_topic("array", 8, 8),
                     lambda: _rgb8_topic("array", 256, 256), id="mcap-array"),
        pytest.param(_camera_contract(IMAGE, resize=None), lambda: _rgb8_topic("ndarray", 8, 8),
                     lambda: _rgb8_topic("ndarray", 256, 256), id="parquet-ndarray"),
        pytest.param(_camera_contract(COMPRESSED_VIDEO), lambda: _lossless_h264_topic(64, 48),
                     lambda: _lossless_h264_topic(1280, 720), id="h264"),
    ],
)
def test_an_episode_handed_to_the_writer_does_not_grow_with_its_camera_payload(
    monkeypatch, tmp_path, contract, small, large,
):
    """With a spool, the pickled episode carries references, so its size does not depend on the camera's
    payload; without one, the larger payload makes the same episode more than 100 KB larger."""
    # Arrange
    small_topics, large_topics = {"/cam": [small()]}, {"/cam": [large()]}

    def growth(spool_dir: Path | None) -> int:
        """How many bytes larger the large camera's episode pickles than the small camera's."""
        small_size, large_size = (
            len(pickle.dumps(_build_episode(monkeypatch, contract, topics, spool_dir)))
            for topics in (small_topics, large_topics)
        )
        return large_size - small_size

    # Act
    in_memory_growth = growth(spool_dir=None)
    spooled_growth = growth(spool_dir=tmp_path)

    # Assert
    assert in_memory_growth > 100_000
    assert spooled_growth < 1_000


@pytest.mark.parametrize("writer_fails", [False, True], ids=["saved", "writer-fails"])
def test_draining_an_episode_deletes_its_spool_file_whether_or_not_it_is_saved(monkeypatch, tmp_path, writer_fails):
    """The drain removes the episode's spool file once the episode is saved, or once writing it failed."""
    # Arrange
    contract = _camera_contract(COMPRESSED_IMAGE)
    episode = _build_episode(monkeypatch, contract, {"/cam": [_compressed_image_topic(".jpg", "bytes")]}, tmp_path)
    assert Path(episode.spool_path).is_file()

    def add_frame(frame, task):
        if writer_fails:
            raise RuntimeError("disk full")

    writer = SimpleNamespace(add_frame=add_frame, save_episode=lambda: None)

    # Act
    try:
        _drain(contract, episode, writer)
    except RuntimeError:
        assert writer_fails

    # Assert
    assert list(tmp_path.iterdir()) == []


def test_a_worker_that_fails_to_build_its_episode_leaves_no_spool_file(monkeypatch, tmp_path):
    """An event the worker fails to build leaves no spool file, whether or not its other camera's payloads
    were already spooled."""
    # Arrange
    contract = _camera_contract(COMPRESSED_IMAGE)
    contract.videos.append(ObservationSpec(key="observation.images.broken", topic="/broken", type=COMPRESSED_IMAGE))
    topics = {"/cam": [_compressed_image_topic(".jpg", "bytes")], "/broken": [_FailingTopic({"data": [b""]})]}

    # Act
    with pytest.raises(RuntimeError, match="fetch failed"):
        _build_episode(monkeypatch, contract, topics, tmp_path)

    # Assert
    assert list(tmp_path.iterdir()) == []
