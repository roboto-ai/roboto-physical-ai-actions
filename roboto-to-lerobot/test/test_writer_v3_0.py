"""Tests that the v3_0 writer saves every episode whole, or fails.

They drive the real lerobot writer and SVT-AV1 encoder on a tiny camera.
LeRobot's encoder thread is slowed, stopped or made to fail by replacing its
``run``; the encoder it would have used is otherwise unchanged.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import av
import numpy as np
import pytest

video_utils = pytest.importorskip("lerobot.datasets.video_utils", reason="needs lerobot >= 0.6")
if not hasattr(video_utils, "StreamingVideoEncoder"):
    pytest.skip("needs lerobot >= 0.6", allow_module_level=True)

from roboto_to_lerobot.writers.base import EpisodeWriteError  # noqa: E402
from roboto_to_lerobot.writers.v3_0 import LeRobotWriter  # noqa: E402

_CAM = "observation.images.cam"
_FEATURES = {
    _CAM: {"dtype": "video", "shape": (64, 64, 3), "names": ["height", "width", "channels"]},
    "observation.state": {"dtype": "float32", "shape": (2,), "names": None},
}
_STOCK_RUN = video_utils._CameraEncoderThread.run


def _writer(root: Path, *, queue_maxsize: int = 30) -> LeRobotWriter:
    return LeRobotWriter.create(
        repo_id="test", fps=10, features=_FEATURES, root=root, robot_type="test",
        image_writer_threads=1, streaming_encoding=True, encoder_threads=1, encoder_queue_maxsize=queue_maxsize,
    )


def _add_frames(writer: LeRobotWriter, n: int) -> None:
    rng = np.random.default_rng(0)
    for i in range(n):
        frame = {
            _CAM: rng.integers(0, 255, (64, 64, 3), dtype=np.uint8),
            "observation.state": np.array([i, 0], dtype=np.float32),
        }
        writer.add_frame(frame, task="t")


def _video_frames(root: Path) -> int:
    with av.open(str(root / "videos" / _CAM / "chunk-000" / "file-000.mp4")) as container:
        return sum(1 for _ in container.decode(video=0))


def _start_after(delay_s: float):
    def run(self) -> None:
        time.sleep(delay_s)
        _STOCK_RUN(self)

    return run


def _lose_third_frame(self) -> None:
    """Encode as LeRobot does, except that the episode's third frame never reaches the video."""

    class _LosesThirdFrame:
        def __init__(self, frames) -> None:
            self._frames = frames
            self._taken = 0

        def get(self, *args, **kwargs):
            self._taken += 1
            if self._taken == 3:
                self._frames.get(*args, **kwargs)
            return self._frames.get(*args, **kwargs)

    self.frame_queue = _LosesThirdFrame(self.frame_queue)
    _STOCK_RUN(self)


def _fail_at_end(self) -> None:
    """Consume frames until the end of the episode, then report a failure."""
    while self.frame_queue.get() is not None:
        pass
    self.result_queue.put(("error", "injected encoder failure"))


def test_slow_encoder_keeps_every_frame(tmp_path, monkeypatch):
    """Frames that arrive faster than the encoder takes them all reach the video."""
    # Arrange: a one-frame queue and an encoder that starts late; LeRobot's
    # own encoder drops a frame after waiting 0.1 s for queue space.
    monkeypatch.setattr(video_utils._CameraEncoderThread, "run", _start_after(0.5))
    writer = _writer(tmp_path / "ds", queue_maxsize=1)

    # Act
    _add_frames(writer, 10)
    writer.save_episode()
    writer.finalize()

    # Assert
    assert _video_frames(tmp_path / "ds") == 10


def test_encode_longer_than_lerobot_deadline_is_complete(tmp_path, monkeypatch):
    """An episode whose encode outlasts LeRobot's wait at save time is saved whole."""
    # Arrange: LeRobot gives each encode 120 s to finish; shorten that to 0.3 s
    # and make the encoder take 1 s.
    stock_join = video_utils._CameraEncoderThread.join
    monkeypatch.setattr(
        video_utils._CameraEncoderThread, "join",
        lambda self, timeout=None: stock_join(self, None if timeout is None else min(timeout, 0.3)),
    )
    monkeypatch.setattr(video_utils._CameraEncoderThread, "run", _start_after(1.0))
    writer = _writer(tmp_path / "ds")

    # Act
    _add_frames(writer, 10)
    writer.save_episode()
    writer.finalize()

    # Assert
    assert _video_frames(tmp_path / "ds") == 10


def test_video_missing_a_frame_fails_save(tmp_path, monkeypatch):
    """An episode whose video lost a frame in the encoder fails to save instead of misaligning the dataset."""
    # Arrange
    monkeypatch.setattr(video_utils._CameraEncoderThread, "run", _lose_third_frame)
    writer = _writer(tmp_path / "ds")
    _add_frames(writer, 10)

    # Act / Assert
    with pytest.raises(EpisodeWriteError):
        writer.save_episode()


def test_encoder_failure_at_save_fails_save(tmp_path, monkeypatch):
    """An encoder that fails while finishing the episode fails the save as an incomplete episode."""
    # Arrange
    monkeypatch.setattr(video_utils._CameraEncoderThread, "run", _fail_at_end)
    writer = _writer(tmp_path / "ds")
    _add_frames(writer, 10)

    # Act / Assert
    with pytest.raises(EpisodeWriteError):
        writer.save_episode()


def test_encoder_stopping_mid_episode_fails_add_frame(tmp_path, monkeypatch):
    """An encoder that stops mid-episode fails the next frame, and the next episode saves whole."""
    # Arrange: the first encoder thread exits after taking two frames; later
    # threads encode normally.
    started = []

    def stop_first(self) -> None:
        started.append(self)
        if len(started) > 1:
            _STOCK_RUN(self)
            return
        for _ in range(2):
            self.frame_queue.get()
        self.result_queue.put(("error", "injected encoder stop"))

    monkeypatch.setattr(video_utils._CameraEncoderThread, "run", stop_first)
    writer = _writer(tmp_path / "ds", queue_maxsize=1)

    # Act
    with pytest.raises(RuntimeError, match="injected encoder stop"):
        _add_frames(writer, 10)
    writer.discard_episode()
    _add_frames(writer, 5)
    writer.save_episode()
    writer.finalize()

    # Assert
    assert _video_frames(tmp_path / "ds") == 5



def test_episode_without_cameras_saves(tmp_path):
    """A dataset with no camera features saves its episodes; there is no video to wait for or check."""
    # Arrange
    writer = LeRobotWriter.create(
        repo_id="test", fps=10, features={"observation.state": _FEATURES["observation.state"]},
        root=tmp_path / "ds", robot_type="test", streaming_encoding=True,
    )

    # Act
    for i in range(5):
        writer.add_frame({"observation.state": np.array([i, 0], dtype=np.float32)}, task="t")
    writer.save_episode()
    writer.finalize()

    # Assert
    info = json.loads((tmp_path / "ds" / "meta" / "info.json").read_text())
    assert (info["total_episodes"], info["total_frames"]) == (1, 5)
