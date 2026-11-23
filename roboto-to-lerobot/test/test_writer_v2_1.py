"""Tests that the v2_1 writer saves every episode whole, or fails.

They drive the real lerobot 0.3.x writer and video encoding on a tiny
camera, and make one step of a save fail.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import PIL.Image
import pytest

lerobot_dataset = pytest.importorskip("lerobot.datasets.lerobot_dataset")
if not hasattr(lerobot_dataset, "LeRobotDatasetMetadata") or hasattr(
    pytest.importorskip("lerobot.datasets.video_utils"), "StreamingVideoEncoder"
):
    pytest.skip("needs lerobot 0.3.x", allow_module_level=True)

from roboto_to_lerobot.writers.base import EpisodeWriteError  # noqa: E402
from roboto_to_lerobot.writers.v2_1 import LeRobotWriter  # noqa: E402

_CAM = "observation.images.cam"
_FEATURES = {
    _CAM: {"dtype": "video", "shape": (32, 32, 3), "names": ["height", "width", "channels"]},
    "observation.state": {"dtype": "float32", "shape": (2,), "names": None},
}


def _writer(root: Path) -> LeRobotWriter:
    return LeRobotWriter.create(repo_id="test", fps=10, features=_FEATURES, root=root, robot_type="test")


def _add_frames(writer: LeRobotWriter, n: int) -> None:
    rng = np.random.default_rng(0)
    for i in range(n):
        frame = {
            _CAM: rng.integers(0, 255, (32, 32, 3), dtype=np.uint8),
            "observation.state": np.array([i, 0], dtype=np.float32),
        }
        writer.add_frame(frame, task="t")


def test_lost_image_fails_save(tmp_path, monkeypatch):
    """An episode whose video lost a frame fails to save instead of misaligning the dataset."""
    # Arrange: lerobot 0.3.x prints and ignores a failed image write. Frame 1
    # is not among the frames its image statistics read, so nothing else notices.
    stock_save = PIL.Image.Image.save

    def fail_frame_1(self, fp, *args, **kwargs):
        if Path(fp).name == "frame_000001.png":
            raise OSError("injected write failure")
        return stock_save(self, fp, *args, **kwargs)

    monkeypatch.setattr(PIL.Image.Image, "save", fail_frame_1)
    writer = _writer(tmp_path / "ds")
    _add_frames(writer, 200)

    # Act / Assert
    with pytest.raises(EpisodeWriteError, match=r"given 200 frames but its videos hold \{'observation.images.cam': "):
        writer.save_episode()


def test_save_failing_after_video_fails_save(tmp_path, monkeypatch):
    """A save that fails after writing the episode's video fails as an incomplete episode,
    since the next episode would reuse that video."""
    # Arrange: fail the metadata step, which lerobot 0.3.x runs after encoding.
    def fail(*args, **kwargs):
        raise RuntimeError("injected metadata failure")

    monkeypatch.setattr(lerobot_dataset.LeRobotDatasetMetadata, "save_episode", fail)
    writer = _writer(tmp_path / "ds")
    _add_frames(writer, 10)

    # Act / Assert
    with pytest.raises(EpisodeWriteError, match=r"videos for \['observation.images.cam'\] were written.*injected"):
        writer.save_episode()
