"""Adapter for lerobot 0.5.x (LeRobot dataset format 3.0).

API differences vs 0.3.x that this adapter papers over:
- ``LeRobotDataset.add_frame`` reads ``task`` from the frame dict; passing it
  as a kwarg is rejected.
- ``LeRobotDataset.finalize()`` must be called once after the last episode
  to write the dataset-level metadata.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

# vcodec is locked to libsvtav1 because alternatives degrade downstream
# training quality. Not exposed on the Protocol so callers cannot override.
_VCODEC: Final[str] = "libsvtav1"


class LeRobotWriter:
    """0.5.x adapter — see :class:`roboto_to_lerobot.writers.base.LeRobotWriter`."""

    def __init__(self, dataset: Any) -> None:
        self._dataset = dataset

    @classmethod
    def create(
        cls,
        *,
        repo_id: str,
        fps: int,
        features: Mapping[str, Any],
        root: Path,
        robot_type: str,
        image_writer_threads: int = 8,
        batch_encoding_size: int = 1,
        streaming_encoding: bool = False,
        encoder_threads: int | None = None,
        encoder_queue_maxsize: int = 30,
    ) -> LeRobotWriter:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=dict(features),
            root=root,
            robot_type=robot_type,
            use_videos=True,
            vcodec=_VCODEC,
            image_writer_threads=image_writer_threads,
            batch_encoding_size=batch_encoding_size,
            streaming_encoding=streaming_encoding,
            encoder_threads=encoder_threads,
            encoder_queue_maxsize=encoder_queue_maxsize,
        )
        return cls(dataset)

    def add_frame(self, frame: dict[str, Any], task: str) -> None:
        # 0.5.x reads ``task`` from the frame dict. The explicit kwarg wins
        # over any pre-existing ``task`` key so the writer stays the single
        # source of truth.
        frame_with_task = {**frame, "task": task}
        self._dataset.add_frame(frame_with_task)

    def save_episode(self) -> None:
        self._dataset.save_episode()

    def discard_episode(self) -> None:
        # delete_images=True purges the staged image tempfiles alongside the
        # in-memory buffer, so the next save_episode cannot inherit
        # half-buffered frames from a failed conversion.
        self._dataset.clear_episode_buffer(delete_images=True)

    def finalize(self) -> None:
        self._dataset.finalize()
