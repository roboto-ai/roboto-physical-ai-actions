"""Adapter for lerobot 0.3.x (LeRobot dataset format 2.1).

API differences vs 0.5.x that this adapter papers over:
- ``LeRobotDataset.add_frame`` takes ``task`` as a separate kwarg; ``task``
  must NOT be inside the frame dict.
- ``LeRobotDataset.finalize()`` does not exist on 0.3.x — episodes are
  considered finalised on every ``save_episode``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..logger import logger


class LeRobotWriter:
    """0.3.x adapter — see :class:`roboto_to_lerobot.writers.base.LeRobotWriter`."""

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
        # 0.5.x-only encoder knobs: 0.3.x's LeRobotDataset.create rejects
        # them, so we keep them in the adapter signature (so main.py has a
        # single call shape) and drop them before calling lerobot.
        _ = (
            batch_encoding_size,
            streaming_encoding,
            encoder_threads,
            encoder_queue_maxsize,
        )
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        # use_videos=True is set explicitly to match v3_0 and guarantee
        # video-backed output regardless of the 0.3.x default.
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=dict(features),
            root=root,
            robot_type=robot_type,
            use_videos=True,
            image_writer_threads=image_writer_threads,
        )
        return cls(dataset)

    def add_frame(self, frame: dict[str, Any], task: str) -> None:
        # 0.3.x: ``task`` must be passed as a kwarg, not as a frame key.
        frame_no_task = {k: v for k, v in frame.items() if k != "task"}
        self._dataset.add_frame(frame_no_task, task=task)

    def save_episode(self) -> None:
        self._dataset.save_episode()

    def discard_episode(self) -> None:
        # 0.3.x has no documented abort hook, so fall through a ladder:
        #   1. clear_episode_buffer (0.5.x-style, kept in case 0.3.x exposes it)
        #   2. rebuild the buffer via create_episode_buffer
        #   3. warn — never raise. discard_episode is the soft-drop path; if
        #      it propagates, a single bad event takes down the whole run.
        clear_fn = getattr(self._dataset, "clear_episode_buffer", None)
        if callable(clear_fn):
            try:
                clear_fn(delete_images=True)
            except TypeError:
                clear_fn()
            return

        create_fn = getattr(self._dataset, "create_episode_buffer", None)
        if callable(create_fn):
            try:
                self._dataset.episode_buffer = create_fn()
                return
            except Exception:
                pass

        logger.warning(
            "v2_1 writer: no clear_episode_buffer hook and could not reset "
            "episode_buffer; next episode may inherit stale frames.",
        )

    def finalize(self) -> None:
        # 0.3.x does not expose a finalize step — save_episode is sufficient.
        pass
