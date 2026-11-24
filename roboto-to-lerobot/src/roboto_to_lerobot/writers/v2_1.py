"""Adapter for lerobot 0.3.x (LeRobot dataset format 2.1).

API differences vs 0.6.x that this adapter papers over:
- ``LeRobotDataset.add_frame`` takes ``task`` as a separate kwarg; ``task``
  must NOT be inside the frame dict.
- ``LeRobotDataset.finalize()`` does not exist on 0.3.x — episodes are
  considered finalised on every ``save_episode``.

It also ensures that every saved episode is complete, or that its save
fails. lerobot 0.3.x writes each frame's image to a PNG, prints and ignores a
failed write, then encodes the episode's video from the PNGs that exist, so
after each save the adapter checks that every video holds as many frames as
were added.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..logger import logger
from .base import EpisodeWriteError


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
        video_encoding: Mapping[str, Any] | None = None,
    ) -> LeRobotWriter:
        # 0.6.x-only encoder knobs: 0.3.x's LeRobotDataset.create rejects
        # them, so we keep them in the adapter signature (so main.py has a
        # single call shape) and drop them before calling lerobot.
        # ``video_encoding`` holds no settings, because
        # ``resolve_video_encoding`` refuses every one.
        _ = (
            batch_encoding_size,
            streaming_encoding,
            encoder_threads,
            encoder_queue_maxsize,
            video_encoding,
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

    @classmethod
    def resolve_video_encoding(cls, overrides: Mapping[str, int]) -> dict[str, Any]:
        """Return no settings; raise ``ValueError`` if ``overrides`` sets any.

        lerobot 0.3.x encodes with settings it fixes itself, so ignoring a
        requested setting would silently change the output's quality.
        """
        if overrides:
            raise ValueError(
                "video_crf, video_preset and video_gop need roboto-to-lerobot-v3_0; "
                "lerobot 0.3.x, which roboto-to-lerobot-v2_1 runs, fixes its encoder settings."
            )
        return {}

    def add_frame(self, frame: dict[str, Any], task: str) -> None:
        # 0.3.x: ``task`` must be passed as a kwarg, not as a frame key.
        frame_no_task = {k: v for k, v in frame.items() if k != "task"}
        self._dataset.add_frame(frame_no_task, task=task)

    def save_episode(self) -> None:
        dataset, meta = self._dataset, self._dataset.meta
        frames = dataset.episode_buffer["size"]
        videos = {key: dataset.root / meta.get_video_file_path(meta.total_episodes, key) for key in meta.video_keys}
        try:
            dataset.save_episode()
        except Exception as exc:
            # lerobot 0.3.x encodes the videos before it records the episode
            # in the metadata. A save that fails between the two keeps the
            # episode index, and lerobot skips encoding a video file that
            # already exists, so the next episode would take this episode's
            # video. A save that fails later has already recorded the episode.
            left = sorted(key for key, path in videos.items() if path.exists())
            if left:
                raise EpisodeWriteError(
                    f"Saving an episode of {frames} frames failed after its videos for {left} were written: "
                    f"{exc.__class__.__name__}: {exc}"
                ) from exc
            raise
        counts = {key: _count_video_frames(path) for key, path in videos.items()}
        wrong = {key: count for key, count in counts.items() if count != frames}
        if wrong:
            raise EpisodeWriteError(
                f"Episode {meta.total_episodes - 1} was given {frames} frames but its videos hold {wrong}; "
                "its data rows and videos would be misaligned."
            )

    def discard_episode(self) -> None:
        # 0.3.x has no documented abort hook, so fall through a ladder:
        #   1. clear_episode_buffer (0.3.3 has it, without the delete_images
        #      argument that 0.6.x takes)
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
        # 0.3.x has no finalize step: save_episode writes every file.
        # Stopping the image writer joins its threads, which would otherwise
        # idle until the process exits. The episode cache creates one writer
        # per episode, so without this a shard would keep every episode's
        # threads alive.
        self._dataset.stop_image_writer()


def _count_video_frames(path: Path) -> int:
    """Count the encoded frames in ``path``'s first video stream, one per non-empty packet.

    ``demux`` ends with an empty packet that flushes the decoder; it holds no frame.
    """
    import av

    with av.open(str(path)) as container:
        return sum(1 for packet in container.demux(video=0) if packet.size)
