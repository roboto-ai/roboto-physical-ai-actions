"""Adapter for lerobot 0.6.x and 0.7.x (LeRobot dataset format 3.0).

API differences from 0.3.x that this adapter handles:
- ``LeRobotDataset.add_frame`` reads ``task`` from the frame dict; passing it
  as a kwarg is rejected.
- ``LeRobotDataset.finalize()`` must be called once after the last episode
  to write the dataset-level metadata.
- ``LeRobotDataset.create`` takes the codec inside an ``RGBEncoderConfig``
  (``rgb_encoder``) rather than as a ``vcodec`` kwarg.

It also ensures that every saved episode is complete, or that its save
fails. With streaming encoding it replaces LeRobot's encoder with one that
waits instead of dropping frames (see
:mod:`roboto_to_lerobot.writers.streaming_encoder`), and after each save it
checks that every video holds as many frames as the episode has rows.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from .base import EpisodeWriteError

# vcodec is locked to libsvtav1 because alternatives degrade downstream
# training quality. Not exposed on the Protocol so callers cannot override.
_VCODEC: Final[str] = "libsvtav1"


class LeRobotWriter:
    """0.6.x and 0.7.x adapter — see :class:`roboto_to_lerobot.writers.base.LeRobotWriter`."""

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
        from lerobot.configs.video import RGBEncoderConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            features=dict(features),
            root=root,
            robot_type=robot_type,
            use_videos=True,
            # Other RGBEncoderConfig defaults (g=2, crf=30, preset=12) match
            # what 0.5.x hard-coded, so encoded videos are unchanged.
            rgb_encoder=RGBEncoderConfig(vcodec=_VCODEC),
            image_writer_threads=image_writer_threads,
            batch_encoding_size=batch_encoding_size,
            streaming_encoding=streaming_encoding,
            encoder_threads=encoder_threads,
            encoder_queue_maxsize=encoder_queue_maxsize,
        )
        # LeRobot builds a streaming encoder only for a dataset with videos.
        if streaming_encoding and dataset.meta.video_keys:
            _install_waiting_encoder(dataset)
        return cls(dataset)

    def add_frame(self, frame: dict[str, Any], task: str) -> None:
        # lerobot 0.6 and later read ``task`` from the frame dict. The
        # ``task`` argument replaces any ``task`` key already in ``frame``.
        frame_with_task = {**frame, "task": task}
        self._dataset.add_frame(frame_with_task)

    def save_episode(self) -> None:
        # LeRobot writes the episode's data rows before it finishes the
        # videos, so a failure after that leaves rows in the dataset that the
        # episode metadata does not cover.
        writer = self._dataset.writer
        rows_before = writer._recorded_frames
        frames = writer.episode_buffer["size"]
        try:
            self._dataset.save_episode()
        except Exception as exc:
            if writer._recorded_frames != rows_before:
                raise EpisodeWriteError(
                    f"Saving an episode of {frames} frames failed after its data rows were written, "
                    f"so the dataset holds rows no episode describes: {exc.__class__.__name__}: {exc}"
                ) from exc
            raise
        _check_video_frames(self._dataset.meta, frames)

    def discard_episode(self) -> None:
        # delete_images=True purges the staged image tempfiles alongside the
        # in-memory buffer, so the next save_episode cannot inherit
        # half-buffered frames from a failed conversion.
        self._dataset.clear_episode_buffer(delete_images=True)

    def finalize(self) -> None:
        self._dataset.finalize()


def _install_waiting_encoder(dataset: Any) -> None:
    """Replace ``dataset``'s streaming encoder with a ``WaitingStreamingEncoder`` of the same settings.

    Reads LeRobot's private ``DatasetWriter._streaming_encoder``, so a
    LeRobot release that renames it fails here, at create, rather than
    silently keeping the encoder that drops frames.
    """
    from lerobot.datasets.video_utils import StreamingVideoEncoder

    from .streaming_encoder import WaitingStreamingEncoder

    writer = dataset.writer
    stock = writer._streaming_encoder
    if type(stock) is not StreamingVideoEncoder:
        raise RuntimeError(f"Expected LeRobot's StreamingVideoEncoder on the dataset writer, found {stock!r}")
    writer._streaming_encoder = WaitingStreamingEncoder(
        fps=stock.fps,
        rgb_encoder=stock._rgb_encoder,
        depth_encoder=stock._depth_encoder,
        queue_maxsize=stock.queue_maxsize,
        encoder_threads=stock._encoder_threads,
    )
    stock.close()


def _check_video_frames(meta: Any, frames: int) -> None:
    """Raise ``EpisodeWriteError`` unless the episode just saved has ``frames`` rows and ``frames`` frames per video.

    LeRobot records each video's span in the episode metadata from the
    encoded file's duration, which is its frame count divided by the fps.
    """
    episode = meta.latest_episode
    counts = {"rows": int(episode["length"][0])}
    for key in meta.video_keys:
        span_s = episode[f"videos/{key}/to_timestamp"][0] - episode[f"videos/{key}/from_timestamp"][0]
        counts[key] = round(span_s * meta.fps)
    wrong = {name: count for name, count in counts.items() if count != frames}
    if wrong:
        raise EpisodeWriteError(
            f"Episode {episode['episode_index'][0]} was given {frames} frames but was saved with {wrong}; "
            "its data rows and videos would be misaligned."
        )
