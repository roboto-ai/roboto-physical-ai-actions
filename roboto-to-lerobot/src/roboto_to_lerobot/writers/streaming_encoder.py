"""A LeRobot streaming video encoder that never loses frames.

LeRobot's ``StreamingVideoEncoder`` is built for recording from a live robot,
where stalling the recording loop is worse than losing a frame. When a
camera's frame queue stays full for 0.1 s it drops the frame and logs a
warning, and when an episode's encode takes longer than 120 s to finish it
logs an error and returns the unfinished video. Either way the episode's data
rows and video no longer line up, and nothing fails.

A conversion reads recorded data, so it can always wait for the encoder.
:class:`WaitingStreamingEncoder` waits for queue space and for each encode to
finish, and raises if an encoder thread stops before the episode is done.

This module imports lerobot 0.6 or later at import time; only the v3_0
adapter imports it, inside ``LeRobotWriter.create``.
"""

from __future__ import annotations

import logging
import queue
import time
from pathlib import Path
from typing import Any

import numpy as np
from lerobot.datasets.video_utils import StreamingVideoEncoder

logger = logging.getLogger(__name__)

# How long one blocking ``put`` or ``join`` waits before the encoder thread
# is checked again.
_POLL_S = 1.0
# How often a wait on the encoder is logged, so a long or hung encode shows
# in the logs.
_LOG_EVERY_S = 60.0


class WaitingStreamingEncoder(StreamingVideoEncoder):
    """``StreamingVideoEncoder`` that waits for the encoder instead of losing frames."""

    def feed_frame(self, video_key: str, image: np.ndarray) -> None:
        """Queue a frame for ``video_key``'s encoder, waiting as long as the queue is full.

        Raises:
            RuntimeError: No episode is active, or the encoder thread stopped.
        """
        if not self._episode_active:
            raise RuntimeError("No active episode. Call start_episode() first.")
        self._put(video_key, image.copy())

    def finish_episode(self) -> dict[str, tuple[Path, dict | None]]:
        """Finish every camera's encode, waiting as long as each one takes.

        Returns:
            Each video key's temporary MP4 path and its frame statistics.

        Raises:
            RuntimeError: No episode is active, or an encoder thread failed or
                stopped without a result.
        """
        if not self._episode_active:
            raise RuntimeError("No active episode to finish.")
        for video_key in self._frame_queues:
            self._put(video_key, None)
        results: dict[str, tuple[Path, dict | None]] = {}
        for video_key, thread in self._threads.items():
            started = time.monotonic()
            logged = started
            while thread.is_alive():
                thread.join(timeout=_POLL_S)
                logged = _log_wait(video_key, "to finish the episode's video", started, logged)
            try:
                status, data = self._result_queues[video_key].get_nowait()
            except queue.Empty:
                raise RuntimeError(f"Encoder thread for {video_key} stopped without a result") from None
            if status == "error":
                raise RuntimeError(f"Encoder thread for {video_key} failed: {data}")
            results[video_key] = (self._video_paths[video_key], data)
        self._cleanup()
        self._episode_active = False
        return results

    def _put(self, video_key: str, item: Any) -> None:
        """Put ``item`` on ``video_key``'s frame queue, waiting while its encoder thread runs."""
        frame_queue = self._frame_queues[video_key]
        thread = self._threads[video_key]
        started = time.monotonic()
        logged = started
        while thread.is_alive():
            try:
                frame_queue.put(item, timeout=_POLL_S)
                return
            except queue.Full:
                logged = _log_wait(video_key, "for queue space", started, logged)
        try:
            _, message = self._result_queues[video_key].get_nowait()
        except queue.Empty:
            message = "no error was reported"
        raise RuntimeError(f"Encoder thread for {video_key} stopped before the episode was finished: {message}")


def _log_wait(video_key: str, what: str, started: float, logged: float) -> float:
    """Log a wait on ``video_key``'s encoder once per ``_LOG_EVERY_S``; return when it last logged."""
    now = time.monotonic()
    if now - logged < _LOG_EVERY_S:
        return logged
    logger.info("Waited %.0f s for the %s video encoder %s.", now - started, video_key, what)
    return now
