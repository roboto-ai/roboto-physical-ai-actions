"""Stable interface for writing a LeRobot dataset across lerobot versions.

The action's ``main.py`` only ever sees the :class:`LeRobotWriter` Protocol
defined here. ``writers/__init__.py`` picks the concrete adapter (one per
lerobot major.minor) at import time by reading ``lerobot.__version__``.

Why a Protocol and not an ABC: the Protocol lets us avoid a base-class
import inside the adapter modules. Each adapter module can import the
lerobot package lazily (inside ``create``) without dragging anything else
along. The Protocol is documentation / type-hint only — Python's
``@runtime_checkable`` does not validate ``@classmethod`` members, so
``isinstance(writer, LeRobotWriter)`` would not actually verify ``create``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol


class LeRobotWriter(Protocol):
    """Minimal write-side surface for a LeRobot dataset.

    Concrete implementations wrap ``lerobot.datasets.lerobot_dataset.LeRobotDataset``
    and translate version-specific call signatures (e.g. how ``task`` is passed
    on ``add_frame``, whether ``finalize`` exists) into this stable shape.
    """

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
        """Create a fresh on-disk dataset rooted at ``root``.

        ``batch_encoding_size`` / ``streaming_encoding`` / ``encoder_threads``
        / ``encoder_queue_maxsize`` are lerobot 0.5.x encoder knobs. The v2_1
        adapter accepts them in its signature but drops them on the call into
        lerobot, since 0.3.x's ``LeRobotDataset.create`` would reject them.
        This keeps a single call shape in ``main.py`` across versions.

        ``vcodec`` is deliberately not in the Protocol: the v3_0 adapter
        locks it to ``"libsvtav1"`` because alternatives degrade downstream
        training quality.
        """
        ...

    def add_frame(self, frame: dict[str, Any], task: str) -> None:
        """Append one frame to the in-memory episode buffer.

        ``task`` is always passed as a kwarg; whether the underlying lerobot
        version expects it inside ``frame`` or as a separate argument is the
        adapter's problem, not the caller's.
        """
        ...

    def save_episode(self) -> None:
        """Flush the current episode to disk and start a new one."""
        ...

    def discard_episode(self) -> None:
        """Discard the in-progress (post-``add_frame``, pre-``save_episode``) buffer.

        Called when a frame iteration or ``save_episode`` raises mid-event so
        the next episode starts from a clean writer state. Must be safe to
        call even when no frames have been added (no-op in that case).
        """
        ...

    def finalize(self) -> None:
        """Finalise the on-disk dataset (write metadata, close handles).

        On lerobot 0.3.x this is a no-op; on 0.5.x it must be called once
        after the last episode.
        """
        ...
