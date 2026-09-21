"""Non-destructive orchestration of the v3.0 -> v2.1 downgrade.

The vendored :mod:`convert_dataset_v30_to_v21` module exposes the individual
conversion steps. Its top-level ``convert_dataset()`` runs them but (a) may
``snapshot_download`` from the Hub and (b) swaps the result in place over the
source tree. The action wants neither: the input is already local and must stay
read-only, and the v2.1 tree belongs in the output directory. So we drive the
building blocks here with explicit source/destination roots.
"""

from __future__ import annotations

import pathlib
from typing import Any

from .convert import convert_dataset_v30_to_v21 as v30_to_v21
from .logger import logger
from .video import convert_videos_frame_exact


def downgrade_v30_to_v21(
    source_root: pathlib.Path, dest_root: pathlib.Path
) -> dict[str, Any]:
    """Write a v2.1 copy of the v3.0 dataset at ``source_root`` into ``dest_root``.

    ``source_root`` is treated as read-only. Returns a small report describing
    the conversion (episode and video-stream counts).
    """
    dest_root.mkdir(parents=True, exist_ok=True)

    episode_records = v30_to_v21.load_episode_records(source_root)
    info = v30_to_v21.load_info(source_root)
    video_keys = [
        key for key, ft in info["features"].items() if ft.get("dtype") == "video"
    ]
    logger.info(
        "Downgrading %d episode(s), %d video stream(s): %s",
        len(episode_records),
        len(video_keys),
        video_keys,
    )

    v30_to_v21.convert_info(source_root, dest_root, episode_records, video_keys)
    v30_to_v21.convert_tasks(source_root, dest_root)
    v30_to_v21.convert_data(source_root, dest_root, episode_records)
    # Frame-exact per-episode split (lossless stream-copy where the boundary is
    # keyframe-aligned; exact re-encode otherwise). Replaces the vendored
    # convert_videos(), whose plain `-c copy` is only keyframe-accurate.
    video_report = convert_videos_frame_exact(
        source_root, dest_root, episode_records, video_keys, fps=int(info["fps"])
    )
    v30_to_v21.convert_episodes_metadata(dest_root, episode_records)
    v30_to_v21.copy_ancillary_directories(source_root, dest_root)

    return {
        "episodes": len(episode_records),
        "video_keys": list(video_keys),
        "video": video_report,
    }
