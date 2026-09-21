from __future__ import annotations

import numpy as np

from ..core.aggregation import safe_float
from ..core.types import EpisodeData, MetricResult
from ._helpers import usable_2d


def compute(episodes: list[EpisodeData], config: dict | None = None) -> MetricResult:
    """Episode-length outliers + zero-variance dim detection.

    Also aggregates flags from other metrics at the orchestrator level; here we
    only emit `outlier_length` and `zero_variance_dim`. The CLI cleanup command
    is synthesized at the report level once all metrics have been collected.
    """
    lengths = [int(ep.n_frames) for ep in episodes]
    if lengths:
        median_len = float(np.median(lengths))
        mad = float(np.median(np.abs(np.asarray(lengths) - median_len))) or 1.0
    else:
        median_len = 0.0
        mad = 1.0

    per_episode: list[dict] = []
    flags: list[dict] = []

    for ep, length in zip(episodes, lengths, strict=True):
        outlier = abs(length - median_len) > 3 * mad
        zero_var_dim = False
        for arr in (ep.state, ep.action):
            if not usable_2d(arr):
                continue
            stds = np.std(arr, axis=0)
            # 1e-6 rather than 1e-8: float32-quantized sensor values on a truly
            # dead channel show std ~1e-7 from precision jitter, not 0.0. Any
            # real physical signal has orders of magnitude more variance.
            if np.any(stds < 1e-6):
                zero_var_dim = True
                break
        per_episode.append(
            {
                "episode_index": ep.episode_index,
                "n_frames": length,
                "length_deviation": safe_float(length - median_len),
            }
        )
        flags.append(
            {
                "episode_index": ep.episode_index,
                "outlier_length": bool(outlier),
                "zero_variance_dim": bool(zero_var_dim),
            }
        )

    per_dataset = {
        "median_length": median_len,
        "length_mad": mad,
        "n_episodes": len(episodes),
    }

    return MetricResult(
        name="filtering_flags",
        per_episode=per_episode,
        per_dataset=per_dataset,
        flags=flags,
        artifacts={},
    )


def build_cleanup_command(repo_id: str, flagged_indices: list[int]) -> str | None:
    if not flagged_indices:
        return None
    indices_repr = "[" + ", ".join(str(i) for i in sorted(set(flagged_indices))) + "]"
    return (
        f"lerobot-edit-dataset --repo_id {repo_id} "
        f"--operation.type delete_episodes "
        f"--operation.episode_indices \"{indices_repr}\""
    )
