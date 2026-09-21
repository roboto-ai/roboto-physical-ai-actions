from __future__ import annotations

import numpy as np

from ..core.aggregation import safe_float, to_json_serializable
from ..core.types import EpisodeData, MetricResult
from ._helpers import usable_2d

_HF_BIN_COUNT = 30


def compute(episodes: list[EpisodeData], config: dict | None = None) -> MetricResult:
    """Upstream panel's 5th section: Δa histogram + 'most jerky episodes'.

    Per-dim Δa histograms use HF's 30 equal-width bins on that dim's own
    [min, max] range (no pooling, no quantile trim). Verdict thresholds
    std/maxStd 0.4 / 0.7 → Smooth / Moderate / Jerky.

    Flags `high_action_velocity` on episodes whose mean |Δaction| is a
    dataset-level outlier on the high side (z > 3 vs the median/MAD of all
    episodes' mean_abs_delta), mirroring variance.py's variance_outlier
    convention but one-sided since only unusually *fast* motion should trip
    a flag named "high".
    """
    per_episode: list[dict] = []
    all_deltas_by_dim: list[list[float]] = []
    jerky_rank: list[tuple[float, int]] = []

    max_std_per_dim: dict[int, float] = {}

    for ep in episodes:
        if not usable_2d(ep.action):
            per_episode.append({"episode_index": ep.episode_index})
            continue
        delta = np.diff(ep.action, axis=0)
        sigmas = np.std(delta, axis=0)
        max_abs = np.max(np.abs(delta), axis=0)

        for d, s in enumerate(sigmas):
            max_std_per_dim[d] = max(max_std_per_dim.get(d, 0.0), float(s))

        while len(all_deltas_by_dim) < delta.shape[1]:
            all_deltas_by_dim.append([])
        for d in range(delta.shape[1]):
            all_deltas_by_dim[d].extend(delta[:, d].tolist())

        mean_abs = float(np.mean(np.abs(delta)))
        jerky_rank.append((mean_abs, ep.episode_index))

        per_episode.append(
            {
                "episode_index": ep.episode_index,
                "mean_abs_delta": safe_float(mean_abs),
                "max_abs_delta": safe_float(float(np.max(max_abs))),
                "std_delta_mean": safe_float(float(np.mean(sigmas))),
            }
        )

    verdicts = {str(d): {"max_std": s} for d, s in max_std_per_dim.items()}

    jerky_rank.sort(reverse=True)
    top_jerky = [
        {"episode_index": ep_idx, "mean_abs_delta": safe_float(score)}
        for score, ep_idx in jerky_rank[:10]
    ]

    per_dataset = {
        "top_jerky_episodes": top_jerky,
        "verdict_thresholds": {"smooth_below": 0.4, "moderate_below": 0.7},
    }

    # HF-parity: 30 equal-width bins per-dim over that dim's own range.
    per_dim_hist: list[dict] = []
    for d, deltas in enumerate(all_deltas_by_dim):
        if not deltas:
            per_dim_hist.append({"dim_index": d, "bins": [], "counts": []})
            continue
        arr = np.asarray(deltas)
        lo, hi = float(arr.min()), float(arr.max())
        if hi <= lo:
            hi = lo + 1e-9
        edges = np.linspace(lo, hi, _HF_BIN_COUNT + 1)
        counts, _ = np.histogram(arr, bins=edges)
        per_dim_hist.append(
            {"dim_index": d, "bins": edges.tolist(), "counts": counts.tolist()}
        )

    artifacts = {
        "delta_hist_per_dim": per_dim_hist,
        "verdicts_per_dim": verdicts,
    }

    # Flag episodes whose mean |Δaction| is a dataset-level outlier on the
    # high side, using variance.py's median/MAD z-score convention. One-sided
    # (z > 3, not |z| > 3): the flag name asserts "high", so an unusually
    # *low* mean_abs_delta (e.g. a near-still episode) must not trip it —
    # that direction is already `low_movement`'s job in speed.py.
    mean_abs_deltas = [
        r["mean_abs_delta"] for r in per_episode if r.get("mean_abs_delta") is not None
    ]
    if mean_abs_deltas:
        arr = np.asarray(mean_abs_deltas, dtype=float)
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median))) or 1.0
    else:
        median, mad = 0.0, 1.0

    flags = []
    for r in per_episode:
        mean_abs = r.get("mean_abs_delta")
        if mean_abs is None:
            flags.append(
                {"episode_index": r["episode_index"], "high_action_velocity": False}
            )
            continue
        z = (mean_abs - median) / mad
        flags.append(
            {
                "episode_index": r["episode_index"],
                "high_action_velocity": bool(z > 3),
            }
        )

    return MetricResult(
        name="action_velocity",
        per_episode=per_episode,
        per_dataset=to_json_serializable(per_dataset),
        flags=flags,
        artifacts=to_json_serializable(artifacts),
    )
