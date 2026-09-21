from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from ..core.aggregation import safe_float
from ..core.types import EpisodeData, MetricResult
from ._helpers import usable_2d


def compute(episodes: list[EpisodeData], config: dict | None = None) -> MetricResult:
    """Per-dim range-visited fraction + multivariate k-NN density entropy.

    `(p99 - p01) / declared_range` per state dim when declared ranges are present
    from `info.json["features"][key].stats.{min,max}`.
    """
    pooled_state = _pool(episodes, "state")
    pooled_action = _pool(episodes, "action")

    per_dataset: dict = {}
    spec = _first_state_spec(episodes)

    if pooled_state is not None and pooled_state.shape[0] > 1:
        has_declared = (
            spec is not None
            and spec.declared_min is not None
            and spec.declared_max is not None
        )
        if has_declared:
            p99 = np.percentile(pooled_state, 99, axis=0)
            p01 = np.percentile(pooled_state, 1, axis=0)
            declared = np.asarray(spec.declared_max) - np.asarray(spec.declared_min)
            coverage_frac = np.where(
                declared > 0, (p99 - p01) / declared, np.nan
            ).tolist()
            per_dataset["state"] = {
                "visited_range": (p99 - p01).tolist(),
                "coverage_fraction": coverage_frac,
                "density_entropy": safe_float(_knn_density_entropy(pooled_state)),
            }
        else:
            # Coverage as a fraction requires declared min/max from info.json —
            # without them there is no denominator to normalize by. Surface
            # that as explicit metadata rather than faking a plot.
            per_dataset["state"] = {"declared_range_available": False}

    if pooled_action is not None and pooled_action.shape[0] > 1:
        p99 = np.percentile(pooled_action, 99, axis=0).tolist()
        p01 = np.percentile(pooled_action, 1, axis=0).tolist()
        per_dataset["action"] = {
            "visited_range_p99_p01": [p99, p01],
            "density_entropy": safe_float(_knn_density_entropy(pooled_action)),
        }

    per_episode = [{"episode_index": ep.episode_index} for ep in episodes]
    flags = [{"episode_index": ep.episode_index} for ep in episodes]

    return MetricResult(
        name="state_coverage",
        per_episode=per_episode,
        per_dataset=per_dataset,
        flags=flags,
        artifacts={},
    )


def _first_state_spec(episodes: list[EpisodeData]):
    for ep in episodes:
        if ep.state_spec is not None:
            return ep.state_spec
    return None


def _pool(episodes: list[EpisodeData], role: str) -> np.ndarray | None:
    parts = []
    for ep in episodes:
        arr = ep.state if role == "state" else ep.action
        if not usable_2d(arr):
            continue
        parts.append(arr)
    if not parts:
        return None
    widths = {p.shape[1] for p in parts}
    if len(widths) > 1:
        w = min(widths)
        parts = [p[:, :w] for p in parts]
    return np.concatenate(parts, axis=0)


def _knn_density_entropy(x: np.ndarray, k: int = 5, sample_cap: int = 5000) -> float:
    """Entropy of the histogram of k-NN distances. Uniform → high; clustered → low."""
    if x.shape[0] > sample_cap:
        rng = np.random.default_rng(0)
        x = x[rng.choice(x.shape[0], sample_cap, replace=False)]
    n = x.shape[0]
    k = min(k, max(1, n - 1))
    try:
        nn = NearestNeighbors(n_neighbors=k + 1).fit(x)
        dists, _ = nn.kneighbors(x)
    except ValueError:
        return float("nan")
    kth = dists[:, -1]
    kth = kth[np.isfinite(kth) & (kth > 0)]
    if kth.size < 4:
        return float("nan")
    hist, _ = np.histogram(kth, bins=min(30, max(4, kth.size // 10)), density=True)
    hist = hist[hist > 0]
    hist = hist / hist.sum()
    return float(-np.sum(hist * np.log(hist)) / np.log(len(hist) or 2))
