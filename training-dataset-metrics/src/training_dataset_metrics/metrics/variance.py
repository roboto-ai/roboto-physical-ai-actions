from __future__ import annotations

import numpy as np

from ..core.aggregation import (
    resample_to_time_bins,
    safe_float,
    to_json_serializable,
)
from ..core.types import EpisodeData, MetricResult
from ._helpers import finite_diff, usable_2d

_HF_TIME_BINS = 50


def compute(episodes: list[EpisodeData], config: dict | None = None) -> MetricResult:
    """Cross-episode variance heatmap.

    Emits two artifact payloads:
      - `primary` mirrors HF's Action Insights "Cross-Episode Action Variance"
        panel: each episode is nearest-neighbor-resampled to 50 normalized time
        bins; the panel heatmap is `(D_a × 50)` population variance **across
        episodes** at each `(dim, time_bin)`. Stored as raw variance — renderer
        sqrt-remaps for color only (matches HF).
      - `extended` keeps our (episode × dim) per-episode std heatmaps for
        position / speed / acceleration on both state and action.
    """
    per_episode: list[dict] = []
    flags: list[dict] = []

    extended_heatmaps: dict[str, dict] = {}
    for role in ("state", "action"):
        for kind in ("position", "speed", "acceleration"):
            matrix, ep_indices, names = _build_matrix(episodes, role, kind)
            if matrix is None:
                continue
            normalized = _normalize_by_column_median(matrix)
            extended_heatmaps[f"{role}_{kind}"] = {
                "episode_index": ep_indices,
                "names": names,
                "matrix": normalized.tolist(),
            }

    # HF-parity primary heatmap: action, time-binned cross-episode variance.
    primary = _hf_time_binned_variance(episodes)

    # Per-episode summaries computed from the action-position extended heatmap
    # as primary (unchanged behavior: z-score + near-zero-columns fraction).
    primary_ext = extended_heatmaps.get("action_position") or extended_heatmaps.get(
        "state_position"
    )
    if primary_ext:
        matrix = np.asarray(primary_ext["matrix"], dtype=float)
        ep_indices = primary_ext["episode_index"]
        median_row = np.nanmedian(matrix, axis=0)
        deltas = matrix - median_row
        norms = np.linalg.norm(np.nan_to_num(deltas), axis=1)
        mad = float(np.median(np.abs(norms - np.median(norms))) or 1.0)
        z_scores = (norms - np.median(norms)) / mad
        near_zero_frac = np.mean(matrix < 0.1, axis=1)

        for ep_idx, z, nz in zip(ep_indices, z_scores, near_zero_frac, strict=False):
            per_episode.append(
                {
                    "episode_index": int(ep_idx),
                    "variance_z": safe_float(z),
                    "near_zero_columns_fraction": safe_float(nz),
                }
            )
            flags.append(
                {
                    "episode_index": int(ep_idx),
                    "variance_outlier": bool(abs(z) > 3),
                }
            )
    else:
        for ep in episodes:
            per_episode.append({"episode_index": ep.episode_index})
            flags.append(
                {"episode_index": ep.episode_index, "variance_outlier": False}
            )

    artifacts = {
        "primary": primary,
        "extended": {"heatmaps": extended_heatmaps},
    }

    return MetricResult(
        name="cross_episode_variance",
        per_episode=per_episode,
        per_dataset={
            "heatmap_kinds": list(extended_heatmaps.keys()),
            "primary_shape": (
                list(np.asarray(primary["matrix"]).shape)
                if primary and primary.get("matrix") is not None
                else None
            ),
        },
        flags=flags,
        artifacts=to_json_serializable(artifacts),
    )


def _hf_time_binned_variance(episodes: list[EpisodeData]) -> dict:
    """Nearest-neighbor resample each episode's action matrix to
    `_HF_TIME_BINS` samples, then compute population variance across episodes
    per `(dim, bin)`. Returns a dict with a `(D × 50)` raw variance matrix;
    the renderer sqrt-remaps for color only, matching HF."""
    usable: list[np.ndarray] = []
    dim: int | None = None
    names: list[str] = []
    for ep in episodes:
        if not usable_2d(ep.action):
            continue
        if dim is None:
            dim = ep.action.shape[1]
            names = list(ep.action_spec.names) if ep.action_spec else []
        if ep.action.shape[1] != dim:
            continue
        usable.append(ep.action)
    if not usable or dim is None:
        return {
            "matrix": [],
            "dim_names": [],
            "time_bins": _HF_TIME_BINS,
        }
    # (n_episodes, time_bins, dim)
    resampled = np.stack(
        [resample_to_time_bins(arr, _HF_TIME_BINS) for arr in usable], axis=0
    )
    # Population variance across episodes per (dim, bin), HF convention.
    var = np.var(resampled, axis=0, ddof=0)  # (time_bins, dim)
    return {
        "matrix": var.T.tolist(),  # (dim, time_bins), raw variance
        "dim_names": names[:dim] if names else [f"dim_{i}" for i in range(dim)],
        "time_bins": _HF_TIME_BINS,
        "n_episodes": int(resampled.shape[0]),
    }


def _build_matrix(
    episodes: list[EpisodeData], role: str, kind: str
) -> tuple[np.ndarray | None, list[int], list[str]]:
    ep_indices: list[int] = []
    rows: list[np.ndarray] = []
    names: list[str] = []
    for ep in episodes:
        arr = ep.state if role == "state" else ep.action
        spec = ep.state_spec if role == "state" else ep.action_spec
        if not usable_2d(arr):
            continue
        if kind == "speed":
            arr = finite_diff(arr, ep.fps)
        elif kind == "acceleration":
            arr = finite_diff(finite_diff(arr, ep.fps), ep.fps)
        std = np.std(arr, axis=0)
        rows.append(std)
        ep_indices.append(ep.episode_index)
        if spec and not names:
            names = list(spec.names[: std.size])
    if not rows:
        return None, [], []
    widths = {r.size for r in rows}
    if len(widths) > 1:
        width = min(widths)
        rows = [r[:width] for r in rows]
    return np.vstack(rows), ep_indices, names


def _normalize_by_column_median(matrix: np.ndarray) -> np.ndarray:
    med = np.nanmedian(matrix, axis=0)
    med = np.where(med == 0, 1.0, med)
    return matrix / med
