from __future__ import annotations

import math

import numpy as np

from ..core.aggregation import safe_float, scalar_summary, to_json_serializable
from ..core.types import EpisodeData, MetricResult
from ._helpers import finite_diff, sparc, usable_2d


def compute(episodes: list[EpisodeData], config: dict | None = None) -> MetricResult:
    """Per-channel |velocity| histogram, stillness ratio, jerk, SPARC.

    Emits two artifact payloads:
      - `primary` mirrors HF's Action Insights "Demonstrator Speed Variance"
        panel: one scalar per episode = `mean_t(||a[t]-a[t-1]||₂)` (no fps
        scaling), histogrammed across episodes with `min(30, ⌈√n⌉)` bins.
      - `extended` keeps our per-role per-channel |Δx|·fps bins, stillness,
        jerk, and SPARC stats.

    Flags `low_movement` on episodes whose stillness > 0.9 on any action dim.
    """
    per_episode: list[dict] = []
    flags: list[dict] = []

    # First pass: compute per-episode velocity and jerk arrays.
    per_ep_velocity: dict[int, dict[str, np.ndarray]] = {}
    per_ep_jerk: dict[int, dict[str, np.ndarray]] = {}

    all_abs_velocity: list[float] = []

    # HF-parity: one scalar per episode = mean over t of L2 norm of Δa (no fps).
    per_ep_hf_speed: list[tuple[int, float]] = []

    for ep in episodes:
        per_ep_velocity[ep.episode_index] = {}
        per_ep_jerk[ep.episode_index] = {}
        for role, arr in (("state", ep.state), ("action", ep.action)):
            if not usable_2d(arr):
                continue
            v = finite_diff(arr, ep.fps)
            j = finite_diff(v, ep.fps)
            per_ep_velocity[ep.episode_index][role] = v
            per_ep_jerk[ep.episode_index][role] = j
            if role == "action":
                all_abs_velocity.extend(np.abs(v).ravel().tolist())

        if usable_2d(ep.action):
            delta = np.diff(ep.action, axis=0)
            hf_speed = float(np.mean(np.linalg.norm(delta, axis=1)))
            per_ep_hf_speed.append((ep.episode_index, hf_speed))

    # Pooled IQR for stillness ε.
    if all_abs_velocity:
        v_arr = np.asarray(all_abs_velocity)
        iqr = float(np.percentile(v_arr, 75) - np.percentile(v_arr, 25))
    else:
        iqr = 1.0
    eps = 1e-3 * (iqr if iqr > 0 else 1.0)

    # Shared histogram bins from the pooled action velocity distribution.
    if all_abs_velocity:
        low, high = np.percentile(all_abs_velocity, [0.1, 99.9])
        if high <= low:
            high = low + 1.0
        bins = np.linspace(low, high, 51)
    else:
        bins = np.linspace(0, 1, 51)

    dataset_jerk_p95: list[float] = []

    for ep in episodes:
        v_map = per_ep_velocity[ep.episode_index]
        j_map = per_ep_jerk[ep.episode_index]
        ep_row: dict = {"episode_index": ep.episode_index}
        low_movement = False

        for role, v in v_map.items():
            abs_v = np.abs(v)
            ep_row[f"{role}_mean_speed"] = safe_float(np.mean(abs_v))
            ep_row[f"{role}_median_speed"] = safe_float(np.median(abs_v))
            ep_row[f"{role}_p95_speed"] = safe_float(np.percentile(abs_v, 95))
            ep_row[f"{role}_max_speed"] = safe_float(np.max(abs_v))

            stillness_per_dim = np.mean(abs_v < eps, axis=0)
            ep_row[f"{role}_stillness_max_dim"] = safe_float(np.max(stillness_per_dim))
            ep_row[f"{role}_stillness_mean"] = safe_float(np.mean(stillness_per_dim))
            if role == "action" and np.max(stillness_per_dim) > 0.9:
                low_movement = True

            if role in j_map:
                abs_j = np.abs(j_map[role])
                p95_j = float(np.percentile(abs_j, 95))
                ep_row[f"{role}_p95_jerk"] = safe_float(p95_j)
                if role == "action":
                    dataset_jerk_p95.append(p95_j)
                # SPARC per dim (averaged to scalar)
                sparcs = [sparc(v[:, d], ep.fps) for d in range(v.shape[1])]
                ep_row[f"{role}_sparc_mean"] = safe_float(
                    np.nanmean([s for s in sparcs if np.isfinite(s)])
                )

        # Stamp the HF-parity scalar onto the per-episode row too, so users
        # see it in the per-episode table.
        hf_speed_lookup = dict(per_ep_hf_speed)
        if ep.episode_index in hf_speed_lookup:
            ep_row["hf_action_speed"] = safe_float(hf_speed_lookup[ep.episode_index])

        per_episode.append(ep_row)
        flags.append(
            {"episode_index": ep.episode_index, "low_movement": low_movement}
        )

    # HF-parity dataset verdict: CV of the per-episode L2-step scalar.
    if per_ep_hf_speed:
        scalars = np.asarray([v for _, v in per_ep_hf_speed], dtype=float)
        mu = float(np.mean(scalars))
        sigma = float(np.std(scalars))
        cv = sigma / mu if mu > 0 else float("nan")
    else:
        scalars = np.empty(0, dtype=float)
        cv = float("nan")

    if not np.isfinite(cv):
        verdict = "unknown"
    elif cv < 0.2:
        verdict = "consistent"
    elif cv < 0.4:
        verdict = "moderate"
    else:
        verdict = "high_variance"

    # HF-parity histogram: min(30, ⌈√n⌉) equal-width bins on the observed range.
    hf_hist: dict = {"bins": [], "counts": []}
    if scalars.size > 0:
        n_bins = max(1, min(30, math.ceil(math.sqrt(scalars.size))))
        lo = float(np.min(scalars))
        hi = float(np.max(scalars))
        if hi <= lo:
            hi = lo + 1e-9
        edges = np.linspace(lo, hi, n_bins + 1)
        counts, _ = np.histogram(scalars, bins=edges)
        hf_hist = {"bins": edges.tolist(), "counts": counts.tolist()}

    per_dataset = {
        "action_speed_cv": safe_float(cv),
        "verdict": verdict,
        "stillness_eps": safe_float(eps),
        "hf_action_speed": scalar_summary(scalars.tolist()),
        "action_jerk_p95": scalar_summary(dataset_jerk_p95),
    }

    # Jerky threshold: dataset p95-median — flag jerky episodes whose p95 jerk > 3×
    if dataset_jerk_p95:
        median_p95 = float(np.median(dataset_jerk_p95))
        for i, _ep in enumerate(episodes):
            p95 = per_episode[i].get("action_p95_jerk", float("nan"))
            flags[i]["jerky_motion"] = bool(
                np.isfinite(p95) and median_p95 > 0 and p95 > 3 * median_p95
            )
    else:
        for i in range(len(episodes)):
            flags[i]["jerky_motion"] = False

    artifacts = {
        "primary": {
            "per_episode_scalar": [
                {"episode_index": idx, "value": safe_float(val)}
                for idx, val in per_ep_hf_speed
            ],
            "hist": hf_hist,
            "verdict": verdict,
            "median": safe_float(float(np.median(scalars))) if scalars.size else float("nan"),
        },
        "extended": {
            "bins": bins.tolist(),
            "verdict_thresholds": {"consistent_below": 0.2, "moderate_below": 0.4},
        },
    }

    return MetricResult(
        name="speed_distribution",
        per_episode=per_episode,
        per_dataset=per_dataset,
        flags=flags,
        artifacts=to_json_serializable(artifacts),
    )
