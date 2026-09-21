from __future__ import annotations

import numpy as np

from ..core.aggregation import safe_float, scalar_summary
from ..core.types import EpisodeData, MetricResult
from ._helpers import autocorrelation, sokal_tau_int, usable_2d


def compute(episodes: list[EpisodeData], config: dict | None = None) -> MetricResult:
    """ESS_d = T / τ_int_d per channel. Report per-episode + dataset total.

    The dataset-total ESS is the single number users should track to size their
    dataset — naive frame counts massively overstate training signal when
    autocorrelation is high.
    """
    per_episode: list[dict] = []
    dataset_ess_per_channel: dict[str, list[float]] = {"state": [], "action": []}
    total_frames = 0
    total_ess_action = 0.0
    total_ess_state = 0.0

    for ep in episodes:
        total_frames += int(ep.n_frames)
        ep_row: dict = {"episode_index": ep.episode_index, "n_frames": int(ep.n_frames)}
        for role, arr in (("state", ep.state), ("action", ep.action)):
            if not usable_2d(arr, min_T=4):
                continue
            T, D = arr.shape
            max_lag = int(min(T // 4, max(1, 2 * round(ep.fps)))) if ep.fps > 0 else T // 4
            if max_lag < 1:
                continue
            acf = autocorrelation(arr, max_lag=max_lag)
            tau_ints = np.asarray(
                [sokal_tau_int(acf[:, d]) for d in range(D)], dtype=float
            )
            ess = np.where(np.isfinite(tau_ints) & (tau_ints > 0), T / tau_ints, np.nan)
            ep_row[f"{role}_ess_median"] = safe_float(np.nanmedian(ess))
            ep_row[f"{role}_ess_min"] = safe_float(np.nanmin(ess))
            ep_row[f"{role}_ess_mean"] = safe_float(np.nanmean(ess))
            dataset_ess_per_channel[role].extend(ess.tolist())
            if role == "action":
                total_ess_action += float(np.nansum(ess))
            else:
                total_ess_state += float(np.nansum(ess))
        per_episode.append(ep_row)

    per_dataset = {
        "total_frames": total_frames,
        "ess_per_action_channel": scalar_summary(dataset_ess_per_channel["action"]),
        "ess_per_state_channel": scalar_summary(dataset_ess_per_channel["state"]),
        # Dataset-wide "total" = sum of per-episode per-channel ESS, averaged across channels.
        "total_ess_action_approx": safe_float(
            total_ess_action
            / max(1, len([x for x in dataset_ess_per_channel["action"] if np.isfinite(x)]))
            if dataset_ess_per_channel["action"]
            else float("nan")
        ),
    }

    return MetricResult(
        name="effective_sample_size",
        per_episode=per_episode,
        per_dataset=per_dataset,
        flags=[{"episode_index": r["episode_index"]} for r in per_episode],
        artifacts={},
    )
