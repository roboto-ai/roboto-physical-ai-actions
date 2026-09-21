from __future__ import annotations

import numpy as np

from ..core.aggregation import safe_float, scalar_summary, to_json_serializable
from ..core.types import EpisodeData, MetricResult
from ._helpers import autocorrelation, sokal_tau_int, tau_half, usable_2d


def compute(episodes: list[EpisodeData], config: dict | None = None) -> MetricResult:
    """Centered, normalized ACF per channel per episode.

    Emits two artifact payloads:
      - `primary` mirrors the HF Action Insights "Action Autocorrelation" panel:
        actions-only, `maxLag = min(T//2, 100)`, divisor `Σ centered²`.
      - `extended` keeps the state+action τ_int / τ_half summary for diagnostics.

    Flags `stuck_sensor` when r(1) > 0.999 for ≥ 3 consecutive lags on any state
    channel (sensor likely saturated or held).
    """
    per_episode: list[dict] = []
    per_dataset: dict = {}
    flags: list[dict] = []

    tau_int_all: list[float] = []
    tau_half_all: list[float] = []
    extended_curves: dict[int, dict[str, dict]] = {}
    primary_curves: dict[int, dict] = {}

    for ep in episodes:
        signals = {"state": ep.state, "action": ep.action}
        names = {
            "state": ep.state_spec.names if ep.state_spec else [],
            "action": ep.action_spec.names if ep.action_spec else [],
        }

        ep_row: dict = {"episode_index": ep.episode_index}
        stuck_flag = False

        for role, arr in signals.items():
            if not usable_2d(arr, min_T=4):
                continue
            T = arr.shape[0]
            max_lag = int(min(T // 4, max(1, 2 * round(ep.fps)))) if ep.fps > 0 else T // 4
            if max_lag < 1:
                continue
            acf = autocorrelation(arr, max_lag=max_lag)  # (max_lag+1, D)
            tau_ints = [safe_float(sokal_tau_int(acf[:, d])) for d in range(acf.shape[1])]
            tau_halves = [safe_float(tau_half(acf[:, d])) for d in range(acf.shape[1])]

            ep_row[f"{role}_tau_int_median"] = safe_float(np.nanmedian(tau_ints))
            ep_row[f"{role}_tau_half_median"] = safe_float(np.nanmedian(tau_halves))
            ep_row[f"{role}_tau_int_max"] = safe_float(np.nanmax(tau_ints))

            tau_int_all.extend(tau_ints)
            tau_half_all.extend(tau_halves)

            if role == "state":
                # Near-zero-variance state columns ⇒ stuck sensor. The FFT-based
                # ACF returns NaN for such columns, so check directly on the
                # signal. Threshold 1e-6: float32 quantization on a truly dead
                # channel shows std ~1e-7, not exact zero.
                if np.any(np.std(arr, axis=0) < 1e-6):
                    stuck_flag = True
                for d in range(acf.shape[1]):
                    col = acf[:, d]
                    # r(1..3) all > 0.999 → stuck sensor
                    if len(col) > 3 and np.all(col[1:4] > 0.999):
                        stuck_flag = True

            extended_curves.setdefault(ep.episode_index, {})[role] = {
                "lags": list(range(max_lag + 1)),
                "names": list(names[role]),
                "curves": acf.T.tolist(),
            }

        # HF-parity primary payload: actions only, maxLag cap 100, Σx² divisor.
        if usable_2d(ep.action, min_T=4):
            T = ep.action.shape[0]
            primary_max_lag = int(min(T // 2, 100))
            if primary_max_lag >= 1:
                primary_curves[ep.episode_index] = {
                    "lags": list(range(primary_max_lag + 1)),
                    "names": list(names["action"]),
                    "curves": [
                        _hf_autocorrelation(ep.action[:, d], primary_max_lag).tolist()
                        for d in range(ep.action.shape[1])
                    ],
                }

        per_episode.append(ep_row)
        flags.append(
            {"episode_index": ep.episode_index, "stuck_sensor": bool(stuck_flag)}
        )

    per_dataset["tau_int"] = scalar_summary(tau_int_all)
    per_dataset["tau_half"] = scalar_summary(tau_half_all)

    artifacts = {
        "primary": {"curves_by_episode": primary_curves},
        "extended": {"curves_by_episode": extended_curves},
    }

    return MetricResult(
        name="autocorrelation",
        per_episode=per_episode,
        per_dataset=per_dataset,
        flags=flags,
        artifacts=to_json_serializable(artifacts),
    )


def _hf_autocorrelation(col: np.ndarray, max_lag: int) -> np.ndarray:
    """HF visualizer parity: r(k) = Σ x[t]·x[t+k] / Σ x[t]², with x centered."""
    x = np.asarray(col, dtype=float)
    x = x - x.mean()
    denom = float(np.sum(x * x))
    if denom <= 0 or not np.isfinite(denom):
        return np.full(max_lag + 1, np.nan, dtype=float)
    out = np.empty(max_lag + 1, dtype=float)
    for k in range(max_lag + 1):
        n = len(x) - k
        if n <= 0:
            out[k] = np.nan
        else:
            out[k] = float(np.sum(x[:n] * x[k:])) / denom
    return out
