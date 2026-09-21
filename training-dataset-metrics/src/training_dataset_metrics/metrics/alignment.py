from __future__ import annotations

import numpy as np

from ..core.aggregation import safe_float, scalar_summary, to_json_serializable
from ..core.types import EpisodeData, FeaturePairing, MetricResult
from ._helpers import finite_diff, usable_2d


def compute(
    episodes: list[EpisodeData],
    config: dict | None = None,
    pairing: FeaturePairing | None = None,
) -> MetricResult:
    """Cross-correlation of Δa vs Δs per paired dim, reporting peak lag.

    Emits two artifact payloads:
      - `primary` mirrors HF's Action Insights "State-Action Temporal Alignment"
        panel: explicit Pearson correlation `r(τ) = corr(Δa(t), Δs(t+τ))` on
        each paired dim, aggregated across pairs into max/mean/min envelope
        curves. `max_lag = min(T//4, 30)`. Positive τ ⇒ state responds after
        action (the causal direction for well-behaved teleop).
      - `extended` uses the same sign convention on z-scored raw Δ signals
        for per-pair diagnostics.

    Flag `alignment_fail` when the HF primary envelope's peak lag satisfies
    `|τ*| >= 2` frames, where τ* = argmax_τ |max_envelope(τ)| over pairs.
    """
    per_episode: list[dict] = []
    per_dataset: dict = {}
    flags: list[dict] = []
    extended_by_ep: dict[int, dict] = {}
    primary_by_ep: dict[int, dict] = {}

    pairing_used = pairing or FeaturePairing(method="none")

    peak_lags_all: list[float] = []

    for ep in episodes:
        ep_row: dict = {"episode_index": ep.episode_index}
        if not (usable_2d(ep.state, min_T=4) and usable_2d(ep.action, min_T=4)):
            flags.append({"episode_index": ep.episode_index, "alignment_fail": False})
            per_episode.append(ep_row)
            continue

        ds = finite_diff(ep.state, ep.fps)
        da = finite_diff(ep.action, ep.fps)

        # Reduce to min common length
        T = min(ds.shape[0], da.shape[0])
        ds = ds[:T]
        da = da[:T]

        max_lag = int(max(1, min(T // 4, round(ep.fps) if ep.fps > 0 else T // 4)))

        pair_peaks: list[float] = []
        pair_curves: list[dict] = []

        pairs = _resolve_pairs(ep, pairing_used)
        for s_idx, a_idx, label in pairs:
            x = ds[:, s_idx]
            y = da[:, a_idx]
            if np.std(x) == 0 or np.std(y) == 0:
                continue
            # HF convention: lag τ is how far state *lags* action; positive τ
            # ⇒ action leads. We pass (Δa, Δs) so xcorr(τ) = corr(Δa[t], Δs[t+τ]).
            lags, xcorr = _normalized_xcorr(y, x, max_lag)
            peak_idx = int(np.argmax(np.abs(xcorr)))
            peak_lag = int(lags[peak_idx])
            pair_peaks.append(float(peak_lag))
            pair_curves.append(
                {
                    "pair": label,
                    "lags": lags.tolist(),
                    "xcorr": xcorr.tolist(),
                    "peak_lag": peak_lag,
                    "peak_value": safe_float(xcorr[peak_idx]),
                }
            )

        if pair_peaks:
            ep_row["max_abs_peak_lag"] = float(np.max(np.abs(pair_peaks)))
            ep_row["median_peak_lag"] = float(np.median(pair_peaks))
            peak_lags_all.extend(pair_peaks)
        else:
            ep_row["max_abs_peak_lag"] = float("nan")
            ep_row["median_peak_lag"] = float("nan")

        extended_by_ep[ep.episode_index] = {"pair_curves": pair_curves}

        # HF-parity primary: Pearson on Δa(t) vs Δs(t+τ) per pair, envelope
        # (max/mean/min across pairs). Flag is derived from this envelope.
        primary = _hf_alignment_envelope(ep, pairs)
        envelope_peak_lag: float | None = None
        if primary is not None:
            primary_by_ep[ep.episode_index] = primary
            envelope_peak_lag = _envelope_peak_lag(primary)
            if envelope_peak_lag is not None:
                ep_row["envelope_peak_lag"] = float(envelope_peak_lag)

        alignment_fail = bool(
            envelope_peak_lag is not None and abs(envelope_peak_lag) >= 2
        )
        flags.append(
            {"episode_index": ep.episode_index, "alignment_fail": alignment_fail}
        )
        per_episode.append(ep_row)

    per_dataset["peak_lag"] = scalar_summary(peak_lags_all)
    per_dataset["pairing_method"] = pairing_used.method

    artifacts = {
        "primary": {"by_episode": primary_by_ep},
        "extended": {"by_episode": extended_by_ep},
    }

    return MetricResult(
        name="state_action_alignment",
        per_episode=per_episode,
        per_dataset=per_dataset,
        flags=flags,
        artifacts=to_json_serializable(artifacts),
    )


def _resolve_pairs(ep: EpisodeData, pairing: FeaturePairing) -> list[tuple[int, int, str]]:
    if ep.state_spec is None or ep.action_spec is None:
        return []
    state_index = {n: i for i, n in enumerate(ep.state_spec.names)}
    action_index = {n: i for i, n in enumerate(ep.action_spec.names)}

    pairs: list[tuple[int, int, str]] = []
    if pairing.method in ("dot_path_match", "positional_fallback"):
        for s_name, a_name in pairing.state_to_action.items():
            s_idx = state_index.get(s_name)
            a_idx = action_index.get(a_name)
            if s_idx is not None and a_idx is not None:
                pairs.append((s_idx, a_idx, f"{s_name}↔{a_name}"))
    elif ep.state_spec.dim == ep.action_spec.dim:
        for i in range(ep.state_spec.dim):
            pairs.append((i, i, f"[{i}]↔[{i}]"))
    return pairs


def _normalized_xcorr(
    x: np.ndarray, y: np.ndarray, max_lag: int
) -> tuple[np.ndarray, np.ndarray]:
    xn = (x - x.mean()) / (x.std() or 1.0)
    yn = (y - y.mean()) / (y.std() or 1.0)
    lags = np.arange(-max_lag, max_lag + 1)
    out = np.empty(lags.shape, dtype=float)
    T = len(xn)
    for i, lag in enumerate(lags):
        if lag >= 0:
            n = T - lag
            out[i] = float(np.dot(xn[:n], yn[lag : lag + n])) / max(n, 1)
        else:
            n = T + lag
            out[i] = float(np.dot(xn[-lag : -lag + n], yn[:n])) / max(n, 1)
    return lags, out


def _hf_alignment_envelope(
    ep: EpisodeData, pairs: list[tuple[int, int, str]]
) -> dict | None:
    """Pearson correlation of Δa(t) vs Δs(t+τ) per paired dim, then
    max/mean/min envelope across pairs. Mirrors HF visualizer's panel."""
    if ep.state is None or ep.action is None or not pairs:
        return None
    dx = np.diff(ep.state, axis=0)
    dy = np.diff(ep.action, axis=0)
    T = min(dx.shape[0], dy.shape[0])
    if T < 4:
        return None
    dx = dx[:T]
    dy = dy[:T]
    max_lag = int(min(T // 4, 30))
    if max_lag < 1:
        return None
    lags = list(range(-max_lag, max_lag + 1))
    per_pair: list[np.ndarray] = []
    for s_idx, a_idx, _ in pairs:
        xs = dx[:, s_idx]
        ys = dy[:, a_idx]
        if np.std(xs) == 0 or np.std(ys) == 0:
            continue
        # HF convention: r(τ) = corr(Δa[t], Δs[t+τ]). Positive τ ⇒ state
        # responds after action (action leads).
        per_pair.append(_pearson_lag_curve(ys, xs, max_lag))
    if not per_pair:
        return None
    stack = np.vstack(per_pair)  # (n_pairs, n_lags)
    with np.errstate(invalid="ignore"):
        # Columns that are NaN across every pair (edge lags where a short
        # pair drops out) legitimately aggregate to NaN — not a bug.
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            envelope_max = np.nanmax(stack, axis=0)
            envelope_mean = np.nanmean(stack, axis=0)
            envelope_min = np.nanmin(stack, axis=0)
    return {
        "lags": lags,
        "max": envelope_max.tolist(),
        "mean": envelope_mean.tolist(),
        "min": envelope_min.tolist(),
        "n_pairs": int(stack.shape[0]),
    }


def _envelope_peak_lag(primary: dict) -> float | None:
    """Peak lag from the HF primary envelope: argmax_τ |max_envelope(τ)|.

    The `max` curve is `max_pairs r(τ)` — the best-case correlation across
    paired dims at each lag. Peaking by absolute value handles the rare case
    where the tightest coupling is anti-correlated."""
    lags = primary.get("lags")
    curve = primary.get("max")
    if not lags or not curve:
        return None
    arr = np.asarray(curve, dtype=float)
    if not np.any(np.isfinite(arr)):
        return None
    abs_arr = np.where(np.isfinite(arr), np.abs(arr), -np.inf)
    peak_idx = int(np.argmax(abs_arr))
    return float(lags[peak_idx])


def _pearson_lag_curve(dx: np.ndarray, dy: np.ndarray, max_lag: int) -> np.ndarray:
    """Pearson r of dx(t) vs dy(t+τ) for τ ∈ [-max_lag, +max_lag]."""
    n_lags = 2 * max_lag + 1
    out = np.full(n_lags, np.nan, dtype=float)
    for i, tau in enumerate(range(-max_lag, max_lag + 1)):
        if tau >= 0:
            a = dx[: len(dy) - tau]
            b = dy[tau : tau + len(a)]
        else:
            a = dx[-tau : -tau + (len(dy) + tau)]
            b = dy[: len(a)]
        if a.size < 2:
            continue
        am = a - a.mean()
        bm = b - b.mean()
        denom = float(np.sqrt(np.sum(am * am) * np.sum(bm * bm)))
        if denom < 1e-12:
            continue
        out[i] = float(np.sum(am * bm)) / denom
    return out
