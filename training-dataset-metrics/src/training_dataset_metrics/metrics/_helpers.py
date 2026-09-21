"""Shared signal-processing primitives used by multiple metric modules."""
from __future__ import annotations

import numpy as np
from scipy.signal import correlate


def usable_2d(arr: np.ndarray | None, min_T: int = 2) -> bool:
    """True iff `arr` is a 2D array with at least `min_T` rows. Used by every
    metric module to gate episode-level work."""
    return arr is not None and arr.ndim == 2 and arr.shape[0] >= min_T


def centered(x: np.ndarray) -> np.ndarray:
    return x - np.mean(x, axis=0, keepdims=True)


def autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Biased, normalized ACF per column.

    Returns shape (max_lag + 1, D). Column `d` is NaN everywhere if `std(x_d) == 0`.
    """
    x = np.asarray(x, dtype=float)
    _T, D = x.shape
    out = np.full((max_lag + 1, D), np.nan, dtype=float)
    for d in range(D):
        col = x[:, d]
        sd = np.std(col)
        if sd == 0 or not np.isfinite(sd):
            continue
        c = correlate(col - col.mean(), col - col.mean(), mode="full", method="fft")
        mid = len(c) // 2
        positive = c[mid : mid + max_lag + 1]
        # Biased normalization: divide by T * var (i.e. c(0) == 1)
        denom = positive[0] if positive[0] != 0 else 1.0
        out[:, d] = positive / denom
    return out


def sokal_tau_int(acf_col: np.ndarray) -> float:
    """Integrated autocorrelation time with Sokal window (first negative crossing)."""
    if np.all(np.isnan(acf_col)):
        return float("nan")
    tau = 1.0
    for lag in range(1, len(acf_col)):
        r = acf_col[lag]
        if not np.isfinite(r) or r < 0:
            break
        tau += 2.0 * r
    return float(tau)


def tau_half(acf_col: np.ndarray) -> float:
    """First lag at which ACF drops below 0.5, or NaN if never."""
    for lag in range(1, len(acf_col)):
        if not np.isfinite(acf_col[lag]):
            return float("nan")
        if acf_col[lag] < 0.5:
            return float(lag)
    return float("nan")


def finite_diff(x: np.ndarray, fps: float) -> np.ndarray:
    if x.shape[0] < 2:
        return np.zeros_like(x)
    d = np.diff(x, axis=0) * fps
    # Repeat last row to keep length T for easy downstream alignment.
    return np.vstack([d, d[-1:]])


def sparc(velocity: np.ndarray, fps: float, padlevel: int = 4, fc: float = 10.0) -> float:
    """Spectral Arc Length smoothness. Less negative = smoother.

    `velocity` is 1D (frames,). `fps` is sample rate. `fc` caps evaluation band.
    Implementation follows Balasubramanian et al. 2015.
    """
    v = np.asarray(velocity, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 4 or np.allclose(v, 0):
        return float("nan")

    n = 2 ** int(np.ceil(np.log2(v.size) + padlevel))
    spec = np.abs(np.fft.rfft(v, n=n))
    freqs = np.fft.rfftfreq(n, d=1.0 / fps)
    spec /= np.max(spec) if np.max(spec) > 0 else 1.0

    mask = freqs <= fc
    if mask.sum() < 2:
        return float("nan")
    f = freqs[mask]
    m = spec[mask]
    df = np.diff(f) / (f[-1] if f[-1] != 0 else 1.0)
    dm = np.diff(m)
    arc = -np.sum(np.sqrt(df ** 2 + dm ** 2))
    return float(arc)
