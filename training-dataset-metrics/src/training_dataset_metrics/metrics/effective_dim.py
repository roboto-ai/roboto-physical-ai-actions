from __future__ import annotations

import numpy as np

from ..core.aggregation import safe_float
from ..core.types import EpisodeData, MetricResult
from ._helpers import usable_2d


def compute(episodes: list[EpisodeData], config: dict | None = None) -> MetricResult:
    """PCA participation ratio on pooled state & action.

    PR = (Σλ)² / Σλ²  — counts effective dimensions used.

    `low_effective_dim` (PR / D < 0.3) is a dataset-level verdict, not a
    per-episode one — a single pooled PCA over every episode's data can't
    single out which episode is responsible. It is surfaced only via
    `per_dataset["low_effective_dim_flag"]` (and from there into the report's
    dataset summary); it is deliberately NOT emitted as a per-episode flag,
    so it never broadcasts onto every episode's audit tags.
    """
    per_dataset: dict = {}
    flags: list[dict] = []

    for role in ("state", "action"):
        arr = _pool(episodes, role)
        if arr is None or arr.shape[0] < 2 or arr.shape[1] == 0:
            per_dataset[role] = {"pr": float("nan"), "dim": 0, "pcs_95": 0}
            continue
        eigvals = _pca_eigvals(arr)
        if eigvals.size == 0:
            per_dataset[role] = {"pr": float("nan"), "dim": arr.shape[1], "pcs_95": 0}
            continue
        pr = float((eigvals.sum()) ** 2 / (np.sum(eigvals ** 2) or 1.0))
        cumulative = np.cumsum(eigvals) / (eigvals.sum() or 1.0)
        pcs_95 = int(np.searchsorted(cumulative, 0.95) + 1)
        per_dataset[role] = {
            "pr": safe_float(pr),
            "dim": int(arr.shape[1]),
            "pcs_95": pcs_95,
            "explained_variance_ratio": (eigvals / (eigvals.sum() or 1.0)).tolist(),
        }

    low_effective_dim = False
    for _role, payload in per_dataset.items():
        pr = payload.get("pr", float("nan"))
        dim = payload.get("dim", 0)
        if np.isfinite(pr) and dim > 0 and pr / dim < 0.3:
            low_effective_dim = True

    per_dataset["low_effective_dim_flag"] = low_effective_dim

    # Effective dim is a dataset-level metric; still emit per-episode rows for
    # schema uniformity (empty rows), but no per-episode flags — the verdict
    # above is the only place `low_effective_dim` is surfaced.
    per_episode = [{"episode_index": ep.episode_index} for ep in episodes]
    flags = []

    return MetricResult(
        name="effective_dimensionality",
        per_episode=per_episode,
        per_dataset=per_dataset,
        flags=flags,
        artifacts={},
    )


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


def _pca_eigvals(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=0, keepdims=True)
    cov = np.cov(x, rowvar=False)
    if np.ndim(cov) == 0:
        return np.asarray([float(cov)])
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 0]
    return np.sort(eigvals)[::-1]
