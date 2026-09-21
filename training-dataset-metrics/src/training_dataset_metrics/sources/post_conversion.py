from __future__ import annotations

import logging
import pathlib
from typing import Any

import numpy as np
import pandas as pd

from ..core.types import EpisodeData, FeatureSpec, SourceDescriptor
from .lerobot_fs import (
    convert_to_v30_if_necessary,
    find_lerobot_dataset_root,
    load_from_directory,
)

logger = logging.getLogger(__name__)


def load_post_conversion_episodes(
    input_dir: pathlib.Path,
) -> tuple[list[EpisodeData], SourceDescriptor, dict[str, Any]]:
    """Walk `input_dir`, find a LeRobot dataset (v2.1 or v3.0), and yield one
    EpisodeData per episode.

    Shape contract between this loader and downstream metrics:
    - `state` is the full `observation.state` feature concatenated as declared by
      `meta/info.json` (shape `(T, D_s)`).
    - `action` is the full `action` feature (shape `(T, D_a)`).
    - Per-dim names come from `info.json["features"][key]["names"]`.
    - `fps` is `info.json["fps"]`.
    """
    dataset_root = find_lerobot_dataset_root(input_dir)
    convert_to_v30_if_necessary(dataset_root)
    dataset = load_from_directory(dataset_root)

    info: dict[str, Any] = dataset.meta.info
    fps = float(info.get("fps", 0.0))
    features = info.get("features", {})

    state_spec = _feature_spec("observation.state", features)
    action_spec = _feature_spec("action", features)

    hf_dataset = dataset.hf_dataset

    source = SourceDescriptor(
        kind="lerobot_dataset",
        identifier=str(dataset_root.name),
        detail={
            "root": str(dataset_root),
            "codebase_version": info.get("codebase_version"),
            "total_episodes": info.get("total_episodes"),
            "total_frames": info.get("total_frames"),
        },
    )

    # hf_dataset columns come back as lists of scalar torch.Tensors in
    # lerobot 0.5.1 — those tensors don't implement value-based __hash__,
    # so pandas groupby would bucket by object id and report one "episode"
    # per distinct tensor object. Materialize everything as plain numpy up
    # front so groupby keys are raw int64 values.
    episode_rows = pd.DataFrame(
        {
            "episode_index": np.asarray(hf_dataset["episode_index"]),
            "frame_index": np.asarray(hf_dataset["frame_index"]),
        }
    )

    state_col = _pull_feature_column(hf_dataset, "observation.state")
    action_col = _pull_feature_column(hf_dataset, "action")
    timestamp_col = _pull_feature_column(hf_dataset, "timestamp")
    task_index_col = _pull_feature_column(hf_dataset, "task_index")

    episodes: list[EpisodeData] = []
    for episode_index, group in episode_rows.groupby("episode_index"):
        order = np.argsort(group["frame_index"].to_numpy())
        row_indices = group.index.to_numpy()[order]

        state_arr = _stack_rows(state_col, row_indices) if state_col is not None else None
        action_arr = _stack_rows(action_col, row_indices) if action_col is not None else None
        ts_arr = _stack_rows(timestamp_col, row_indices) if timestamp_col is not None else None
        task_idx_arr = (
            _stack_rows(task_index_col, row_indices) if task_index_col is not None else None
        )

        n_frames = len(row_indices)
        task_index = int(task_idx_arr[0]) if task_idx_arr is not None and len(task_idx_arr) else None

        episodes.append(
            EpisodeData(
                episode_index=int(episode_index),
                fps=fps,
                n_frames=n_frames,
                state=state_arr,
                state_spec=state_spec,
                action=action_arr,
                action_spec=action_spec,
                timestamps=ts_arr,
                task_index=task_index,
            )
        )

    episodes.sort(key=lambda e: e.episode_index)

    metadata = {
        "state_spec": state_spec,
        "action_spec": action_spec,
        "codebase_version": info.get("codebase_version"),
    }
    return episodes, source, metadata


def _feature_spec(key: str, features: dict[str, Any]) -> FeatureSpec | None:
    info = features.get(key)
    if info is None:
        return None
    names = info.get("names") or []
    if isinstance(names, dict):
        names = [v for _, v in sorted(names.items())]
    shape = info.get("shape") or ()
    dim = int(shape[0]) if shape else len(names)
    stats = info.get("stats") or {}
    declared_min = stats.get("min") if isinstance(stats.get("min"), list) else None
    declared_max = stats.get("max") if isinstance(stats.get("max"), list) else None
    if not names:
        names = [f"{key}.{i}" for i in range(dim)]
    return FeatureSpec(
        key=key,
        names=list(names),
        dim=dim,
        declared_min=declared_min,
        declared_max=declared_max,
    )


def _pull_feature_column(hf_dataset, key: str):
    if key not in hf_dataset.column_names:
        return None
    return np.asarray(hf_dataset[key])


def _stack_rows(column: np.ndarray, row_indices: np.ndarray) -> np.ndarray:
    subset = column[row_indices]
    # hf_dataset returns vector features (observation.state, action) as a
    # list of per-frame tensors; np.asarray wraps that as a 1D object array
    # rather than a 2D float array. Downstream metrics gate on `usable_2d`
    # (ndim == 2), so without this coercion state/action silently skip every
    # per-column check — including `zero_variance_dim` and `stuck_sensor` —
    # giving the false impression of a clean dataset. Scalars (timestamp,
    # task_index) have numeric dtype and pass through unchanged.
    if subset.dtype == object:
        return np.stack([np.asarray(r, dtype=float) for r in subset], axis=0)
    return np.asarray(subset)
