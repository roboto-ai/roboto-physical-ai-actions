"""Locate and classify a LeRobot dataset within the action's input directory.

Version detection and feature inspection read ``meta/info.json`` directly (plain
JSON) rather than going through lerobot loaders. That keeps these helpers free of
heavy imports and means they never raise on a ``codebase_version`` a loader does
not expect -- the action needs to *recognise* such datasets in order to reject
them with a clear message.
"""

from __future__ import annotations

import json
import pathlib

V21 = "v2.1"
V30 = "v3.0"


def find_lerobot_dataset_root(directory: pathlib.Path) -> pathlib.Path:
    """Return the root of the LeRobot dataset found under ``directory``.

    The root is the parent of the ``meta/`` directory that holds ``info.json``.
    Raises :class:`FileNotFoundError` if no LeRobot dataset is present.
    """
    for info_path in sorted(directory.rglob("info.json")):
        if info_path.parent.name == "meta":
            return info_path.parent.parent
    raise FileNotFoundError(
        f"Unable to find the root of a LeRobot dataset under {directory} "
        "(no meta/info.json was found)."
    )


def _load_info(dataset_root: pathlib.Path) -> dict:
    return json.loads((dataset_root / "meta" / "info.json").read_text())


def detect_codebase_version(dataset_root: pathlib.Path) -> str | None:
    """Return the ``codebase_version`` recorded in ``meta/info.json`` (or None)."""
    return _load_info(dataset_root).get("codebase_version")


def image_in_parquet_features(dataset_root: pathlib.Path) -> list[str]:
    """Return feature keys whose frames are stored as images in the data parquet.

    A v3.0 dataset may carry camera streams either as videos (``dtype: video``,
    which this action de-aggregates) or as images embedded in the data parquet
    (``dtype: image``). The latter are not yet supported, so callers reject a
    dataset when this returns a non-empty list.
    """
    features = _load_info(dataset_root).get("features", {})
    return [key for key, ft in features.items() if ft.get("dtype") == "image"]
