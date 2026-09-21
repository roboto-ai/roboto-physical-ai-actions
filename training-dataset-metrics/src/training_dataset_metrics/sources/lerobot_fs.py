"""Filesystem helpers for locating and loading a LeRobot dataset directory.

Ported inline (rather than imported from a sibling action) because the logic
is short and lets this action stand alone.
"""

from __future__ import annotations

import logging
import pathlib
import shutil
import tempfile

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import INFO_PATH
from lerobot.scripts.convert_dataset_v21_to_v30 import (
    V21,
    V30,
    convert_dataset,
    validate_local_dataset_version,
)

logger = logging.getLogger(__name__)


def find_lerobot_dataset_root(directory: pathlib.Path) -> pathlib.Path:
    """Return the directory containing `meta/info.json` under `directory`."""
    for path in directory.rglob(INFO_PATH):
        return path.parent.parent
    raise FileNotFoundError(
        f"Unable to find a LeRobot dataset under {directory} "
        f"(no '{INFO_PATH}' found)"
    )


def load_from_directory(directory: pathlib.Path) -> LeRobotDataset:
    dataset_root = find_lerobot_dataset_root(directory)
    return LeRobotDataset(repo_id=dataset_root.name, root=dataset_root)


def convert_to_v30_if_necessary(dataset_root: pathlib.Path) -> None:
    """If `dataset_root` is a v2.1 dataset, rewrite it as v3.0 where it sits.

    Always called on the invocation's input directory — the action's own
    downloaded copy, which the runtime discards on exit — so this never
    reaches the dataset stored in Roboto.

    No-ops on v3.0 datasets; raises on unknown versions.
    """
    version = _detect_dataset_version(dataset_root)
    if version == V30:
        return
    _convert_v21_to_v30(dataset_root)


def _detect_dataset_version(dataset_root: pathlib.Path) -> str:
    try:
        validate_local_dataset_version(dataset_root)
        return V21
    except Exception as exc:
        info_path = dataset_root / INFO_PATH
        import json

        info = json.loads(info_path.read_text())
        version = info.get("codebase_version", "unknown")
        if version == V30:
            return V30
        raise ValueError(
            f"Unsupported LeRobot dataset version {version!r}; expected v2.1 or v3.0"
        ) from exc


def _convert_v21_to_v30(source_root: pathlib.Path) -> None:
    """In-place v2.1 → v3.0 conversion, preserving top-level non-meta files."""
    logger.info("Converting v2.1 dataset at %s to v3.0", source_root)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        for file in source_root.iterdir():
            if file.is_file():
                shutil.copy(file, tmp_path)
        convert_dataset(repo_id="", root=str(source_root), push_to_hub=False)
        for file in tmp_path.iterdir():
            shutil.copy(file, source_root)
