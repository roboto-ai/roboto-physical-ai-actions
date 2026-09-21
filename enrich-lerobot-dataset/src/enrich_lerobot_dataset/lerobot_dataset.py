import pathlib
import shutil
import tempfile

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import INFO_PATH, load_info
from lerobot.datasets.v30.convert_dataset_v21_to_v30 import (
    V21,
    V30,
    convert_dataset,
    validate_local_dataset_version,
)

from .logger import logger


def find_lerobot_dataset_root(directory: pathlib.Path):
    for path in directory.rglob(INFO_PATH):
        return path.parent.parent

    raise FileNotFoundError(
        f"Unable to find root of LeRobot dataset from {directory!s}"
    )


def load_from_directory(directory: pathlib.Path):
    logger.info("Searching for LeRobot dataset root in %s", str(directory))
    dataset_root = find_lerobot_dataset_root(directory)
    logger.info("Found LeRobot dataset root: %s", str(dataset_root))

    return LeRobotDataset(repo_id=dataset_root.name, root=dataset_root)


def _detect_dataset_version(dataset_root: pathlib.Path) -> str:
    """Detect the version of a LeRobot dataset.

    Args:
        dataset_root: Path to the root directory of the LeRobot dataset

    Returns:
        Version string (either "v2.1" or "v3.0")

    Raises:
        FileNotFoundError: If meta/info.json is not found
        ValueError: If codebase_version is not recognized
    """
    try:
        validate_local_dataset_version(dataset_root)
        # If no exception was raised, it's v2.1
        return V21
    except Exception as exc:
        # Not v2.1, check if it's v3.0
        info = load_info(dataset_root)
        version = info.get("codebase_version", "unknown")
        if version == V30:
            return V30
        else:
            raise ValueError(
                f"Unable to work with LeRobot datasets that are not v2.1 or v3.0. Got: {version}"
            ) from exc


def _convert_v21_to_v30(source_root: pathlib.Path):
    """Convert a v2.1 LeRobot dataset to v3.0 format."""
    logger.info(f"Converting v2.1 dataset from {source_root} to v3.0 format...")
    try:
        # converting is destructive, so capture any top level files like action input manifest, etc
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = pathlib.Path(tmp_dir)
            for file in source_root.iterdir():
                if file.is_file():
                    shutil.copy(file, tmp_dir)

            convert_dataset(
                repo_id="",
                root=str(source_root),
                push_to_hub=False,
            )

            # copy any top-levels files back into source_root
            for file in tmp_dir.iterdir():
                shutil.copy(file, source_root)

    except Exception:
        logger.exception("Failed to convert dataset from v2.1 to v3.0")
        raise


def convert_to_v30_if_necessary(dataset_root: pathlib.Path):
    try:
        version = _detect_dataset_version(dataset_root)
        logger.info(f"Detected dataset version: {version}")
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Failed to detect dataset version: {e}")
        raise

    if version == V30:
        return

    _convert_v21_to_v30(dataset_root)
