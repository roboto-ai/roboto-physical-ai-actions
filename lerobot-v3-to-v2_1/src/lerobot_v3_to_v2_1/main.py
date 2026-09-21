import shutil

import roboto

from .downgrade import downgrade_v30_to_v21
from .lerobot_dataset import (
    V21,
    V30,
    detect_codebase_version,
    find_lerobot_dataset_root,
    image_in_parquet_features,
)
from .logger import logger
from .roboto_io import publish_converted_dataset


def main(context: roboto.InvocationContext) -> None:
    logger.setLevel(context.log_level)

    source_root = find_lerobot_dataset_root(context.input_dir)
    version = detect_codebase_version(source_root)
    logger.info(
        "Found LeRobot dataset at %s (codebase_version=%s)", source_root, version
    )

    if version not in (V21, V30):
        raise ValueError(
            f"Unsupported LeRobot codebase_version {version!r}; this action "
            f"converts {V30} datasets to {V21}."
        )
    if version == V30:
        image_features = image_in_parquet_features(source_root)
        if image_features:
            raise ValueError(
                "This dataset stores camera frames as images in the data parquet "
                f"(features with dtype=image: {image_features}). Only video-backed "
                "v3.0 datasets are supported; image-in-parquet conversion is not yet "
                "implemented."
            )

    # Stage the v2.1 tree under output_dir (provisioned storage). We upload it to a
    # dedicated dataset ourselves and then clear it, so the platform's auto-upload of
    # output_dir is a no-op and the v2.1 tree never lands back in the source dataset.
    staging_root = context.output_dir / source_root.name
    if version == V21:
        logger.info("Dataset is already v2.1; passing it through unchanged.")
        shutil.copytree(source_root, staging_root, dirs_exist_ok=True)
        report: dict = {"episodes": None, "passthrough": True}
    else:
        logger.info("Converting %s -> %s ...", V30, V21)
        report = downgrade_v30_to_v21(source_root, staging_root)

    if context.is_dry_run:
        logger.info(
            "Dry-run: skipping dataset creation/upload. v2.1 tree left at %s",
            staging_root,
        )
        return

    target_dataset_id = context.get_optional_parameter("output_dataset_id")
    dataset = publish_converted_dataset(
        context, staging_root, report, version, target_dataset_id=target_dataset_id
    )
    # Keep output_dir empty so the platform's auto-upload cannot re-dump the tree into
    # the source dataset — we've already uploaded it to the dedicated dataset.
    shutil.rmtree(staging_root)
    logger.info("Done. Converted v2.1 dataset: %s", dataset.dataset_id)
