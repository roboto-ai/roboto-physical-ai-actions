"""Publish the converted v2.1 dataset back to Roboto.

By default the action creates a NEW dataset (name inherited from the source) and
uploads the v2.1 tree into it, stamping provenance metadata that links back to the
source dataset. An optional ``output_dataset_id`` targets an existing dataset
instead. The source dataset is never modified.

The heavy SDK imports (``roboto``, ``lerobot``) are deferred into
:func:`publish_converted_dataset` so the pure helpers below stay importable — and
unit-testable — without them.
"""

from __future__ import annotations

import pathlib
from typing import Any

from .logger import logger

VENDORED_COMMIT = "2ef2370d66"
ACTION_NAME = "lerobot-v3-to-v2_1"


def derive_output_name(source_name: str | None, source_dataset_id: str) -> str:
    """Name for the converted dataset, inherited from the source where possible."""
    base = source_name.strip() if source_name and source_name.strip() else None
    if base:
        return f"{base} (LeRobot v2.1)"
    return f"LeRobot v2.1 conversion of {source_dataset_id}"


def build_provenance(
    *,
    source_dataset_id: str,
    source_dataset_name: str | None,
    source_codebase_version: str,
    invocation_id: str,
    lerobot_version: str,
    report: dict[str, Any],
) -> dict[str, Any]:
    """Provenance to stamp on the converted dataset.

    Top-level keys are convenient for RoboQL filtering; the per-invocation block
    (``invocations.<id>``) carries the full detail, matching the convention the
    other actions in this repo use.
    """
    detail = {
        "action": ACTION_NAME,
        "source_dataset_id": source_dataset_id,
        "source_dataset_name": source_dataset_name,
        "source_codebase_version": source_codebase_version,
        "target_codebase_version": "v2.1",
        "lerobot_version": lerobot_version,
        "vendored_commit": VENDORED_COMMIT,
        "episodes": report.get("episodes"),
        "video": report.get("video"),
    }
    return {
        "source_dataset_id": source_dataset_id,
        "source_dataset_name": source_dataset_name,
        "codebase_version": "v2.1",
        "converted_by": ACTION_NAME,
        f"invocations.{invocation_id}": detail,
    }


def publish_converted_dataset(
    context,
    local_root: pathlib.Path,
    report: dict[str, Any],
    source_codebase_version: str,
    *,
    target_dataset_id: str | None = None,
):
    """Create-or-target the output dataset, upload ``local_root``, stamp provenance.

    Returns the destination :class:`roboto.Dataset`.
    """
    import lerobot
    import roboto
    from roboto.updates import MetadataChangeset

    source_dataset_id = context.dataset_id
    source_dataset_name = getattr(context.dataset, "name", None)

    if target_dataset_id:
        dataset = roboto.Dataset.from_id(target_dataset_id)
        logger.info(
            "Uploading converted dataset into existing dataset %s", target_dataset_id
        )
    else:
        name = derive_output_name(source_dataset_name, source_dataset_id)
        description = f"LeRobot v2.1 conversion of dataset {source_dataset_id}"
        if source_dataset_name:
            description += f" ({source_dataset_name})"
        dataset = roboto.Dataset.create(
            name=name,
            description=description + ".",
            tags=["lerobot", "lerobot-v2.1", "converted"],
            caller_org_id=context.org_id,
        )
        logger.info("Created output dataset %s (%r)", dataset.dataset_id, name)

    dataset.upload_directory(pathlib.Path(local_root))
    logger.info("Uploaded v2.1 tree to dataset %s", dataset.dataset_id)

    provenance = build_provenance(
        source_dataset_id=source_dataset_id,
        source_dataset_name=source_dataset_name,
        source_codebase_version=source_codebase_version,
        invocation_id=context.invocation_id,
        lerobot_version=lerobot.__version__,
        report=report,
    )
    builder = MetadataChangeset.Builder()
    for key, value in provenance.items():
        builder = builder.put_field(key, value)
    dataset.update(metadata_changeset=builder.build())
    logger.info("Stamped provenance on dataset %s", dataset.dataset_id)

    return dataset
