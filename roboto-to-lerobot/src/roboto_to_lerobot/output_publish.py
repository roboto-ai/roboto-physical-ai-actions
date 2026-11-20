"""Upload a finished conversion and tag its dataset, in that order.

The runtime uploads the output directory only after the action exits, too late
for the action to tag the dataset afterwards. A trigger on ``dataset.tag_added``
needs every file in place when it fires, so when ``output_tags`` is set the
action uploads the output itself and then applies the tags.
"""

from __future__ import annotations

import pathlib
import shutil
from typing import Any

from .logger import logger


def parse_tags(raw: object) -> list[str]:
    """Split a comma-separated parameter value into its non-blank tags."""
    return [tag.strip() for tag in str(raw or "").split(",") if tag.strip()]


def publish_tagged_output(
    dataset: Any,
    output_dir: pathlib.Path,
    subfolder: pathlib.Path,
    tags: list[str],
) -> None:
    """Upload ``subfolder`` to ``dataset``, then apply ``tags``.

    Args:
        dataset: The dataset the output belongs in.
        output_dir: The invocation's output directory; destination paths are relative to it,
            as they are when the runtime uploads.
        subfolder: The directory to upload. It is removed, so the runtime has nothing left
            to upload after the tags have fired.
        tags: The tags to apply. A tag the dataset already has is removed first, since adding
            it again would otherwise publish no ``dataset.tag_added`` event.
    """
    files = [path for path in subfolder.rglob("*") if path.is_file()]
    dataset.upload_files(
        files,
        file_destination_paths={path: str(path.relative_to(output_dir)) for path in files},
        print_progress=False,
    )
    shutil.rmtree(subfolder)

    already_tagged = [tag for tag in tags if tag in dataset.tags]
    if already_tagged:
        dataset.remove_tags(already_tagged)
    dataset.put_tags(tags)
    logger.info("Uploaded %d file(s), then tagged the dataset %s.", len(files), tags)
