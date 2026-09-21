"""Version-dispatched LeRobot dataset writer.

At import time we read the installed ``lerobot`` package's version and bind
``LeRobotWriter`` to the matching adapter. ``main.py`` imports
``LeRobotWriter`` from here and never touches lerobot directly.
"""

from __future__ import annotations

import re
from importlib.metadata import version as _pkg_version
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import LeRobotWriter as _Protocol


# PEP-440 versions can carry suffixes (``0.5.0+cpu``, ``0.5.0a1``,
# ``0.5.0.dev1``). Pull the leading ``MAJOR.MINOR`` digits so we don't crash
# on pre-releases or local-version installs.
_VERSION_HEAD = re.compile(r"^(\d+)\.(\d+)")


def _select_writer_class(version_str: str) -> type[_Protocol]:
    """Return the adapter class matching a lerobot version string.

    Split out from module-level dispatch so unit tests can drive it without
    re-importing the package.
    """
    match = _VERSION_HEAD.match(version_str)
    if match is None:
        raise RuntimeError(
            f"Could not parse lerobot version: {version_str!r}"
        )
    parts = (int(match.group(1)), int(match.group(2)))
    if parts >= (0, 5):
        from .v3_0 import LeRobotWriter as _Writer
        return _Writer
    if parts >= (0, 3):
        from .v2_1 import LeRobotWriter as _Writer
        return _Writer
    raise RuntimeError(
        f"Unsupported lerobot version: {version_str!r} "
        f"(roboto-to-lerobot supports 0.3.x and 0.5.x)"
    )


LeRobotWriter = _select_writer_class(_pkg_version("lerobot"))

__all__ = ("LeRobotWriter", "_select_writer_class")
