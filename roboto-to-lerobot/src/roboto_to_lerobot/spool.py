"""Camera payload bytes kept in a local file while an episode is built and written.

An episode's camera messages are most of its size: a minute of one JPEG camera
at 30 fps is about 100 MB of encoded frames, a raw ``sensor_msgs/Image``
camera several GB. Held in memory, those bytes are alive in the worker from
the fetch until the episode is returned, then copied into the pickle that
carries the episode to the writer process, then unpickled there, so a
worker and the writer together hold about three times the payload.

With a spool, each fetch writes its camera payloads to the episode's spool
file as soon as it returns and keeps a :py:class:`SpoolRef` (the file, an
offset, a length) in place of each one. Alignment, the worker-to-writer
handoff and the drain move those references, and the writer reads one
frame's bytes back just before decoding it.

A reference also records the payload's Python type, and reading it back
returns that type. The decoders branch on that type: the SDK's MCAP
reader returns ``uint8[]`` fields as ``array.array('B')``, on which the
PNG/JPEG signature checks in ``extract.py`` and ``runtime/decoders.py``
(``data[:8] == b"..."``) are always false, so the same bytes as ``bytes``
would take a different branch.
"""

from __future__ import annotations

import array
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ("SpoolRef", "SpoolWriter", "unspool")


@dataclass(frozen=True, slots=True)
class SpoolRef:
    """Where one payload's bytes are in a spool file, and the type to read them back as.

    ``container`` is ``bytes``, ``array.array`` or ``numpy.ndarray``;
    ``item_type`` is the array's typecode or the ndarray's dtype string, and
    empty for ``bytes``. A ``bytearray`` or ``memoryview`` payload reads back
    as ``bytes``, which is what ``lerobot.generate_frames`` turns it into
    when the payload is not spooled.
    """

    path: str
    offset: int
    length: int
    container: type = bytes
    item_type: str = ""

    def read(self) -> Any:
        """Return the payload as the type it was written from.

        Raises:
            OSError: If the spool file is gone or shorter than the reference.
        """
        fd = os.open(self.path, os.O_RDONLY)
        try:
            if self.container is np.ndarray:
                # Read into a writable buffer, so the array is writable as the
                # SDK's arrays are.
                buffer = bytearray(self.length)
                bytes_read = os.preadv(fd, [buffer], self.offset)
                data: Any = buffer
            else:
                data = os.pread(fd, self.length, self.offset)
                bytes_read = len(data)
        finally:
            os.close(fd)
        if bytes_read != self.length:
            raise OSError(
                f"Spool file {self.path} holds {bytes_read} of the {self.length} bytes at offset {self.offset}."
            )
        if self.container is array.array:
            values = array.array(self.item_type)
            values.frombytes(data)
            return values
        if self.container is np.ndarray:
            return np.frombuffer(data, dtype=np.dtype(self.item_type))
        return data


def unspool(value: Any) -> Any:
    """Return ``value``'s payload if it is a :py:class:`SpoolRef`, else ``value`` itself."""
    return value.read() if isinstance(value, SpoolRef) else value


class SpoolWriter:
    """Appends payloads to one episode's spool file; safe to share between fetch threads.

    Each :py:meth:`put` is flushed before it returns, so its reference can be
    read at once, by this process or another, while the writer stays open.

    Args:
        directory: An existing directory to create the file in.
        prefix: The start of the file's name; a unique suffix is added.
    """

    def __init__(self, directory: Path | str, prefix: str = "episode-") -> None:
        fd, self.path = tempfile.mkstemp(prefix=prefix, suffix=".spool", dir=directory)
        self._file = os.fdopen(fd, "wb")
        self._lock = threading.Lock()
        self._size = 0

    def put(self, value: Any) -> Any:
        """Write ``value`` to the file and return its :py:class:`SpoolRef`.

        ``bytes``, ``bytearray``, ``memoryview``, ``array.array`` and
        one-dimensional numeric or boolean ``numpy.ndarray`` values are
        written. Any other value, including ``None`` and NaN, is returned
        unchanged, so a missing payload stays missing.
        """
        if isinstance(value, (bytes, bytearray, memoryview)):
            container, item_type = bytes, ""
        elif isinstance(value, array.array):
            container, item_type = array.array, value.typecode
        elif isinstance(value, np.ndarray) and value.ndim == 1 and value.dtype.kind in "biufc":
            container, item_type = np.ndarray, value.dtype.str
        else:
            return value
        view = memoryview(value)
        if not view.c_contiguous:
            view = memoryview(view.tobytes())
        with self._lock:
            offset = self._size
            self._file.write(view)
            self._file.flush()
            self._size += view.nbytes
        return SpoolRef(self.path, offset, view.nbytes, container, item_type)

    def close(self) -> None:
        """Close the file. References stay readable until the file is deleted."""
        self._file.close()

    def discard(self) -> None:
        """Close and delete the file."""
        self.close()
        Path(self.path).unlink(missing_ok=True)
