"""Fd-level filter that drops SVT-AV1 banner/info output from stderr.

The SVT-AV1 library emits a multi-line banner ("SVT [version]: ...",
"SVT [config]: ...", "Svt[info]: ...", "-----...") directly to fd 2 via
``fprintf``. It bypasses FFmpeg's ``av_log``, PyAV's ``libav`` logger, and
Python's logging entirely, so no level-based filter on the Python side can
suppress it. On a multi-shard run those banners repeat per encoder per
episode and bury the real log output.

This module remaps fd 2 to the write end of a pipe and runs a daemon thread
that reads from the read end, drops SVT noise lines, and forwards everything
else back to the original fd 2. Spawned subprocesses inherit the redirected
fd 2 automatically, so a single install in the parent covers shard
subprocesses and per-event worker subprocesses too.

Real-failure SVT messages ("SVT [error]" / "Svt[error]") still surface so
a broken encode is not silently dropped. Two specific ``Svt[warn]`` lines
are filtered as known-benign — see ``_SVT_BENIGN_WARN``.
"""

from __future__ import annotations

import os
import re
import sys
import threading

# Patterns covering the SVT-AV1 banner / configuration output. The two
# prefix variants ("SVT [tag]:" and "Svt[tag]:") cover both legacy and
# current SVT-AV1 formatters; the "SVT-AV1 Encoder Lib" header and the
# decorative "-----..." separator complete the banner.
_SVT_NOISE = re.compile(
    rb"^("
    rb"\s*SVT \[(version|build|info|config)\]"
    rb"|\s*Svt\[(info|config)\]"
    rb"|\s*SVT-AV1 Encoder Lib"
    rb"|-{4,}\s*$"
    rb")"
)

# Two SVT-AV1 warnings that the installed encoder emits every run despite
# the encode completing correctly:
#   * "Preset M<N> is mapped to M<M>" — lerobot's _get_codec_options defaults
#     preset to "12" for libsvtav1; the installed SVT build's MAX_ENC_PRESET
#     is 10, so it clamps and warns. Output is identical to passing M10
#     directly.
#   * "Failed to set thread priority: Invalid argument" — SVT requests a
#     non-default thread scheduling priority; the container runs without
#     CAP_SYS_NICE so the syscall returns EINVAL. SVT falls back to default
#     scheduling and continues.
# Other Svt[warn] / Svt[error] lines still pass through.
_SVT_BENIGN_WARN = re.compile(
    rb"^\s*Svt\[warn\]:\s*("
    rb"Preset M\d+ is mapped to M\d+"
    rb"|Failed to set thread priority"
    rb")"
)

_INSTALLED = False


def install_svt_stderr_filter() -> None:
    """Reroute fd 2 through a pipe + SVT-noise filter thread.

    Idempotent. Must be called before ``LeRobotWriter.create`` (i.e. before
    any libsvtav1 encoder starts) so the first SVT banner write travels
    through the filter rather than landing on the raw fd 2.
    """
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    # Flush Python-buffered output now so nothing written before the redirect
    # gets re-ordered behind the post-redirect pipe traffic.
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass

    read_fd, write_fd = os.pipe()
    saved_stderr_fd = os.dup(2)
    os.dup2(write_fd, 2)
    os.close(write_fd)

    def pump() -> None:
        try:
            reader = os.fdopen(read_fd, "rb", buffering=0)
            writer = os.fdopen(saved_stderr_fd, "wb", buffering=0)
        except Exception:
            return
        with reader, writer:
            buf = b""
            while True:
                try:
                    chunk = reader.read(4096)
                except Exception:
                    return
                if not chunk:
                    return
                buf += chunk
                while True:
                    nl = buf.find(b"\n")
                    if nl < 0:
                        break
                    line, buf = buf[: nl + 1], buf[nl + 1 :]
                    if _SVT_NOISE.match(line) or _SVT_BENIGN_WARN.match(line):
                        continue
                    try:
                        writer.write(line)
                    except Exception:
                        return

    threading.Thread(
        target=pump, name="svt-stderr-filter", daemon=True,
    ).start()


def set_libav_log_level() -> None:
    """Pin libav's C-level log threshold to WARNING.

    PyAV 15.x exposes two distinct knobs and only one of them matters for
    suppressing terminal output:

    * ``av.logging.set_level(...)`` — gates PyAV's *Python-side* filter
      (what its callback routes into Python's ``logging.getLogger("libav")``).
      Has no effect on what libav's default C callback writes to stderr.

    * ``av.logging.set_libav_level(...)`` — gates libav's C-level threshold
      via ``av_log_set_level``. This is the only switch that hides INFO
      messages from libav's default callback, which lerobot re-installs
      after each encode via ``av.logging.restore_default_callback()``.

    Without this, ``[mp4 @ 0x...] Starting second pass: moving the moov
    atom...`` fires once per video stream finalization (so K times per
    episode × N episodes × per-shard) and drowns the per-episode log
    lines.

    Process-local state: shard subprocesses spawned via
    ``mp_context="spawn"`` re-import pyav fresh and inherit the default
    INFO threshold, so this function must be called inside each shard's
    entrypoint as well.
    """
    try:
        import av.logging  # type: ignore
    except Exception:
        return
    # ``set_libav_level`` is the libav C-level filter; ``set_level`` only
    # affects PyAV's Python-side routing and would leave the mp4 INFO
    # lines on stderr.
    av.logging.set_libav_level(av.logging.WARNING)


def disable_progress_bars() -> None:
    """Silence tqdm and HuggingFace datasets progress bars.

    lerobot's ``aggregate_datasets`` / ``dataset_tools`` modules wrap their
    work loops in ``tqdm.tqdm(...)`` and the embed step calls
    ``dataset.map(...)``, which both emit ``Map: 100%|...|`` style bars to
    stderr. The bars are interleaved across shards and obscure the per-
    episode log lines.

    Disabling is idempotent and applies process-wide. Like
    :func:`set_libav_log_level`, the underlying state is process-local and
    must be re-applied inside each spawned shard subprocess.
    """
    # HF datasets — single runtime call covers ``.map`` progress and the
    # rest of the datasets library.
    try:
        import datasets  # type: ignore

        datasets.disable_progress_bar()
    except Exception:
        pass

    # tqdm — monkey-patch the class so every subsequent ``tqdm.tqdm(...)``
    # constructor (including ``from tqdm import tqdm``-style imports lerobot
    # uses) inherits ``disable=True``. Patching the class object covers all
    # already-imported aliases because they share the same class.
    try:
        from functools import partialmethod

        import tqdm  # type: ignore

        tqdm.tqdm.__init__ = partialmethod(tqdm.tqdm.__init__, disable=True)
        # ``tqdm.auto`` re-exports the same class; patch defensively in
        # case lerobot or a transitive dep used the auto-resolver.
        try:
            from tqdm.auto import tqdm as tqdm_auto  # type: ignore

            tqdm_auto.__init__ = partialmethod(tqdm_auto.__init__, disable=True)
        except Exception:
            pass
    except Exception:
        pass
