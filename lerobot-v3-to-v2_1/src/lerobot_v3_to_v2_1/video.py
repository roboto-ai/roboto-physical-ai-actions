"""Frame-exact per-episode video splitting.

The vendored ``convert_videos`` splits a v3 concatenated video into per-episode
MP4s with a plain ``ffmpeg -ss/-t -c copy``. That is lossless but not
frame-exact: a timestamp-duration cut over-includes the trailing boundary frame,
and ``-ss`` on a non-keyframe boundary snaps to the previous keyframe. This
module guarantees each per-episode MP4 decodes to exactly the episode's frame
count:

* if the episode starts on a keyframe, stream-copy exactly ``length`` frames
  (lossless, no re-encode);
* otherwise re-encode exactly the episode's frame range (quality-preserving
  fallback — only fires when a boundary is not keyframe-aligned).

A decoded-frame-count check after each split fails loudly if either path is
wrong, so a silent off-by-one can never reach the output.
"""

from __future__ import annotations

import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import av

from .convert.convert_dataset_v30_to_v21 import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_VIDEO_PATH,
    LEGACY_VIDEO_PATH_TEMPLATE,
)
from .logger import logger

# Match an episode boundary timestamp to a source keyframe within half a frame.
_KEYFRAME_TOL_FRAMES = 0.5

# Per-codec re-encode args for the (rare) non-keyframe-aligned fallback. -crf 1
# is visually lossless; the source is already lossy so this adds no perceptible
# loss while guaranteeing the exact frame range.
_REENCODE_ARGS: dict[str, list[str]] = {
    "h264": ["-c:v", "libx264", "-crf", "1", "-pix_fmt", "yuv420p"],
    "hevc": ["-c:v", "libx265", "-crf", "1", "-pix_fmt", "yuv420p"],
    "av1": ["-c:v", "libsvtav1", "-crf", "1", "-pix_fmt", "yuv420p"],
}
_DEFAULT_REENCODE_ARGS = ["-c:v", "libx264", "-crf", "1", "-pix_fmt", "yuv420p"]


def _run_ffmpeg(cmd: list[str]) -> None:
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ffmpeg failed: {' '.join(cmd)}\n{exc.stderr}"
        ) from exc
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg executable not found on PATH; it is required for video splitting"
        ) from exc


def _source_codec(path: Path) -> str:
    with av.open(str(path)) as container:
        return container.streams.video[0].codec_context.name


def _keyframe_times(path: Path) -> list[float]:
    """Presentation times (seconds) of keyframes, via demux (no full decode)."""
    times: list[float] = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for packet in container.demux(stream):
            if packet.is_keyframe and packet.pts is not None:
                times.append(float(packet.pts * stream.time_base))
    return sorted(times)


def _decoded_frame_count(path: Path) -> int:
    with av.open(str(path)) as container:
        return sum(1 for _ in container.decode(container.streams.video[0]))


def _matching_keyframe(from_ts: float, keyframe_times: list[float], fps: int) -> float | None:
    """Return the exact keyframe pts matching ``from_ts`` (within tolerance), else None."""
    tolerance = _KEYFRAME_TOL_FRAMES / fps
    best: float | None = None
    for kt in keyframe_times:
        if abs(kt - from_ts) <= tolerance and (
            best is None or abs(kt - from_ts) < abs(best - from_ts)
        ):
            best = kt
    return best


def _stream_copy_segment(src: Path, dst: Path, start_ts: float, num_frames: int) -> None:
    _run_ffmpeg([
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start_ts:.6f}", "-i", str(src),
        "-frames:v", str(num_frames),
        "-c", "copy", "-avoid_negative_ts", "1", "-y", str(dst),
    ])


def _reencode_segment(
    src: Path, dst: Path, start_frame: int, num_frames: int, codec: str
) -> None:
    encode_args = _REENCODE_ARGS.get(codec, _DEFAULT_REENCODE_ARGS)
    end_frame = start_frame + num_frames  # trim end_frame is exclusive
    vf = f"trim=start_frame={start_frame}:end_frame={end_frame},setpts=PTS-STARTPTS"
    _run_ffmpeg([
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(src), "-vf", vf, "-frames:v", str(num_frames),
        "-an", *encode_args, "-y", str(dst),
    ])


def convert_videos_frame_exact(
    source_root: Path,
    dest_root: Path,
    episode_records: list[dict[str, Any]],
    video_keys: list[str],
    fps: int,
) -> dict[str, int]:
    """Split each v3 concatenated video into frame-exact per-episode v2.1 MP4s.

    Returns a report ``{"copied": n, "reencoded": n}`` counting how many episode
    segments were lossless stream-copied vs re-encoded.
    """
    report = {"copied": 0, "reencoded": 0}
    if not video_keys:
        return report

    for video_key in video_keys:
        chunk_col = f"videos/{video_key}/chunk_index"
        file_col = f"videos/{video_key}/file_index"
        from_col = f"videos/{video_key}/from_timestamp"

        grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for record in episode_records:
            if record.get(chunk_col) is None or record.get(file_col) is None:
                continue
            grouped[(int(record[chunk_col]), int(record[file_col]))].append(record)

        for (chunk_idx, file_idx), records in grouped.items():
            src = source_root / DEFAULT_VIDEO_PATH.format(
                video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
            )
            if not src.exists():
                raise FileNotFoundError(f"Expected MP4 file not found: {src}")

            codec = _source_codec(src)
            keyframe_times = _keyframe_times(src)
            records = sorted(records, key=lambda r: float(r[from_col]))

            for record in records:
                episode_index = int(record["episode_index"])
                from_ts = float(record[from_col])
                length = int(record["length"])
                start_frame = round(from_ts * fps)

                dst = dest_root / LEGACY_VIDEO_PATH_TEMPLATE.format(
                    episode_chunk=episode_index // DEFAULT_CHUNK_SIZE,
                    video_key=video_key,
                    episode_index=episode_index,
                )
                dst.parent.mkdir(parents=True, exist_ok=True)

                keyframe_ts = _matching_keyframe(from_ts, keyframe_times, fps)
                reencoded = False
                if keyframe_ts is not None:
                    _stream_copy_segment(src, dst, keyframe_ts, length)
                    if _decoded_frame_count(dst) != length:
                        # B-frame tail or container quirk made the copy wrong;
                        # fall back to an exact re-encode.
                        _reencode_segment(src, dst, start_frame, length, codec)
                        reencoded = True
                else:
                    _reencode_segment(src, dst, start_frame, length, codec)
                    reencoded = True

                produced = _decoded_frame_count(dst)
                if produced != length:
                    raise RuntimeError(
                        f"Frame-exact split failed for {video_key} episode "
                        f"{episode_index}: produced {produced} frames, expected {length}"
                    )
                report["reencoded" if reencoded else "copied"] += 1

    logger.info(
        "Video split complete: %d stream-copied, %d re-encoded",
        report["copied"],
        report["reencoded"],
    )
    return report
