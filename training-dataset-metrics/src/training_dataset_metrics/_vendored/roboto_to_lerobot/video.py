"""GOP-aware decoding of compressed-video camera topics (``foxglove_msgs/CompressedVideo``).

Compressed-video topics store one encoded *access unit* per message, not one
still image. A delta frame is meaningless without the frames since its
preceding keyframe, so the per-message decode contract the rest of the
converter is built on (``DECODERS[type](row, spec) -> HWC uint8 RGB``) cannot
apply: ``cv2.imdecode`` on an H.264 access unit returns nothing decodable.

This module resolves that by moving the decode one layer earlier. A whole
episode's time range is decoded once, in the fetch step, into
``{timestamp, frame}`` rows; the registered decoder for the message type is
then a pass-through that hands the already-decoded array to
``lerobot.generate_frames``. Everything downstream — merge onto the reference
timeline, resize, deferred IPC, the writers — is unchanged.

Decoding itself is delegated entirely to the SDK's GOP-aware decoder
(:py:mod:`roboto.experimental.video`, the ``roboto[video]`` extra). It walks
backward to the keyframe that anchors a range's leading delta frames, skips
frames it cannot decode, and covers H.264/H.265/VP9/AV1 — the four formats the
Foxglove ``CompressedVideo`` spec allows. Nothing here parses a bitstream.

Detection mirrors what Roboto ingestion registers, not a topic-name heuristic:
ingestion tags compressed-video topics with the codec-agnostic representation
format ``compressedVideo`` and declares the actual codec per message in the
message's ``format`` field (see ``ros_ingestion.message_paths``). Both signals
are read here — the representation format and the topic's schema name — with
the contract's declared ``type:`` as the fallback for stubs and older topics.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

import pandas as pd
from roboto.experimental.video import (
    DEFAULT_KEYFRAME_LOOKBACK_NS,
    VideoCodec,
    decode_frames_in_range,
    resolve_codec,
    supported_formats,
)

from .extract import _add_timestamp_column
from .logger import logger
from .runtime.image import resize_image

__all__ = (
    "COMPRESSED_VIDEO_MESSAGE_PATHS",
    "COMPRESSED_VIDEO_SCHEMAS",
    "decode_video_stream_rows",
    "is_compressed_video_schema",
    "is_compressed_video_topic",
)


# Compressed-video schema names, verbatim from
# ``ros_ingestion.message_paths._COMPRESSED_VIDEO_SCHEMAS``. These are also the
# ``type:`` spellings a contract may declare for a video stream, and each one is
# registered in the decoder registry (``runtime/decoders.py``) — a spelling in
# one and not the other would route a spec to the video-stream loader and then
# fail to find a decoder for it.
COMPRESSED_VIDEO_SCHEMAS: frozenset[str] = frozenset({
    "foxglove_msgs/CompressedVideo",
    "foxglove_msgs/msg/CompressedVideo",
    "foxglove.CompressedVideo",
})

# Leaf type names that are themselves a compressed-video stream, matched in
# full so sidecar types such as ``CompressedVideoInfo`` do not qualify. Mirrors
# ``ros_ingestion.message_paths._COMPRESSED_VIDEO_LEAF_RE``.
_COMPRESSED_VIDEO_LEAF_RE = re.compile(r"compressedvideo(packet)?", re.IGNORECASE)

# Representation ``format`` values that mark a topic as compressed video.
# ``compressedVideo`` is the codec-agnostic label ingestion registers
# (``ros_ingestion.message_paths.COMPRESSED_VIDEO_FORMAT``); per-codec labels
# are accepted too, matching the backend's ``is_video_representation``.
_VIDEO_REPRESENTATION_FORMATS: frozenset[str] = supported_formats() | {"compressedvideo"}

# The message paths fetched for a compressed-video topic. ``data`` is the Annex
# B access unit, ``format`` the per-message codec token ("h264", ...). The
# in-message ``timestamp``/``frame_id`` fields are deliberately not fetched:
# frames are keyed by log time (as every other stream in this converter is),
# and a fetched ``timestamp`` column would collide with the one
# ``_add_timestamp_column`` derives.
COMPRESSED_VIDEO_MESSAGE_PATHS: list[str] = ["format", "data"]


def is_compressed_video_schema(schema_name: str) -> bool:
    """True if ``schema_name`` names a compressed-video stream.

    Mirrors ``ros_ingestion.message_paths.is_compressed_video_schema``: the
    well-known Foxglove schemas exactly, plus any schema whose leaf type name is
    ``CompressedVideo`` or ``CompressedVideoPacket``. The leaf must match in
    full, so metadata types like ``CompressedVideoInfo`` — about video, but
    carrying no playable stream — do not qualify.

    Args:
        schema_name: A ROS/protobuf schema name, or a contract ``type:`` value.

    Returns:
        Whether the name denotes a stream of encoded video access units.
    """
    if schema_name in COMPRESSED_VIDEO_SCHEMAS:
        return True
    leaf = schema_name.rsplit("/", 1)[-1].rsplit(".", 1)[-1]
    return _COMPRESSED_VIDEO_LEAF_RE.fullmatch(leaf) is not None


def _representation_formats(topic: Any) -> Iterator[str]:
    """Yield the ``format`` string of every representation registered on ``topic``.

    Defensive by construction: topics come from the SDK in production and from
    hand-built stubs in tests, and older topics may carry no representation at
    all. Anything missing or non-string is skipped rather than raising, so a
    topic that simply does not advertise a format falls through to the schema
    name.
    """
    candidates = [getattr(topic, "default_representation", None)]
    for message_path in getattr(topic, "message_paths", None) or ():
        candidates.extend(getattr(message_path, "representations", None) or ())
    for candidate in candidates:
        fmt = getattr(candidate, "format", None)
        if isinstance(fmt, str) and fmt:
            yield fmt


def is_compressed_video_topic(topic: Any) -> bool:
    """Whether ``topic`` stores compressed video rather than per-message stills.

    Reads the two signals Roboto ingestion actually registers: the
    representation ``format`` (``compressedVideo``, or a per-codec label) and
    the topic's schema name. Used to cross-check a contract's declared
    ``type:`` — a topic that carries video but is declared as
    ``sensor_msgs/msg/CompressedImage`` would otherwise be handed one H.264
    access unit at a time to ``cv2.imdecode``.

    Args:
        topic: A ``roboto.Topic`` (or any object exposing the same attributes).

    Returns:
        ``True`` when either signal identifies the topic as compressed video.
        ``False`` when neither does — including when the topic advertises
        neither, which is why callers treat this as evidence, not as the
        routing decision.
    """
    for fmt in _representation_formats(topic):
        if fmt.lower() in _VIDEO_REPRESENTATION_FORMATS:
            return True
    schema_name = getattr(topic, "schema_name", None)
    return isinstance(schema_name, str) and is_compressed_video_schema(schema_name)


def _resolve_stream_codec(first_format: Any, topic_name: str) -> VideoCodec:
    """Resolve the codec of a compressed-video stream from a message's ``format`` token.

    Mirrors the backend's ``resolve_video_stream_codec``: the codec is declared
    per message, and a message without a ``format`` token is treated as H.264
    (ingestion's legacy behaviour).

    Raises:
        ValueError: If the token names a codec with no decode path. An
            exporter must fail here rather than write an episode with missing
            camera frames.
    """
    if not isinstance(first_format, str) or not first_format:
        return resolve_codec("h264")  # type: ignore[return-value]  # always registered
    codec = resolve_codec(first_format)
    if codec is None:
        raise ValueError(
            f"Topic '{topic_name}' carries compressed video in unsupported codec "
            f"'{first_format}'. Decodable codecs: {', '.join(sorted(supported_formats()))}."
        )
    return codec




def _fetch_messages(topic: Any, range_start: int, range_end: int) -> list[tuple[int, Any, bytes]]:
    """Read one Topic's compressed-video messages for a range as ``(log_time, format, data)``.

    Frames are keyed by log time, derived the same way every other stream in
    this converter derives it (``_add_timestamp_column`` over what
    ``get_data_as_df`` returns), so a video frame and a joint-state sample that
    share a log time align on the reference timeline.
    """
    raw = topic.get_data_as_df(
        start_time=pd.Timestamp(range_start, unit="ns"),
        end_time=pd.Timestamp(range_end, unit="ns"),
        message_paths_include=COMPRESSED_VIDEO_MESSAGE_PATHS,
    )
    if raw is None or len(raw) == 0:
        return []
    df = _add_timestamp_column(raw)
    return [
        (int(log_time), token, bytes(data))
        for log_time, token, data in zip(
            df["timestamp"], df["format"], df["data"], strict=True
        )
        if data is not None
    ]


def decode_video_stream_rows(
    topic: Any,
    start_ns: int,
    end_ns: int,
    *,
    topic_name: str,
    resize: tuple[int, int] | None = None,
    keyframe_lookback_ns: int = DEFAULT_KEYFRAME_LOOKBACK_NS,
) -> list[dict]:
    """Decode a compressed-video topic's ``[start_ns, end_ns]`` range to frame rows.

    One row per decodable frame, shaped so ``DataCollection._load_videos`` can
    concat and timestamp-sort it exactly like the other video loaders' output::

        {"timestamp": <log time, int ns>, "frame": <HWC uint8 RGB ndarray>}

    An episode's range normally starts mid-GOP, so the SDK decoder walks
    backward to the keyframe that anchors the leading delta frames (bounded by
    ``keyframe_lookback_ns``) and decodes that prefix without emitting it. That
    backward walk is why compressed video cannot be decoded one message at a
    time, and why the decode happens here — in the fetch step, over a whole
    range — rather than in a registered per-row decoder.

    The requested range is fetched once and reused; only the keyframe lookback,
    when needed, costs a second fetch.

    ``resize`` is applied here rather than left to ``generate_frames`` purely to
    bound memory: an episode's frames are all held in RAM until the episode is
    written, and a contract asking for 224x224 has no use for 640x480 arrays.
    It calls the same ``resize_image`` ``generate_frames`` would call later, and
    that function no-ops on a shape match, so resizing early is
    indistinguishable from resizing late.

    Args:
        topic: The ``roboto.Topic`` to read, via ``get_data_as_df``.
        start_ns: Start of the range, nanoseconds since epoch (inclusive).
        end_ns: End of the range, nanoseconds since epoch.
        topic_name: Topic name, used only in error and log messages.
        resize: ``(height, width)`` to resize each frame to, or ``None`` to keep
            the stream's native resolution.
        keyframe_lookback_ns: How far before ``start_ns`` to search for the
            keyframe that anchors the range's leading delta frames.

    Returns:
        The decoded rows in ascending log-time order; empty when the topic has
        no messages in the range.

    Raises:
        ValueError: If the stream declares a codec with no decode path, or if it
            has messages in range but none of them decoded. Both would otherwise
            surface as an episode silently missing its camera.
        ImportError: If PyAV is not installed (the ``roboto[video]`` extra).
    """
    in_range = _fetch_messages(topic, start_ns, end_ns)
    if not in_range:
        return []
    codec = _resolve_stream_codec(in_range[0][1], topic_name)

    def load_messages(range_start: int, range_end: int) -> Iterator[tuple[int, bytes]]:
        messages = (
            in_range
            if (range_start, range_end) == (start_ns, end_ns)
            else _fetch_messages(topic, range_start, range_end)
        )
        for log_time, token, data in messages:
            if isinstance(token, str) and token.lower() not in codec.formats:
                logger.warning(
                    "Topic %s: skipping message at %d declaring codec '%s' in a '%s' stream",
                    topic_name, log_time, token, codec.name,
                )
                continue
            yield log_time, data

    rows: list[dict] = []
    for decoded in decode_frames_in_range(
        load_messages, start_ns, end_ns, keyframe_lookback_ns=keyframe_lookback_ns, codec=codec
    ):
        frame = decoded.to_ndarray()
        if resize is not None:
            frame = resize_image(frame, resize[0], resize[1])
        rows.append({"timestamp": decoded.log_time, "frame": frame})

    if not rows:
        raise ValueError(
            f"Topic '{topic_name}' has {len(in_range)} compressed-video message(s) in "
            f"[{start_ns}, {end_ns}] but none decoded. Either no keyframe lies within the "
            f"{keyframe_lookback_ns} ns lookback before the range, or the stream's keyframes "
            f"are missing the in-band parameter sets a decoder needs to initialize."
        )
    if len(rows) < len(in_range):
        logger.warning(
            "Topic %s: decoded %d of %d compressed-video messages in range; the rest had no "
            "reachable keyframe or were corrupt.",
            topic_name, len(rows), len(in_range),
        )
    return rows
