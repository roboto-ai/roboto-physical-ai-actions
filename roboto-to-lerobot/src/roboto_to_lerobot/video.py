"""GOP-aware decoding of compressed-video camera topics (``foxglove_msgs/CompressedVideo``).

Compressed-video topics store one encoded *access unit* per message, not one
still image. A delta frame is meaningless without the frames since its
preceding keyframe, so the per-message decode contract the rest of the
converter is built on (``DECODERS[type](row, spec) -> HWC uint8 RGB``) cannot
apply: ``cv2.imdecode`` on an H.264 access unit returns nothing decodable.

This module resolves that by decoding each episode's range twice. The fetch
step, which has the whole range in hand, runs a *probe*: it feeds the range's
packets through the decoder to learn exactly which frames come out, converts
none of them to pixels, and keeps the packets it read in a picklable
:py:class:`VideoSession`. Each frame becomes a :py:class:`VideoFrameRef` row
(its session, its position in the decoder's output, and its log time), so
alignment, the transfer from worker to main process, and the drain carry a
few bytes per frame instead of a decoded array. Just before the writer takes
a frame, a :py:class:`VideoFrameReader` decodes the session's packets again
and converts only the frames the episode uses. The decoded frames held at
any moment are then the few about to be written, however long the episode.

Decoding itself is delegated entirely to the SDK's GOP-aware decoder
(:py:mod:`roboto.experimental.video`, the ``roboto[video]`` extra). It walks
backward to the keyframe that anchors a range's leading delta frames, skips
frames it cannot decode, and covers H.264/H.265/VP9/AV1 — the four formats the
Foxglove ``CompressedVideo`` spec allows. Nothing here parses a bitstream.

Detection reads what Roboto ingestion registers rather than guessing from
topic names. Ingestion tags a compressed-video topic with the codec-agnostic
representation format ``compressedVideo`` and declares each message's codec in
its ``format`` field. A topic counts as compressed video when its
representation format or its schema name says so; the per-message ``format``
then selects the decoder. A contract's declared ``type:`` decides how a spec is
loaded, and these topic signals catch a video topic declared as stills.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from roboto.experimental.video import (
    DEFAULT_KEYFRAME_LOOKBACK_NS,
    VideoCodec,
    decode_frames_in_range,
    resolve_codec,
    supported_formats,
)

from .extract import _add_timestamp_column, select_message_field
from .logger import logger

__all__ = (
    "COMPRESSED_VIDEO_MESSAGE_PATHS",
    "COMPRESSED_VIDEO_SCHEMAS",
    "VideoFrameReader",
    "VideoFrameRef",
    "VideoSession",
    "is_compressed_video_schema",
    "is_compressed_video_topic",
    "probe_video_stream_rows",
)


# Compressed-video schema names, as Roboto's ROS ingestion recognizes them.
# These are also the ``type:`` spellings a contract may declare for a video
# stream, and each one is registered in the decoder registry
# (``runtime/decoders.py``). A spelling in one and not the other would route a
# spec to the video-stream loader, which would then find no decoder for it.
COMPRESSED_VIDEO_SCHEMAS: frozenset[str] = frozenset({
    "foxglove_msgs/CompressedVideo",
    "foxglove_msgs/msg/CompressedVideo",
    "foxglove.CompressedVideo",
})

# Leaf type names that are themselves a compressed-video stream, matched in
# full so sidecar types such as ``CompressedVideoInfo`` do not qualify. Roboto's
# ROS ingestion applies the same rule.
_COMPRESSED_VIDEO_LEAF_RE = re.compile(r"compressedvideo(packet)?", re.IGNORECASE)

# Representation ``format`` values that mark a topic as compressed video.
# ``compressedVideo`` is the codec-agnostic label ingestion registers;
# per-codec labels are accepted too.
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

    Applies the same rule as Roboto's ROS ingestion: the Foxglove schemas in
    ``COMPRESSED_VIDEO_SCHEMAS`` exactly, plus any schema whose leaf type name
    is ``CompressedVideo`` or ``CompressedVideoPacket``. The leaf must match in
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

    The codec is declared per message, and a message without a ``format`` token
    is treated as H.264, as Roboto ingestion treats it.

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




def _fetch_messages(
    topic: Any, range_start: int, range_end: int, topic_name: str, field: str | None,
) -> list[tuple[int, Any, bytes]]:
    """Read one Topic's compressed-video messages for a range as ``(log_time, format, data)``.

    Frames are keyed by log time, derived the same way every other stream in
    this converter derives it (``_add_timestamp_column`` over what
    ``get_data_as_df`` returns), so a video frame and a joint-state sample that
    share a log time align on the reference timeline. With ``field`` set, each
    message carries its video message at that message path, and ``format`` and
    ``data`` are read from there.
    """
    paths = COMPRESSED_VIDEO_MESSAGE_PATHS
    if field is not None:
        paths = [f"{field}.{path}" for path in paths]
    raw = topic.get_data_as_df(
        start_time=pd.Timestamp(range_start, unit="ns"),
        end_time=pd.Timestamp(range_end, unit="ns"),
        message_paths_include=paths,
    )
    if raw is None or len(raw) == 0:
        return []
    df = _add_timestamp_column(raw)
    if field is not None:
        df = select_message_field(df, field, topic_name)
    return [
        (int(log_time), token, bytes(data))
        for log_time, token, data in zip(
            df["timestamp"], df["format"], df["data"], strict=True
        )
        if data is not None
    ]


@dataclass(eq=False)
class VideoSession:
    """The packets read while probing one range of a compressed-video topic, kept to decode them again.

    ``decode_frames_in_range`` reads a range's messages once and, when the
    range opens mid-GOP, once more for the GOP prefix before it. ``loads``
    holds what each of those reads returned, after the per-message codec
    filter, keyed by the ``(start, end)`` the decoder asked for. Calling
    ``decode_frames_in_range`` again with the same arguments over these reads
    feeds a fresh decoder the same packets in the same order, so it emits the
    same frames. A worker therefore ships an episode's compressed packets,
    a few MB per camera-minute, instead of its decoded frames, about 1.7 GB
    per minute of 640x480 at 30 fps.

    Compared and hashed by identity, so a reader can key its decodes by
    session: one probe makes one session, every row it returns refers to it,
    and pickling an episode pickles it once.
    """

    topic_name: str
    codec_name: str
    start_ns: int
    end_ns: int
    keyframe_lookback_ns: int
    loads: dict[tuple[int, int], list[tuple[int, bytes]]]

    def decode(self) -> Iterator[Any]:
        """Decode the recorded packets again, yielding ``DecodedVideoFrame`` s in the probe's order."""
        return decode_frames_in_range(
            lambda range_start, range_end: self.loads[(range_start, range_end)],
            self.start_ns,
            self.end_ns,
            keyframe_lookback_ns=self.keyframe_lookback_ns,
            codec=resolve_codec(self.codec_name),
        )


@dataclass(frozen=True, slots=True)
class VideoFrameRef:
    """One frame of a compressed-video topic, by its place in a session's decode.

    ``ordinal`` counts the frames the session's decode emits, from 0. Frames
    are addressed by ordinal rather than by log time because a stream with
    B-frames emits frames out of log-time order (log times follow the
    packets, which arrive in decode order). ``log_time`` lets the reader
    check that its decode emitted the frame the probe saw at that position.
    """

    session: VideoSession
    ordinal: int
    log_time: int


# The most decoded frames a ``VideoFrameReader`` keeps for later requests, per
# session. An episode requests frames in log-time order, which is decode order,
# and a decoder emits them in display order. To serve a frame, the reader holds
# it and every frame that the decoder emits before it but decodes after it. A
# P-frame decoded ahead of a run of N B-frames is displayed after them, so it
# needs N + 1 held. The 16-frame limits in H.264 and H.265 do not bound N:
# they cap the opposite case, frames decoded before a frame but displayed after
# it. H.264 sets no limit on N, and H.265 limits it only in a stream that
# declares a maximum latency, which may exceed 16. x264 and x265 encode at
# most 16 consecutive B-frames, which needs 17 held; 32 leaves room for
# encoders that allow longer runs. Past the limit the reader fails rather than
# hold a growing share of the episode decoded.
MAX_HELD_FRAMES = 32


class VideoFrameReader:
    """Turns an episode's ``VideoFrameRef`` s into pixels, decoding each session once, in order.

    Built from every reference the episode will request, so it knows how
    often each frame is used: it converts only those frames to arrays, keeps
    each one until its last request, and passes over the rest of the decoder's
    output unconverted. Requests must follow the decode order, give or take
    ``MAX_HELD_FRAMES`` of reordering; the reference timeline's log-time order
    does.

    Args:
        refs: Every reference the episode will request, once per request.
    """

    def __init__(self, refs: Iterable[VideoFrameRef]) -> None:
        self._replays: dict[VideoSession, _SessionReplay] = {}
        for ref in refs:
            replay = self._replays.get(ref.session)
            if replay is None:
                replay = self._replays[ref.session] = _SessionReplay(ref.session)
            replay.expect(ref)

    def frame(self, ref: VideoFrameRef) -> np.ndarray:
        """Return ``ref``'s frame as an HWC uint8 RGB array, at the stream's resolution.

        Raises:
            ValueError: If ``ref`` was not among the references the reader was
                built from, or is requested more often than it appeared there.
            RuntimeError: If the decode no longer emits the frame the probe
                recorded at ``ref``'s position, or the request order would
                hold more than ``MAX_HELD_FRAMES`` frames.
        """
        replay = self._replays.get(ref.session)
        if replay is None:
            raise ValueError(
                f"Topic '{ref.session.topic_name}': frame {ref.ordinal} belongs to a video session "
                "this reader was not built with."
            )
        return replay.take(ref)


class _SessionReplay:
    """One session's decode, consumed in order by a ``VideoFrameReader``."""

    def __init__(self, session: VideoSession) -> None:
        self._session = session
        self._frames = session.decode()  # a generator: nothing decodes until the first request
        self._emitted = 0
        self._uses: Counter[int] = Counter()
        self._log_times: dict[int, int] = {}
        self._held: dict[int, np.ndarray] = {}

    def expect(self, ref: VideoFrameRef) -> None:
        self._uses[ref.ordinal] += 1
        self._log_times[ref.ordinal] = ref.log_time

    def take(self, ref: VideoFrameRef) -> np.ndarray:
        topic_name = self._session.topic_name
        if self._uses[ref.ordinal] == 0:
            raise ValueError(
                f"Topic '{topic_name}': frame {ref.ordinal} was requested more often than the "
                "episode refers to it."
            )
        while ref.ordinal not in self._held:
            decoded = next(self._frames, None)
            if decoded is None:
                raise RuntimeError(
                    f"Topic '{topic_name}': decoding the episode's packets again emitted "
                    f"{self._emitted} frames, but the probe emitted frame {ref.ordinal}."
                )
            ordinal = self._emitted
            self._emitted += 1
            if ordinal not in self._uses:
                continue
            if decoded.log_time != self._log_times[ordinal]:
                raise RuntimeError(
                    f"Topic '{topic_name}': decoding the episode's packets again emitted frame "
                    f"{ordinal} at log time {decoded.log_time}, but the probe emitted it at "
                    f"{self._log_times[ordinal]}."
                )
            if len(self._held) >= MAX_HELD_FRAMES:
                raise RuntimeError(
                    f"Topic '{topic_name}': reaching frame {ref.ordinal} would hold more than "
                    f"{MAX_HELD_FRAMES} decoded frames; the stream reorders more frames than the reader "
                    "keeps, or frames were requested out of log-time order."
                )
            self._held[ordinal] = decoded.to_ndarray()
        image = self._held[ref.ordinal]
        self._uses[ref.ordinal] -= 1
        if self._uses[ref.ordinal] == 0:
            del self._uses[ref.ordinal], self._held[ref.ordinal]
            if not self._uses:
                self._frames = iter(())  # every frame served; let the decoder go
        return image


def probe_video_stream_rows(
    topic: Any,
    start_ns: int,
    end_ns: int,
    *,
    topic_name: str,
    field: str | None = None,
    keyframe_lookback_ns: int = DEFAULT_KEYFRAME_LOOKBACK_NS,
) -> list[dict]:
    """Find the frames a compressed-video topic's ``[start_ns, end_ns]`` range decodes to.

    One row per decodable frame, shaped so ``DataCollection._load_videos`` can
    concat and timestamp-sort it like the other video loaders' output::

        {"timestamp": <log time, int ns>, "frame": <VideoFrameRef>}

    The range's packets go through the decoder here, so which frames decode
    is known exactly and a range that decodes to nothing fails here, where
    the converter records a skipped event. No frame is converted to pixels:
    each row refers to its frame by position in the decode, and the rows
    share one :py:class:`VideoSession` holding the packets, which a
    :py:class:`VideoFrameReader` decodes again where the frames are written.

    An episode's range normally starts mid-GOP, so the SDK decoder walks
    backward to the keyframe that anchors the leading delta frames (bounded by
    ``keyframe_lookback_ns``) and decodes that prefix without emitting it. That
    backward walk is why compressed video cannot be decoded one message at a
    time.

    The requested range is fetched once and reused; only the keyframe lookback,
    when needed, costs a second fetch.

    Args:
        topic: The ``roboto.Topic`` to read, via ``get_data_as_df``.
        start_ns: Start of the range, nanoseconds since epoch (inclusive).
        end_ns: End of the range, nanoseconds since epoch.
        topic_name: Topic name, used in error and log messages.
        field: Message path of the video message nested in each message of a
            custom type, or ``None`` when the topic's messages are the video
            messages themselves.
        keyframe_lookback_ns: How far before ``start_ns`` to search for the
            keyframe that anchors the range's leading delta frames.

    Returns:
        The rows in the order the decoder emits their frames, which is what
        ``VideoFrameRef.ordinal`` counts; empty when the topic has no
        messages in the range.

    Raises:
        ValueError: If the stream declares a codec with no decode path, or if it
            has messages in range but none of them decoded. Both would otherwise
            surface as an episode silently missing its camera.
        ImportError: If PyAV is not installed (the ``roboto[video]`` extra).
    """
    in_range = _fetch_messages(topic, start_ns, end_ns, topic_name, field)
    if not in_range:
        return []
    codec = _resolve_stream_codec(in_range[0][1], topic_name)
    loads: dict[tuple[int, int], list[tuple[int, bytes]]] = {}

    def load_messages(range_start: int, range_end: int) -> list[tuple[int, bytes]]:
        messages = (
            in_range
            if (range_start, range_end) == (start_ns, end_ns)
            else _fetch_messages(topic, range_start, range_end, topic_name, field)
        )
        kept: list[tuple[int, bytes]] = []
        for log_time, token, data in messages:
            if isinstance(token, str) and token.lower() not in codec.formats:
                logger.warning(
                    "Topic %s: skipping message at %d declaring codec '%s' in a '%s' stream",
                    topic_name, log_time, token, codec.name,
                )
                continue
            kept.append((log_time, data))
        loads[(range_start, range_end)] = kept
        return kept

    session = VideoSession(
        topic_name=topic_name,
        codec_name=codec.name,
        start_ns=start_ns,
        end_ns=end_ns,
        keyframe_lookback_ns=keyframe_lookback_ns,
        loads=loads,
    )
    rows = [
        {"timestamp": decoded.log_time, "frame": VideoFrameRef(session, ordinal, decoded.log_time)}
        for ordinal, decoded in enumerate(
            decode_frames_in_range(
                load_messages, start_ns, end_ns, keyframe_lookback_ns=keyframe_lookback_ns, codec=codec
            )
        )
    ]

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
