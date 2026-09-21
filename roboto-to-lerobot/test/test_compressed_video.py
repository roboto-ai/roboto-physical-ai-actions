"""Tests for compressed-video (``foxglove_msgs/CompressedVideo``) camera topics.

Compressed video is the one camera representation that cannot be decoded one
message at a time: each message is a single H.264 access unit, and a delta
frame only decodes after the keyframe that opens its GOP. These tests exercise
that decode for real — real H.264 bitstreams produced by PyAV's libx264, fed
through the real SDK decoder — because a fake around the decoder would test
nothing about the property that makes this hard.

The fixtures are encoded in-process rather than committed: ``test/fixtures/``
is gitignored in this repo, and an in-process encode lets a test dial GOP
size, keyframe placement, and parameter-set presence, which is exactly what
the interesting cases vary. A separate test runs against a real ingested
REASSEMBLE recording when one is present on the machine.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import av
import numpy as np
import pandas as pd
import pytest
from roboto.experimental.video.h264 import NalUnitType, find_nal_units, is_keyframe
from roboto_to_lerobot.contract_utils import (
    AlignSpec,
    Contract,
    DataCollection,
    ObservationSpec,
    video_spec_kind,
)
from roboto_to_lerobot.lerobot import generate_frames
from roboto_to_lerobot.runtime.converters import DECODERS
from roboto_to_lerobot.video import (
    decode_video_stream_rows,
    is_compressed_video_schema,
    is_compressed_video_topic,
)

COMPRESSED_VIDEO = "foxglove_msgs/msg/CompressedVideo"

FRAME_INTERVAL_NS = 33_000_000  # ~30 fps
FIRST_LOG_TIME_NS = 1_736_427_449_000_000_000  # arbitrary but realistic epoch ns

# A real ingested REASSEMBLE recording, converted to CompressedVideo MCAPs.
# Machine-specific, so it is named by environment rather than hardcoded: point
# ROBOTO_REASSEMBLE_MCAP at such a recording to run the two tests below. Unset
# (as in CI) they skip rather than fail, so the suite stays hermetic.
REASSEMBLE_MCAP = os.environ.get("ROBOTO_REASSEMBLE_MCAP")
REASSEMBLE_TOPIC = "/camera/hand/video"

requires_reassemble_recording = pytest.mark.skipif(
    not (REASSEMBLE_MCAP and os.path.exists(REASSEMBLE_MCAP)),
    reason="set ROBOTO_REASSEMBLE_MCAP to a real CompressedVideo recording to run this",
)


# ── fixtures ────────────────────────────────────────────────────────────────


def _encode_h264_access_units(
    frame_count: int, *, gop: int, width: int = 64, height: int = 48
) -> list[bytes]:
    """Encode a moving test pattern to Annex B H.264, one access unit per frame.

    Matches what Roboto ingestion writes into a CompressedVideo topic: Annex B
    framing, parameter sets in-band at every keyframe, and no B-frames (``bf=0``)
    — the Foxglove CompressedVideo spec forbids them.
    """
    codec_context = av.CodecContext.create("libx264", "w")
    codec_context.width = width
    codec_context.height = height
    codec_context.pix_fmt = "yuv420p"
    codec_context.gop_size = gop
    codec_context.options = {
        "g": str(gop),
        "bf": "0",
        "tune": "zerolatency",
        "preset": "ultrafast",
    }

    access_units: list[bytes] = []
    for i in range(frame_count):
        pattern = np.zeros((height, width, 3), dtype=np.uint8)
        pattern[:, :, 0] = (i * 20) % 256
        pattern[:, : width // 2, 1] = 200
        frame = av.VideoFrame.from_ndarray(pattern, format="rgb24")
        for packet in codec_context.encode(frame.reformat(format="yuv420p")):
            access_units.append(bytes(packet))
    for packet in codec_context.encode(None):
        access_units.append(bytes(packet))
    return access_units


def _messages(
    access_units: list[bytes], *, codec: str = "h264"
) -> list[tuple[int, str, bytes]]:
    """Stamp encoded access units at a steady ~30 fps as ``(log_time, format, data)``."""
    return [
        (FIRST_LOG_TIME_NS + i * FRAME_INTERVAL_NS, codec, data)
        for i, data in enumerate(access_units)
    ]


@dataclass
class FakeVideoTopic:
    """A ``roboto.Topic`` stub serving an in-memory compressed-video stream.

    ``get_data_as_df`` returns what the SDK returns for a message-path read: a
    DataFrame indexed by log time, with one column per requested message path.
    Range filtering is real, which is what lets a test drive the keyframe
    lookback (the decoder asks for an earlier range than it was given).
    """

    topic_id: str = "tp_video"
    name: str = "/camera/hand/video"
    schema_name: str = COMPRESSED_VIDEO
    start_time: int | None = None
    end_time: int | None = None
    messages: list[tuple[int, str, bytes]] = field(default_factory=list)
    requested_ranges: list[tuple[int, int]] = field(default_factory=list)

    def get_data_as_df(
        self,
        *,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        message_paths_include: list[str] | None = None,
    ) -> pd.DataFrame:
        self.requested_ranges.append((start_time.value, end_time.value))
        rows = [
            (log_time, token, data)
            for log_time, token, data in self.messages
            if start_time.value <= log_time <= end_time.value
        ]
        return pd.DataFrame(
            {
                "format": [token for _, token, _ in rows],
                "data": [data for _, _, data in rows],
            },
            index=pd.DatetimeIndex(
                [pd.Timestamp(log_time, unit="ns") for log_time, _, _ in rows],
                name="log_time",
            ),
        )


def _topic_with(access_units: list[bytes], *, codec: str = "h264") -> FakeVideoTopic:
    messages = _messages(access_units, codec=codec)
    return FakeVideoTopic(
        messages=messages,
        start_time=messages[0][0],
        end_time=messages[-1][0],
    )


# ── decoding a range ────────────────────────────────────────────────────────


def test_chunk_starting_on_a_keyframe_decodes_standalone():
    """A range whose first message is a keyframe needs no history at all.

    This is the shape REASSEMBLE chunks have (forced IDR at every chunk
    boundary), so it must decode without a single lookback fetch.
    """
    topic = _topic_with(_encode_h264_access_units(12, gop=4))

    rows = decode_video_stream_rows(
        topic, topic.start_time, topic.end_time, topic_name=topic.name
    )

    assert len(rows) == 12
    assert [row["timestamp"] for row in rows] == [m[0] for m in topic.messages]
    assert topic.requested_ranges == [(topic.start_time, topic.end_time)]
    for row in rows:
        assert row["frame"].shape == (48, 64, 3)
        assert row["frame"].dtype == np.uint8


def test_range_starting_mid_gop_decodes_by_walking_back_to_the_keyframe():
    """Leading delta frames decode because the preceding GOP prefix is fetched.

    The episode boundary lands two frames into a GOP — the naive per-message
    decode this path replaces would produce nothing at all here.
    """
    topic = _topic_with(_encode_h264_access_units(12, gop=4))
    mid_gop_start = topic.messages[6][0]
    assert not is_keyframe(topic.messages[6][2]), "fixture must start mid-GOP"

    rows = decode_video_stream_rows(
        topic, mid_gop_start, topic.end_time, topic_name=topic.name
    )

    assert [row["timestamp"] for row in rows] == [m[0] for m in topic.messages[6:]]
    assert len(topic.requested_ranges) == 2, "the GOP prefix costs one extra fetch"
    lookback_start, lookback_end = topic.requested_ranges[1]
    assert lookback_start < mid_gop_start and lookback_end == mid_gop_start


def test_frames_are_resized_at_decode_time_when_the_contract_asks():
    """``resize`` is honoured where the frames are produced, to bound memory."""
    topic = _topic_with(_encode_h264_access_units(8, gop=4))

    rows = decode_video_stream_rows(
        topic, topic.start_time, topic.end_time, topic_name=topic.name, resize=(24, 32)
    )

    assert all(row["frame"].shape == (24, 32, 3) for row in rows)


def test_empty_range_yields_no_rows_rather_than_failing():
    """A topic with no messages in range is an ordinary empty episode, not an error."""
    topic = _topic_with(_encode_h264_access_units(4, gop=4))

    rows = decode_video_stream_rows(
        topic, topic.end_time + 10**9, topic.end_time + 2 * 10**9, topic_name=topic.name
    )

    assert rows == []


# ── failure modes ───────────────────────────────────────────────────────────


def test_keyframe_beyond_the_lookback_bound_fails_loudly():
    """Undecodable messages must raise, never silently produce a camera-less episode.

    A lookback too short to reach the anchoring keyframe leaves every message in
    range undecodable. The SDK decoder drops those silently (no player could
    render them either), so this layer is what turns the silence into an error.
    """
    topic = _topic_with(_encode_h264_access_units(8, gop=8))
    mid_gop_start = topic.messages[3][0]
    assert not any(is_keyframe(data) for _, _, data in topic.messages[3:])

    with pytest.raises(ValueError, match="none decoded"):
        decode_video_stream_rows(
            topic,
            mid_gop_start,
            topic.end_time,
            topic_name=topic.name,
            keyframe_lookback_ns=1,
        )


def test_frames_before_an_out_of_reach_keyframe_are_dropped_and_reported(caplog):
    """A partially decodable range keeps what it can and says how much it lost.

    Frames whose keyframe is out of reach are unrenderable by any player, so
    dropping them is right — but the loss must be visible, not silent. Here the
    range opens mid-GOP with the lookback disabled and only recovers at the next
    keyframe.
    """
    topic = _topic_with(_encode_h264_access_units(12, gop=8))
    mid_gop_start = topic.messages[5][0]

    with caplog.at_level("WARNING"):
        rows = decode_video_stream_rows(
            topic,
            mid_gop_start,
            topic.end_time,
            topic_name=topic.name,
            keyframe_lookback_ns=1,
        )

    assert [row["timestamp"] for row in rows] == [m[0] for m in topic.messages[8:]]
    assert "decoded 4 of 7" in caplog.text


def test_unsupported_codec_is_rejected_by_name():
    """A codec with no decode path names itself and the codecs that do decode."""
    topic = _topic_with(_encode_h264_access_units(4, gop=4), codec="mjpeg")

    with pytest.raises(ValueError, match="unsupported codec 'mjpeg'") as excinfo:
        decode_video_stream_rows(
            topic, topic.start_time, topic.end_time, topic_name=topic.name
        )

    assert "h264" in str(excinfo.value)


def test_stream_missing_in_band_parameter_sets_fails_loudly():
    """Ingestion guarantees SPS/PPS in-band at every keyframe; violating it must not pass.

    Strip them and no decoder can initialize. The failure mode worth guarding is
    silence — frames quietly missing from the exported episode — so assert the
    exception rather than an empty result.
    """
    stripped = [
        _without_parameter_sets(unit)
        for unit in _encode_h264_access_units(12, gop=4)
    ]
    topic = _topic_with(stripped)

    with pytest.raises(ValueError, match="in-band parameter sets"):
        decode_video_stream_rows(
            topic, topic.start_time, topic.end_time, topic_name=topic.name
        )


def _without_parameter_sets(access_unit: bytes) -> bytes:
    """Re-frame an access unit with its SPS/PPS NAL units removed."""
    return b"".join(
        b"\x00\x00\x00\x01" + unit.data
        for unit in find_nal_units(access_unit)
        if unit.type
        not in (NalUnitType.SEQUENCE_PARAMETER_SET, NalUnitType.PICTURE_PARAMETER_SET)
    )


# ── the packaging guarantees the decoder relies on ──────────────────────────


def test_encoded_keyframes_carry_in_band_parameter_sets():
    """Every keyframe is a self-sufficient decoder entry point.

    This is the Foxglove CompressedVideo packaging rule that makes
    decode-from-an-arbitrary-keyframe work; the fixture asserts it so a fixture
    that stopped honouring it could not quietly weaken every other test here.
    """
    keyframes = [
        unit for unit in _encode_h264_access_units(12, gop=4) if is_keyframe(unit)
    ]

    assert len(keyframes) == 3
    for keyframe in keyframes:
        nal_types = {unit.type for unit in find_nal_units(keyframe)}
        assert NalUnitType.SEQUENCE_PARAMETER_SET in nal_types
        assert NalUnitType.PICTURE_PARAMETER_SET in nal_types


# ── detection ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "schema_name",
    [
        "foxglove_msgs/msg/CompressedVideo",
        "foxglove_msgs/CompressedVideo",
        "foxglove.CompressedVideo",
        "lemi.bot.msg.compressed_video_packet.CompressedVideoPacket",
    ],
)
def test_compressed_video_schemas_are_recognised(schema_name: str):
    assert is_compressed_video_schema(schema_name)


@pytest.mark.parametrize(
    "schema_name",
    [
        "sensor_msgs/msg/CompressedImage",
        "sensor_msgs/msg/Image",
        "foxglove_msgs/msg/CompressedVideoInfo",
    ],
)
def test_non_video_schemas_are_not_recognised(schema_name: str):
    assert not is_compressed_video_schema(schema_name)


def test_topic_is_detected_from_the_representation_format_ingestion_registers():
    """``compressedVideo`` on the representation is the label ingestion writes."""

    class _Representation:
        format = "compressedVideo"

    class _Topic:
        schema_name = "some_vendor/msg/ProprietaryFrame"
        default_representation = _Representation()

    assert is_compressed_video_topic(_Topic())


def test_compressed_image_topic_is_not_mistaken_for_video():
    class _Representation:
        format = "jpeg"

    class _Topic:
        schema_name = "sensor_msgs/msg/CompressedImage"
        default_representation = _Representation()

    assert not is_compressed_video_topic(_Topic())


# ── routing ─────────────────────────────────────────────────────────────────


def _video_spec(
    *, type_str: str = COMPRESSED_VIDEO, resize: tuple[int, int] | None = None
) -> ObservationSpec:
    return ObservationSpec(
        key="observation.images.hand",
        topic="/camera/hand/video",
        type=type_str,
        image={"resize": list(resize)} if resize else {},
        align=AlignSpec(method="hold", tolerance_ms=None),
    )


def test_compressed_video_spec_routes_to_the_stream_loader():
    assert video_spec_kind(_video_spec(), [FakeVideoTopic()]) == "video_stream"


def test_compressed_image_spec_still_routes_to_the_per_message_loader():
    """The shipping still-image path must not be pulled into the video route."""
    spec = _video_spec(type_str="sensor_msgs/msg/CompressedImage")

    class _StillImageTopic:
        schema_name = "sensor_msgs/msg/CompressedImage"

    assert video_spec_kind(spec, [_StillImageTopic()]) == "video_msgs"


def test_video_topic_declared_as_compressed_image_is_rejected():
    """A mis-declared contract must name the problem, not hand H.264 to cv2."""
    spec = _video_spec(type_str="sensor_msgs/msg/CompressedImage")

    with pytest.raises(ValueError, match="stores compressed video"):
        video_spec_kind(spec, [FakeVideoTopic()])


def test_every_routed_schema_has_a_registered_decoder():
    """Routing and decoding are keyed on the same strings; drift would be a runtime error."""
    for schema_name in ("foxglove_msgs/msg/CompressedVideo", "foxglove_msgs/CompressedVideo"):
        assert video_spec_kind(_video_spec(type_str=schema_name), []) == "video_stream"
        assert schema_name in DECODERS


# ── end to end: DataCollection → generate_frames ────────────────────────────


def _contract(video: ObservationSpec, fps: float = 30.0) -> Contract:
    return Contract(
        name="compressed-video",
        version=1,
        fps=fps,
        action_lead_steps=0,
        observations=[],
        videos=[video],
        actions=[],
        tasks=[],
        robot_type=None,
    )


def test_compressed_video_topic_becomes_rgb_frames_on_the_reference_timeline():
    """The whole path: fetch → GOP decode → merge → one HWC uint8 RGB frame per row."""
    topic = _topic_with(_encode_h264_access_units(12, gop=4))
    spec = _video_spec(resize=(24, 32))
    contract = _contract(spec)

    collection = DataCollection(
        contract, {spec.topic: [topic]}, topic.start_time, topic.end_time
    )
    assert list(collection.videos[spec.key].columns) == ["timestamp", "frame"]
    assert len(collection.videos[spec.key]) == 12

    reference = pd.Series(
        [FIRST_LOG_TIME_NS + i * FRAME_INTERVAL_NS for i in range(4)],
        name="timestamp",
    )
    frames = list(generate_frames(contract, collection, reference, task="pick"))

    assert len(frames) == 4
    for frame in frames:
        image = frame[spec.key]
        assert isinstance(image, np.ndarray)
        assert image.shape == (24, 32, 3)
        assert image.dtype == np.uint8
        assert frame["task"] == "pick"


def test_deferred_decode_round_trip_yields_the_same_frames():
    """The worker/main-process split (``defer_image_decode``) is unaffected.

    Compressed video arrives pre-decoded, so its deferred payload is an array
    rather than encoded bytes; ``materialize_deferred`` must still resolve it
    through the registry to the same image.
    """
    from roboto_to_lerobot.lerobot import materialize_deferred

    topic = _topic_with(_encode_h264_access_units(8, gop=4))
    spec = _video_spec(resize=(24, 32))
    contract = _contract(spec)
    collection = DataCollection(
        contract, {spec.topic: [topic]}, topic.start_time, topic.end_time
    )
    reference = pd.Series(
        [FIRST_LOG_TIME_NS + i * FRAME_INTERVAL_NS for i in range(3)], name="timestamp"
    )

    inline = list(generate_frames(contract, collection, reference))
    deferred = list(
        generate_frames(contract, collection, reference, defer_image_decode=True)
    )
    for frame in deferred:
        materialize_deferred(frame, {spec.key: spec})

    assert len(deferred) == len(inline) == 3
    for deferred_frame, inline_frame in zip(deferred, inline, strict=True):
        np.testing.assert_array_equal(deferred_frame[spec.key], inline_frame[spec.key])


# ── real ingested data ──────────────────────────────────────────────────────


@requires_reassemble_recording
def test_real_reassemble_recording_decodes_end_to_end():
    """Decode a genuine ingested H.264 camera stream, not a test-encoded one.

    Guards against the fixtures being unrepresentative: a real recording brings
    its own encoder settings, GOP length, and resolution.
    """
    from mcap_ros2.reader import read_ros2_messages

    messages = [
        (message.log_time_ns, message.ros_msg.format, bytes(message.ros_msg.data))
        for message in read_ros2_messages(REASSEMBLE_MCAP, topics=[REASSEMBLE_TOPIC])
    ]
    assert messages, "recording has no messages on the camera topic"
    assert {token for _, token, _ in messages} == {"h264"}

    topic = FakeVideoTopic(
        name=REASSEMBLE_TOPIC,
        messages=messages,
        start_time=messages[0][0],
        end_time=messages[-1][0],
    )
    # Two seconds from a mid-GOP boundary — the shape an episode event carves out.
    start = messages[0][0] + 5 * 10**9
    end = start + 2 * 10**9
    expected = [log_time for log_time, _, _ in messages if start <= log_time <= end]

    rows = decode_video_stream_rows(topic, start, end, topic_name=REASSEMBLE_TOPIC)

    assert [row["timestamp"] for row in rows] == expected
    assert {row["frame"].shape for row in rows} == {(480, 640, 3)}
    assert all(row["frame"].dtype == np.uint8 for row in rows)


@requires_reassemble_recording
def test_real_recording_carries_parameter_sets_at_every_keyframe():
    """The ingested stream honours the packaging rule the range decode depends on."""
    from mcap_ros2.reader import read_ros2_messages

    keyframes = [
        bytes(message.ros_msg.data)
        for message in read_ros2_messages(REASSEMBLE_MCAP, topics=[REASSEMBLE_TOPIC])
        if is_keyframe(bytes(message.ros_msg.data))
    ]

    assert keyframes
    for keyframe in keyframes:
        nal_types = {unit.type for unit in find_nal_units(keyframe)}
        assert NalUnitType.SEQUENCE_PARAMETER_SET in nal_types
        assert NalUnitType.PICTURE_PARAMETER_SET in nal_types
