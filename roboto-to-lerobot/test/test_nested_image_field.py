"""Tests for camera images nested in a custom message (``image.field``).

A recording can carry every camera inside one custom message, each camera a
whole ``CompressedImage`` or ``Image`` at its own message path, with no topic
per camera. A spec whose ``type`` names the custom message and whose
``image.field`` and ``image.type`` name the path and the image message there
must yield exactly the frames the same images would yield published on their
own topic, offline and live. Each offline test loads its contract from YAML
and decodes frames the way the converter does (``DataCollection``, then
``generate_frames`` with image decoding deferred, then ``materialize_deferred``)
from a fake topic that shapes its reads the way the SDK does for each storage
format: an MCAP-backed read flattens nested fields into dotted columns, and a
Parquet-backed read returns each top-level field as one column of dicts.
"""

from __future__ import annotations

import array
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml
from roboto_to_lerobot.contract_utils import (
    Contract,
    DataCollection,
    collect_topics_from_files,
    load_contract,
)
from roboto_to_lerobot.lerobot import (
    episode_video_reader,
    generate_frames,
    materialize_deferred,
)
from roboto_to_lerobot.runtime import LiveAdapter
from roboto_to_lerobot.runtime import load_contract as load_live_contract

FIRST_LOG_TIME_NS = 1_700_000_000_000_000_000
FRAME_INTERVAL_NS = 50_000_000
FRAME_COUNT = 4
HEIGHT, WIDTH = 24, 32
COMPRESSED_IMAGE = "sensor_msgs/msg/CompressedImage"
IMAGE = "sensor_msgs/msg/Image"
OBSERVATION = "custom_msgs/msg/Observation"


# ── fixtures ────────────────────────────────────────────────────────────────


def _pattern(frame: int, seed: int) -> np.ndarray:
    """A BGR test image that differs by frame and by camera (``seed``)."""
    rows = np.arange(HEIGHT, dtype=np.uint16)[:, None]
    cols = np.arange(WIDTH, dtype=np.uint16)[None, :]
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    image[..., 0] = (rows * 8 + frame * 30 + seed * 70) % 256
    image[..., 1] = (cols * 6 + seed * 50) % 256
    image[..., 2] = (frame * 60 + seed * 90) % 256
    return image


def _image_message(msg_type: str, image_format: str | None, frame: int, seed: int) -> dict[str, Any]:
    """One camera message of ``msg_type`` as a nested dict, the way a decoded record reads.

    ``image_format`` is the CompressedImage ``format`` field; an Image ignores it.
    """
    header = {"stamp": {"sec": frame, "nanosec": 0}, "frame_id": f"camera_{seed}"}
    bgr = _pattern(frame, seed)
    if msg_type == COMPRESSED_IMAGE:
        ok, encoded = cv2.imencode(".jpg", bgr)
        assert ok
        return {"header": header, "format": image_format, "data": encoded.tobytes()}
    return {
        "header": header,
        "height": HEIGHT,
        "width": WIDTH,
        "encoding": "bgr8",
        "is_bigendian": 0,
        "step": WIDTH * 3,
        "data": bgr.tobytes(),
    }


def _nest(path: str, value: Any) -> dict[str, Any]:
    """``value`` placed at the dotted message ``path`` of an otherwise empty message."""
    for part in reversed(path.split(".")):
        value = {part: value}
    return value


def _merge(into: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    """``other`` merged into ``into`` in place, key by key at every depth; returns ``into``."""
    for key, value in other.items():
        if isinstance(value, dict) and isinstance(into.get(key), dict):
            _merge(into[key], value)
        else:
            into[key] = value
    return into


def _project(message: dict[str, Any], paths: list[str]) -> dict[str, Any]:
    """The part of ``message`` the message paths ``paths`` select, nesting kept."""
    out: dict[str, Any] = {}
    for path in paths:
        source = message
        for part in path.split("."):
            source = source[part]
        _merge(out, _nest(path, source))
    return out


def _leaf_paths(message: dict[str, Any], prefix: str = "") -> list[str]:
    paths = []
    for key, value in message.items():
        path = f"{prefix}{key}"
        paths.extend(_leaf_paths(value, f"{path}.") if isinstance(value, dict) else [path])
    return paths


@dataclass
class RecordTopic:
    """A ``roboto.Topic`` stub serving in-memory messages the way the SDK reads them.

    ``store`` picks the shape of ``get_data_as_df``: ``"mcap"`` flattens every
    selected field into dotted columns (``pd.json_normalize``), ``"parquet"``
    returns one column per top-level field holding each message's dict.
    ``path_types`` is the type the topic's record lists for a message path,
    as Roboto records a nested message's type. ``fetches`` counts the reads.
    """

    name: str
    store: str
    messages: list[dict[str, Any]]
    topic_id: str = "tp_record"
    file_id: str = "fl_record"
    schema_name: str = OBSERVATION
    path_types: dict[str, str] = field(default_factory=dict)
    fetches: int = field(init=False, default=0)
    message_count: int = field(init=False)
    log_times: list[int] = field(init=False)

    def __post_init__(self) -> None:
        self.log_times = [FIRST_LOG_TIME_NS + i * FRAME_INTERVAL_NS for i in range(len(self.messages))]
        self.message_count = len(self.messages)

    @property
    def start_time(self) -> int:
        return self.log_times[0]

    @property
    def end_time(self) -> int:
        return self.log_times[-1]

    @property
    def message_paths(self) -> list[SimpleNamespace]:
        paths = dict.fromkeys([*_leaf_paths(self.messages[0]), *self.path_types])
        return [SimpleNamespace(message_path=path, data_type=self.path_types.get(path)) for path in paths]

    def get_data_as_df(
        self,
        *,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        message_paths_include: list[str] | None = None,
    ) -> pd.DataFrame:
        self.fetches += 1
        selected = [
            (log_time, _project(message, message_paths_include) if message_paths_include else message)
            for log_time, message in zip(self.log_times, self.messages, strict=True)
            if start_time.value <= log_time <= end_time.value
        ]
        index = pd.DatetimeIndex([pd.Timestamp(t, unit="ns") for t, _ in selected], name="log_time")
        records = [message for _, message in selected]
        if self.store == "mcap":
            df = pd.json_normalize(records)
            df.index = index
            return df
        return pd.DataFrame(records, index=index)


def _video(key: str, topic: str, image_type: str, path: str | None = None) -> dict[str, Any]:
    """One video spec as a contract's YAML declares it.

    Without ``path`` the topic publishes ``image_type`` messages. With it the
    topic publishes ``OBSERVATION`` messages carrying an ``image_type`` message
    at the message path ``path``.
    """
    spec: dict[str, Any] = {
        "key": key, "topic": topic, "type": image_type, "image": {"resize": [HEIGHT, WIDTH]},
        "align": {"method": "nearest", "tolerance_ms": None},
    }
    if path is not None:
        spec["type"] = OBSERVATION
        spec["image"].update(field=path, type=image_type)
    return spec


def _contract(tmp_path, videos: list[dict[str, Any]]) -> Contract:
    path = tmp_path / "contract.yaml"
    path.write_text(yaml.safe_dump({"name": "nested", "version": 1, "fps": 20, "observations": videos}))
    return load_contract(path)


def _frames(contract: Contract, topics: list[RecordTopic]) -> dict[str, list[np.ndarray]]:
    """Every video spec's frames keyed by spec key, decoded as the converter decodes them
    (``generate_frames`` defers image decoding and ``materialize_deferred`` decodes each frame,
    reading compressed video through one reader over the episode)."""
    first = topics[0]
    collection = DataCollection(
        contract, {topic.name: [topic] for topic in topics}, first.start_time, first.end_time,
    )
    reference = pd.Series(first.log_times, name="timestamp")
    specs_by_key = {spec.key: spec for spec in contract.videos}
    frames = list(generate_frames(contract, collection, reference, defer_image_decode=True))
    video_reader = episode_video_reader(frames)
    for frame in frames:
        materialize_deferred(frame, specs_by_key, video_reader)
    return {spec.key: [frame[spec.key] for frame in frames] for spec in contract.videos}


# ── decoding ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("store", ["mcap", "parquet"])
def test_cameras_nested_in_one_topic_each_yield_their_own_frames_from_one_fetch(tmp_path, store):
    """Specs reading different paths of one topic, CompressedImage and Image at
    different depths, each get their own camera's frames, and the topic is read
    by a single fetch rather than once per camera."""
    # Arrange: three cameras in one message, two CompressedImage one field deep
    # and one Image three fields deep.
    paths = {"front": COMPRESSED_IMAGE, "rear": COMPRESSED_IMAGE, "arm.cameras.wrist": IMAGE}
    cameras = {
        path: [_image_message(image_type, "jpeg", frame, seed) for frame in range(FRAME_COUNT)]
        for seed, (path, image_type) in enumerate(paths.items())
    }
    messages = [{} for _ in range(FRAME_COUNT)]
    for path, images in cameras.items():
        for message, image in zip(messages, images, strict=True):
            _merge(message, _nest(path, image))
    topic = RecordTopic(name="/observation", store=store, messages=messages)
    contract = _contract(tmp_path, [
        _video(f"observation.images.{path}", "/observation", image_type, path)
        for path, image_type in paths.items()
    ])
    alone = _contract(tmp_path, [
        _video(f"observation.images.{path}", f"/{path}", image_type) for path, image_type in paths.items()
    ])
    alone_topics = [
        RecordTopic(name=f"/{path}", store=store, topic_id=f"tp_{path}", messages=images)
        for path, images in cameras.items()
    ]

    # Act
    frames = _frames(contract, [topic])

    # Assert
    assert topic.fetches == 1
    expected = _frames(alone, alone_topics)
    for key, key_frames in frames.items():
        assert len(key_frames) == FRAME_COUNT
        np.testing.assert_array_equal(key_frames, expected[key])


def test_nested_compressed_video_decodes_to_the_frames_of_the_same_stream_on_its_own_topic(tmp_path):
    """A compressed-video stream at a message path decodes like the same stream on its own topic."""
    av = pytest.importorskip("av")
    # Arrange: a short H.264 stream, one access unit per message.
    codec_context = av.CodecContext.create("libx264", "w")
    codec_context.width, codec_context.height, codec_context.pix_fmt = WIDTH, HEIGHT, "yuv420p"
    codec_context.options = {"g": "4", "bf": "0", "tune": "zerolatency", "preset": "ultrafast"}
    access_units: list[bytes] = []
    for frame in [*(_pattern(i, seed=3) for i in range(8)), None]:
        video_frame = None if frame is None else av.VideoFrame.from_ndarray(frame, format="bgr24").reformat(
            format="yuv420p"
        )
        access_units.extend(bytes(packet) for packet in codec_context.encode(video_frame))
    messages = [{"format": "h264", "data": unit} for unit in access_units]
    standalone = RecordTopic(name="/video", store="mcap", messages=messages)
    nested = RecordTopic(
        name="/observation", store="mcap", topic_id="tp_nested",
        messages=[_nest("sensors.video", message) for message in messages],
    )
    contract = _contract(tmp_path, [
        _video("standalone", "/video", "foxglove_msgs/msg/CompressedVideo"),
        _video("nested", "/observation", "foxglove_msgs/msg/CompressedVideo", "sensors.video"),
    ])

    # Act
    frames = _frames(contract, [standalone, nested])

    # Assert
    assert len(frames["nested"]) == len(access_units)
    np.testing.assert_array_equal(frames["nested"], frames["standalone"])


# ── a path the topic does not have ──────────────────────────────────────────


@dataclass
class FakeFile:
    relative_path: str
    topics: list[RecordTopic]

    def get_topics(self) -> list[RecordTopic]:
        return list(self.topics)


def test_path_missing_from_the_topic_fails_when_topics_are_collected(tmp_path):
    """A path the topic lacks fails before any fetch, naming the spec, the topic
    and the path, and listing the paths that have the fields of the declared image message."""
    # Arrange
    topic = RecordTopic(
        name="/observation", store="mcap",
        messages=[
            _merge(
                _nest("sensors.vision.camera", _image_message(COMPRESSED_IMAGE, "jpeg", 0, seed=1)),
                _nest("camera", _image_message(COMPRESSED_IMAGE, "jpeg", 0, seed=2)),
            )
        ],
    )
    contract = _contract(tmp_path, [
        _video("observation.images.front", "/observation", COMPRESSED_IMAGE, "sensors.camera"),
    ])

    # Act / Assert
    with pytest.raises(ValueError) as excinfo:
        collect_topics_from_files([FakeFile("recording.mcap", [topic])], contract, scope="test")
    message = str(excinfo.value)
    assert topic.fetches == 0
    assert "'observation.images.front'" in message
    assert "'/observation'" in message
    assert "'sensors.camera'" in message
    assert "['camera', 'sensors.vision.camera']" in message


@pytest.mark.parametrize(
    ("declared", "recorded"),
    [
        pytest.param(COMPRESSED_IMAGE, "sensor_msgs/CompressedImage", id="contract_spells_msg"),
        pytest.param("sensor_msgs/CompressedImage", COMPRESSED_IMAGE, id="topic_spells_msg"),
    ],
)
def test_image_type_matches_the_recorded_type_in_either_spelling(tmp_path, declared, recorded):
    """``pkg/Name`` and ``pkg/msg/Name`` name the same image type: the topics are
    collected and the image decodes whichever spelling the contract and the
    topic's record use."""
    # Arrange
    images = [_image_message(COMPRESSED_IMAGE, "jpeg", frame, seed=1) for frame in range(FRAME_COUNT)]
    standalone = RecordTopic(name="/camera", store="mcap", messages=images)
    nested = RecordTopic(
        name="/observation", store="mcap", topic_id="tp_nested",
        messages=[_nest("camera", image) for image in images], path_types={"camera": recorded},
    )
    contract = _contract(tmp_path, [
        _video("standalone", "/camera", COMPRESSED_IMAGE),
        _video("nested", "/observation", declared, "camera"),
    ])

    # Act
    topics, _ = collect_topics_from_files([FakeFile("recording.mcap", [standalone, nested])], contract, scope="test")
    frames = _frames(contract, [topics["/camera"][0], topics["/observation"][0]])

    # Assert
    assert len(frames["nested"]) == FRAME_COUNT
    np.testing.assert_array_equal(frames["nested"], frames["standalone"])


@pytest.mark.parametrize(
    ("topic_overrides", "declared", "recorded"),
    [
        pytest.param({"schema_name": "custom_msgs/msg/Status"}, OBSERVATION, "custom_msgs/msg/Status", id="topic_type"),
        pytest.param({"path_types": {"camera": IMAGE}}, COMPRESSED_IMAGE, IMAGE, id="image_type"),
    ],
)
def test_type_the_topic_does_not_record_fails_when_topics_are_collected(
    tmp_path, topic_overrides, declared, recorded,
):
    """A nested image spec whose ``type`` is not the topic's recorded type, or whose
    ``image.type`` is not the type recorded at its path, fails before any fetch,
    naming the spec, the topic, the declared type and the recorded one."""
    # Arrange
    topic = RecordTopic(
        name="/observation", store="mcap",
        messages=[_nest("camera", _image_message(COMPRESSED_IMAGE, "jpeg", 0, seed=1))],
        **topic_overrides,
    )
    contract = _contract(tmp_path, [_video("observation.images.front", "/observation", COMPRESSED_IMAGE, "camera")])

    # Act / Assert
    with pytest.raises(ValueError) as excinfo:
        collect_topics_from_files([FakeFile("recording.mcap", [topic])], contract, scope="test")
    message = str(excinfo.value)
    assert topic.fetches == 0
    assert "'observation.images.front'" in message
    assert "'/observation'" in message
    assert f"'{declared}'" in message
    assert f"'{recorded}'" in message


# ── live inference ──────────────────────────────────────────────────────────


def _ros_message(value: Any) -> Any:
    """A nested dict as a ROS message object: fields read as attributes, at every depth."""
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _ros_message(item) for key, item in value.items()})
    if isinstance(value, bytes):
        # ROS 2 Python messages hold uint8[] fields as array.array("B"), not bytes.
        return array.array("B", value)
    return value


def test_live_adapter_decodes_nested_images_to_the_offline_frames(tmp_path):
    """The live runtime, handed each custom message as a ROS message object,
    decodes every nested image to the frame the offline conversion produces
    from the same message."""
    # Arrange: a CompressedImage two fields deep and an Image three deep.
    compressed = [_image_message(COMPRESSED_IMAGE, "jpeg", frame, seed=1) for frame in range(FRAME_COUNT)]
    raw = [_image_message(IMAGE, None, frame, seed=2) for frame in range(FRAME_COUNT)]
    messages = [
        _merge(_nest("cameras.front", a), _nest("arm.cameras.wrist", b))
        for a, b in zip(compressed, raw, strict=True)
    ]
    topic = RecordTopic(name="/observation", store="mcap", messages=messages)
    contract = _contract(tmp_path, [
        _video("observation.images.front", "/observation", COMPRESSED_IMAGE, "cameras.front"),
        _video("observation.images.wrist", "/observation", IMAGE, "arm.cameras.wrist"),
    ])
    offline = _frames(contract, [topic])
    adapter = LiveAdapter(load_live_contract(tmp_path / "contract.yaml"))

    # Act
    live = []
    for log_time, message in zip(topic.log_times, messages, strict=True):
        adapter.on_message("/observation", _ros_message(message), log_time)
        live.append(adapter.sample(log_time))

    # Assert
    for key, frames in offline.items():
        np.testing.assert_array_equal([sample[key] for sample in live], frames)
