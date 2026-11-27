"""Tests for ``selector.lerobot_names`` without ``selector.names`` (labels by position).

Some recordings' messages carry no names to select values by, such as a
``JointState`` whose ``name`` field is empty. The decoder then emits the whole
vector, and ``lerobot_names`` alone labels it by position. The labels must
reach the dataset's ``meta/info.json``, a message with a different number of
values must fail rather than be mislabelled, offline and live, and a type that
does not decode to a vector of numbers in a fixed order must be refused when
the contract loads.
"""

from __future__ import annotations

import json
from textwrap import dedent
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from roboto_to_lerobot.contract_utils import DataCollection, load_contract
from roboto_to_lerobot.runtime import LiveAdapter
from roboto_to_lerobot.runtime import load_contract as load_live_contract
from test_episode_cache import _SECOND, _T0, _events, _FakeTopic, _Harness, _recording

_JOINT_STATE_CONTRACT = """\
name: positional
version: 1
fps: 10
observations:
  - key: observation.state
    topic: /arm/joint_states
    type: sensor_msgs/msg/JointState
    selector:
      lerobot_names: [shoulder, elbow]
actions: []
"""

_JOINT_STATE_ACTION_CONTRACT = """\
name: positional
version: 1
fps: 10
observations: []
actions:
  - key: action
    topic: /arm/joint_states
    type: sensor_msgs/msg/JointState
    selector:
      lerobot_names: [shoulder, elbow]
"""


def _topic(topic_name: str, timestamps: np.ndarray, **columns) -> _FakeTopic:
    """A topic of messages stamped ``timestamps`` with the given field columns."""
    df = pd.DataFrame(
        {"header.stamp.sec": timestamps // _SECOND, "header.stamp.nanosec": timestamps % _SECOND, **columns},
        index=pd.to_datetime(timestamps),
    )
    return _FakeTopic(topic_name, df)


@pytest.fixture
def harness(tmp_path, monkeypatch) -> _Harness:
    return _Harness(tmp_path, monkeypatch)


def test_positional_names_label_the_features_in_info_json(harness):
    # Arrange
    harness.contract_path.write_text(dedent("""\
        name: positional
        version: 1
        fps: 10
        observations:
          - key: observation.images.cam
            topic: /cam
            type: sensor_msgs/msg/CompressedImage
            image: {resize: [64, 64]}
            align: {method: nearest, tolerance_ms: 100}
          - key: observation.state
            topic: /arm/joint_states
            type: sensor_msgs/msg/JointState
            selector: {lerobot_names: [shoulder, elbow]}
          - key: observation.state
            topic: /gripper
            type: std_msgs/msg/Float64
            selector: {names: [data], lerobot_names: [gripper]}
        actions:
          - key: action
            topic: /arm/command
            type: std_msgs/msg/Float64MultiArray
            selector: {lerobot_names: [shoulder_cmd, elbow_cmd]}
    """))
    ts = _T0 + np.arange(300) * (_SECOND // 50)
    rng = np.random.default_rng(0)
    harness.topics = {
        "/cam": _recording()["/cam"],
        # The name field holds no joint names: it is empty, or holds one label that names no joint.
        "/arm/joint_states": [_topic(
            "/arm/joint_states", ts, name=[[]] * 150 + [["arm"]] * 150, position=rng.normal(size=(300, 2)).tolist(),
        )],
        "/gripper": [_topic("/gripper", ts, data=rng.normal(size=300))],
        "/arm/command": [_topic("/arm/command", ts, data=list(rng.normal(size=(300, 2))))],
    }

    # Act
    root = harness.export(_events(0))

    # Assert
    features = json.loads((root / "meta" / "info.json").read_text())["features"]
    assert features["observation.state"]["names"] == ["shoulder", "elbow", "gripper"]
    assert features["observation.state"]["shape"] == [3]
    assert features["action"]["names"] == ["shoulder_cmd", "elbow_cmd"]
    assert features["action"]["shape"] == [2]


def _decode_offline(contract_path, positions: list[list[float]]) -> None:
    ts = _T0 + np.arange(len(positions)) * (_SECOND // 10)
    topic = _topic("/arm/joint_states", ts, name=[[]] * len(positions), position=positions)
    DataCollection(load_contract(contract_path), {"/arm/joint_states": [topic]}, int(ts[0]), int(ts[-1]))


def _decode_live(contract_path, positions: list[list[float]]) -> None:
    adapter = LiveAdapter(load_live_contract(contract_path))
    for i, position in enumerate(positions):
        adapter.on_message("/arm/joint_states", SimpleNamespace(name=[], position=position), ts_ns=_T0 + i)


@pytest.mark.parametrize(
    ("decode", "contract", "key"),
    [
        (_decode_offline, _JOINT_STATE_CONTRACT, "observation.state"),
        (_decode_offline, _JOINT_STATE_ACTION_CONTRACT, "action"),
        (_decode_live, _JOINT_STATE_CONTRACT, "observation.state"),
    ],
    ids=["converter-observation", "converter-action", "live"],
)
def test_message_with_another_number_of_values_fails_naming_key_topic_and_lengths(tmp_path, decode, contract, key):
    # Arrange
    path = tmp_path / "contract.yaml"
    path.write_text(contract)
    # The first message matches the two names, so only the second can fail.
    positions = [[0.1, 0.2], [0.1, 0.2, 0.3]]

    # Act
    with pytest.raises(ValueError) as exc:
        decode(path, positions)

    # Assert
    message = str(exc.value)
    assert key in message
    assert "/arm/joint_states" in message
    assert "3 values" in message
    assert "labels 2 values" in message


@pytest.mark.parametrize("msg_type", ["std_msgs/msg/String", "string_typed_msg"])
def test_positional_names_on_a_type_without_a_fixed_vector_are_refused_on_load(tmp_path, msg_type):
    # Arrange
    path = tmp_path / "contract.yaml"
    path.write_text(_JOINT_STATE_CONTRACT.replace("sensor_msgs/msg/JointState", msg_type))

    # Act
    with pytest.raises(ValueError) as exc:
        load_contract(path)

    # Assert
    message = str(exc.value)
    assert "observation.state" in message
    assert msg_type in message
