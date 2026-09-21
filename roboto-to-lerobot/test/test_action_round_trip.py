"""Round-trip tests for the live adapter's action path.

Each encoder/decoder pair should be a true inverse on its own message
shape — encode a synthetic action, run the resulting payload back
through the matching decoder, recover the input array. If a future
encoder change drifts away from the decoder's read pattern, these tests
turn into the early-warning signal.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from roboto_to_lerobot.contract_utils import (
    ActionSpec,
    ObservationSpec,
    TransformSpec,
)
from roboto_to_lerobot.contract_utils import (
    Contract as InnerContract,
)
from roboto_to_lerobot.runtime.contract_io import Contract as RuntimeContract
from roboto_to_lerobot.runtime.converters import decode_value
from roboto_to_lerobot.runtime.live_adapter import LiveAdapter


def _make_contract(
    *,
    fps: float = 30.0,
    observations: list[ObservationSpec] | None = None,
    actions: list[ActionSpec] | None = None,
) -> RuntimeContract:
    inner = InnerContract(
        name="test",
        version=1,
        fps=fps,
        observations=observations or [],
        videos=[],
        actions=actions or [],
        tasks=[],
    )
    return RuntimeContract(inner=inner, source_path=Path("/tmp/fake-contract.yaml"))


# ----------------------------------------------------------------------------
# Single-spec encoders
# ----------------------------------------------------------------------------


def test_joint_state_round_trips_through_adapter():
    spec = ActionSpec(
        key="action",
        topic="/teleop/action",
        type="sensor_msgs/msg/JointState",
        selector={"names": ["joint_a", "joint_b", "joint_c"]},
    )
    adapter = LiveAdapter(_make_contract(actions=[spec]))

    src = np.array([0.1, -0.2, 1.5], dtype=np.float64)
    pairs = adapter.encode_action({"action": src}, now_ns=0)

    assert len(pairs) == 1
    topic, payload = pairs[0]
    assert topic == "/teleop/action"
    assert payload["name"] == ["joint_a", "joint_b", "joint_c"]
    # encode_action casts the action to float32 to match the converter's action
    # feature dtype, so positions carry float32 precision rather than the
    # float64 src — compare with a tolerance, not exact equality.
    np.testing.assert_allclose(payload["position"], [0.1, -0.2, 1.5], rtol=1e-6)

    # The JointState decoder takes its pandas-Series/dict branch via
    # ``msg['name']`` / ``msg['position']`` item access, so feed it the row
    # dict the encoder produced and confirm it round-trips back to src.
    decoded = decode_value({"name": payload["name"], "position": payload["position"]}, spec)
    np.testing.assert_allclose(decoded, src, rtol=1e-6)


def test_float64_multiarray_round_trips_through_adapter():
    spec = ActionSpec(
        key="action",
        topic="/action_vec",
        type="std_msgs/msg/Float64MultiArray",
    )
    adapter = LiveAdapter(_make_contract(actions=[spec]))

    src = np.array([1.5, 2.5, 3.5])
    pairs = adapter.encode_action({"action": src}, now_ns=0)
    topic, payload = pairs[0]
    assert topic == "/action_vec"
    msg = SimpleNamespace(data=payload["data"])
    np.testing.assert_allclose(decode_value(msg, spec), src)


def test_float64_scalar_round_trips_through_adapter():
    spec = ActionSpec(
        key="action",
        topic="/action_scalar",
        type="std_msgs/msg/Float64",
    )
    adapter = LiveAdapter(_make_contract(actions=[spec]))

    pairs = adapter.encode_action({"action": np.array([0.7])}, now_ns=0)
    topic, payload = pairs[0]
    assert topic == "/action_scalar"
    msg = SimpleNamespace(data=payload["data"])
    np.testing.assert_allclose(decode_value(msg, spec), [0.7])


# ----------------------------------------------------------------------------
# Multi-spec same key (slicing)
# ----------------------------------------------------------------------------


def test_multi_action_same_key_slices_in_declaration_order():
    spec_left = ActionSpec(
        key="action",
        topic="/arm_left/action",
        type="sensor_msgs/msg/JointState",
        selector={"names": ["left_a", "left_b"]},
    )
    spec_right = ActionSpec(
        key="action",
        topic="/arm_right/action",
        type="sensor_msgs/msg/JointState",
        selector={"names": ["right_a", "right_b", "right_c"]},
    )
    adapter = LiveAdapter(_make_contract(actions=[spec_left, spec_right]))

    action = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pairs = adapter.encode_action({"action": action}, now_ns=0)

    assert pairs == [
        ("/arm_left/action", {"name": ["left_a", "left_b"], "position": [1.0, 2.0]}),
        ("/arm_right/action", {"name": ["right_a", "right_b", "right_c"], "position": [3.0, 4.0, 5.0]}),
    ]


def test_multi_action_short_array_raises():
    spec_left = ActionSpec(
        key="action",
        topic="/arm_left/action",
        type="sensor_msgs/msg/JointState",
        selector={"names": ["left_a", "left_b"]},
    )
    spec_right = ActionSpec(
        key="action",
        topic="/arm_right/action",
        type="sensor_msgs/msg/JointState",
        selector={"names": ["right_a", "right_b"]},
    )
    adapter = LiveAdapter(_make_contract(actions=[spec_left, spec_right]))

    with pytest.raises(ValueError):
        adapter.encode_action({"action": np.array([1.0, 2.0, 3.0])}, now_ns=0)


# ----------------------------------------------------------------------------
# Init-time refusals
# ----------------------------------------------------------------------------


def test_init_refuses_action_type_without_encoder():
    # Twist has a decoder but no registered encoder — the adapter must refuse
    # at boot rather than at first tick.
    spec = ActionSpec(
        key="action",
        topic="/cmd_vel",
        type="geometry_msgs/msg/Twist",
    )
    with pytest.raises(ValueError) as exc:
        LiveAdapter(_make_contract(actions=[spec]))
    assert "geometry_msgs/msg/Twist" in str(exc.value)


def test_init_refuses_action_with_transforms():
    # The live runtime can't yet apply or invert transforms, so a contract that
    # declares them on an action must refuse at boot rather than silently feed
    # the policy values the converter would have transformed.
    spec = ActionSpec(
        key="action",
        topic="/teleop/action",
        type="std_msgs/msg/Float64MultiArray",
        transforms=[TransformSpec(type="finite_difference", params={})],
    )
    with pytest.raises(ValueError) as exc:
        LiveAdapter(_make_contract(actions=[spec]))
    assert "transform" in str(exc.value).lower()


def test_init_refuses_multi_spec_without_selector_names():
    spec_a = ActionSpec(
        key="action",
        topic="/a",
        type="std_msgs/msg/Float64MultiArray",
    )
    spec_b = ActionSpec(
        key="action",
        topic="/b",
        type="std_msgs/msg/Float64MultiArray",
    )
    with pytest.raises(ValueError) as exc:
        LiveAdapter(_make_contract(actions=[spec_a, spec_b]))
    assert "selector.names" in str(exc.value)


def test_encode_action_requires_all_keys_present():
    spec = ActionSpec(
        key="action",
        topic="/teleop/action",
        type="sensor_msgs/msg/JointState",
        selector={"names": ["joint_a"]},
    )
    adapter = LiveAdapter(_make_contract(actions=[spec]))
    with pytest.raises(KeyError):
        adapter.encode_action({"wrong_key": np.array([0.1])}, now_ns=0)


def test_action_topics_property_lists_in_declaration_order():
    spec_a = ActionSpec(
        key="action",
        topic="/a",
        type="std_msgs/msg/Float64MultiArray",
        selector={"names": ["x"]},
    )
    spec_b = ActionSpec(
        key="action",
        topic="/b",
        type="std_msgs/msg/Float64MultiArray",
        selector={"names": ["y"]},
    )
    spec_c = ActionSpec(
        key="other",
        topic="/c",
        type="std_msgs/msg/Float64",
    )
    adapter = LiveAdapter(_make_contract(actions=[spec_a, spec_b, spec_c]))
    assert adapter.action_topics == ["/a", "/b", "/c"]
