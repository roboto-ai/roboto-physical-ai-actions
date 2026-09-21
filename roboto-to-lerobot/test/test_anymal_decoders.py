"""Unit tests for the ANYmal quadruped decoders and the ``_dot_get`` fix.

These cover the two custom ANYmal message types added for legged-robot
datasets — ``anymal_msgs/AnymalState`` (leg joint proprioception → state) and
``series_elastic_actuator_msgs/SeActuatorReadings`` (``commanded.position`` →
action) — plus the ``_dot_get`` fix that lets the stock ``sensor_msgs/msg/Imu``
decoder read dotted selector paths off an offline pandas-Series row.

Offline rows are pandas Series whose nested fields Roboto has flattened into
dot-notation columns (parallel-array joints) or lists of dicts; the decoders
also accept attribute-style objects for the live rclpy path. Both shapes are
exercised here without any network access.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import roboto_to_lerobot.runtime.decoders  # noqa: F401  (registers decoders)
from roboto_to_lerobot.contract_utils import ActionSpec, ObservationSpec
from roboto_to_lerobot.runtime.converters import decode_value

# Canonical ANYmal leg-joint order: <LF|RF|LH|RH>_<HAA|HFE|KFE>.
JOINTS = [
    "LF_HAA", "LF_HFE", "LF_KFE",
    "RF_HAA", "RF_HFE", "RF_KFE",
    "LH_HAA", "LH_HFE", "LH_KFE",
    "RH_HAA", "RH_HFE", "RH_KFE",
]


# ── fixtures ────────────────────────────────────────────────────────────────

def _anymal_state_row() -> pd.Series:
    """One AnymalState message as Roboto's flattened offline row."""
    return pd.Series(
        {
            "joints.name": list(JOINTS),
            "joints.position": tuple(float(i) for i in range(12)),
            "joints.velocity": tuple(float(i) + 100.0 for i in range(12)),
            "joints.effort": tuple(float(i) + 200.0 for i in range(12)),
            "joints.acceleration": tuple(float(i) + 300.0 for i in range(12)),
        }
    )


def _actuator_readings_row() -> pd.Series:
    """One SeActuatorReadings message: 12 positional readings, dict-shaped."""
    readings = [
        {
            "commanded": {"position": float(i), "velocity": float(i) + 10.0},
            "state": {"joint_position": float(i) + 20.0},
        }
        for i in range(12)
    ]
    return pd.Series({"readings": readings})


def _imu_row() -> pd.Series:
    """One Imu message as Roboto's flattened offline row (dotted columns)."""
    return pd.Series(
        {
            "orientation.x": 0.0, "orientation.y": 0.0,
            "orientation.z": 0.0, "orientation.w": 1.0,
            "angular_velocity.x": 1.0, "angular_velocity.y": 2.0,
            "angular_velocity.z": 3.0,
            "linear_acceleration.x": 4.0, "linear_acceleration.y": 5.0,
            "linear_acceleration.z": 6.0,
        }
    )


# ── sensor_msgs/msg/Imu (the _dot_get fix) ─────────────────────────────────

def test_imu_dotted_selectors_on_series_row():
    """Regression: dotted selector paths must resolve against a Series row.

    Before the ``_dot_get`` fix this raised ``AttributeError`` because the
    helper walked attributes on a Series whose columns are dotted labels.
    """
    spec = ObservationSpec(
        key="observation.state", topic="/anymal/imu", type="sensor_msgs/msg/Imu",
        selector={"names": [
            "angular_velocity.x", "angular_velocity.y", "angular_velocity.z",
            "linear_acceleration.x", "linear_acceleration.y", "linear_acceleration.z",
        ]},
    )
    out = decode_value(_imu_row(), spec)
    np.testing.assert_array_equal(out, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


def test_imu_dotted_selectors_on_live_object():
    """The same decoder still walks attributes for a live ROS message object."""
    msg = SimpleNamespace(
        angular_velocity=SimpleNamespace(x=1.0, y=2.0, z=3.0),
        linear_acceleration=SimpleNamespace(x=4.0, y=5.0, z=6.0),
    )
    spec = ObservationSpec(
        key="observation.state", topic="/anymal/imu", type="sensor_msgs/msg/Imu",
        selector={"names": ["angular_velocity.z", "linear_acceleration.x"]},
    )
    np.testing.assert_array_equal(decode_value(msg, spec), [3.0, 4.0])


# ── anymal_msgs/AnymalState ────────────────────────────────────────────────

def test_anymal_state_all_joint_positions_in_order():
    spec = ObservationSpec(
        key="observation.state", topic="/anymal/state_estimator/anymal_state",
        type="anymal_msgs/AnymalState",
        selector={"names": list(JOINTS)},
    )
    out = decode_value(_anymal_state_row(), spec)
    np.testing.assert_array_equal(out, [float(i) for i in range(12)])
    assert out.dtype == np.float64


def test_anymal_state_no_selector_returns_all_positions():
    spec = ObservationSpec(
        key="observation.state", topic="/t", type="anymal_msgs/AnymalState",
    )
    out = decode_value(_anymal_state_row(), spec)
    np.testing.assert_array_equal(out, [float(i) for i in range(12)])


def test_anymal_state_field_prefix_selects_velocity():
    spec = ObservationSpec(
        key="observation.state", topic="/t", type="anymal_msgs/AnymalState",
        selector={"names": ["velocity.LF_HAA", "LF_HFE"]},  # 2nd is bare -> position
    )
    # LF_HAA velocity is index 0 + 100; LF_HFE position is index 1.
    np.testing.assert_array_equal(decode_value(_anymal_state_row(), spec), [100.0, 1.0])


def test_anymal_state_unknown_joint_raises():
    spec = ObservationSpec(
        key="o", topic="/t", type="anymal_msgs/AnymalState",
        selector={"names": ["NOPE_HAA"]},
    )
    with pytest.raises(ValueError, match="Joint 'NOPE_HAA' not in message"):
        decode_value(_anymal_state_row(), spec)


def test_anymal_state_unknown_field_raises():
    spec = ObservationSpec(
        key="o", topic="/t", type="anymal_msgs/AnymalState",
        selector={"names": ["torque.LF_HAA"]},
    )
    with pytest.raises(ValueError, match="Unknown AnymalState joint field 'torque'"):
        decode_value(_anymal_state_row(), spec)


# ── series_elastic_actuator_msgs/SeActuatorReadings ────────────────────────

def test_actuator_bare_names_default_to_commanded_position():
    spec = ActionSpec(
        key="action", topic="/anymal/actuator_readings",
        type="series_elastic_actuator_msgs/SeActuatorReadings",
        selector={"names": list(JOINTS)},
    )
    out = decode_value(_actuator_readings_row(), spec)
    np.testing.assert_array_equal(out, [float(i) for i in range(12)])
    assert out.dtype == np.float64


def test_actuator_no_selector_returns_all_commanded_positions():
    spec = ActionSpec(
        key="action", topic="/t",
        type="series_elastic_actuator_msgs/SeActuatorReadings",
    )
    out = decode_value(_actuator_readings_row(), spec)
    np.testing.assert_array_equal(out, [float(i) for i in range(12)])


def test_actuator_explicit_field_paths():
    spec = ActionSpec(
        key="action", topic="/t",
        type="series_elastic_actuator_msgs/SeActuatorReadings",
        selector={"names": [
            "commanded.velocity.RF_HAA",     # index 3 -> 3 + 10
            "state.joint_position.RF_HAA",   # index 3 -> 3 + 20
        ]},
    )
    np.testing.assert_array_equal(decode_value(_actuator_readings_row(), spec), [13.0, 23.0])


def test_actuator_wrong_reading_count_raises():
    row = pd.Series({"readings": [{"commanded": {"position": 0.0}}]})  # only 1, expect 12
    spec = ActionSpec(
        key="action", topic="/t",
        type="series_elastic_actuator_msgs/SeActuatorReadings",
    )
    with pytest.raises(ValueError, match="Expected 12 actuator readings"):
        decode_value(row, spec)


def test_actuator_unknown_joint_raises():
    spec = ActionSpec(
        key="action", topic="/t",
        type="series_elastic_actuator_msgs/SeActuatorReadings",
        selector={"names": ["NOPE_HAA"]},
    )
    with pytest.raises(ValueError, match="Joint 'NOPE_HAA' not in ANYmal actuator order"):
        decode_value(_actuator_readings_row(), spec)


def test_actuator_live_object_path():
    """Readings as attribute objects (live rclpy path) resolve identically."""
    readings = [
        SimpleNamespace(commanded=SimpleNamespace(position=float(i)))
        for i in range(12)
    ]
    msg = SimpleNamespace(readings=readings)
    spec = ActionSpec(
        key="action", topic="/t",
        type="series_elastic_actuator_msgs/SeActuatorReadings",
        selector={"names": list(JOINTS)},
    )
    np.testing.assert_array_equal(decode_value(msg, spec), [float(i) for i in range(12)])
