"""Tests that contract transforms make ``align: linear`` interpolate rotations, angles and polar vectors correctly.

Each test converts a synthetic topic through the real ``DataCollection`` and
``generate_frames``: a 10 Hz source resampled to a 30 fps timeline.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
import pytest
from roboto_to_lerobot.contract_utils import (
    DataCollection,
    fps_to_time_step_ns,
    load_contract,
)
from roboto_to_lerobot.lerobot import generate_frames
from scipy.spatial.transform import Rotation, Slerp
from test_task_resolution import FakeTopic

_T0 = 1_700_000_000_000_000_000
_SOURCE_S = np.arange(0, 21) * 0.1  # 10 Hz for 2 s
_STEP_NS = fps_to_time_step_ns(30)
_REFERENCE_S = np.arange(0, 61) * _STEP_NS / 1e9  # 30 fps over the same 2 s


def _topic_df(columns: dict[str, np.ndarray], *, text_field: bool = True) -> pd.DataFrame:
    """Rows shaped like a fetched topic: one column per field and an int64 ``timestamp``.

    ``text_field`` adds a string field, as a ROS message's ``header.frame_id``.
    """
    df = pd.DataFrame({name: np.asarray(values, dtype=np.float64) for name, values in columns.items()})
    if text_field:
        df["frame_id"] = "base"
    df["timestamp"] = pd.Series(_T0 + np.round(_SOURCE_S * 1e9).astype(np.int64), dtype="int64")
    return df


def _convert(
    tmp_path: Path, stream: str, df: pd.DataFrame, *, section: str = "observations", key: str = "observation.state",
) -> np.ndarray:
    """Convert ``df`` with a contract whose one stream, under ``section``, is ``stream``; return each frame's ``key``.

    The frames are on the ``_REFERENCE_S`` timeline, minus any the stream has no sample for.
    """
    other = "actions" if section == "observations" else "observations"
    path = tmp_path / "contract.yaml"
    path.write_text(f"name: t\nversion: 1\nfps: 30\n{other}: []\n{section}:\n" + dedent(stream))
    contract = load_contract(path)
    topics = {"/pose": [FakeTopic(topic_id="tp_pose", name="/pose", df=df)]}
    collection = DataCollection(contract, topics, _T0 - 10**9, _T0 + 3 * 10**9)
    reference = pd.Series(_T0 + np.arange(len(_REFERENCE_S), dtype=np.int64) * _STEP_NS, name="timestamp")
    frames = list(generate_frames(contract, collection, reference))
    return np.stack([f[key] for f in frames]).astype(np.float64)


def test_quaternion_pose_follows_slerp(tmp_path):
    """Quaternions that flip sign between samples interpolate within 0.01° of SLERP, with unit
    length, while the pose's other columns interpolate linearly."""
    # Arrange: a rotation at 90°/s about a tilted axis, with every third sample stored as -q.
    rotations = Rotation.from_rotvec(np.outer(_SOURCE_S * np.pi / 2, [0.3, 0.2, 0.93]))
    quats = rotations.as_quat()
    quats[::3] *= -1
    position = np.linspace(0.0, 1.0, len(_SOURCE_S)) ** 2
    df = _topic_df({"px": position, "qx": quats[:, 0], "qy": quats[:, 1], "qz": quats[:, 2], "qw": quats[:, 3]})

    # Act
    values = _convert(tmp_path, stream="""
        - key: observation.state
          topic: /pose
          type: string_typed_msg
          selector: {names: [px, qx, qy, qz, qw]}
          align: {method: linear, tolerance_ms: 150}
          transforms:
            - {type: quaternion_continuity, stage: pre, columns: [qx, qy, qz, qw]}
            - {type: quaternion_normalize, columns: [qx, qy, qz, qw]}
    """, df=df)

    # Assert
    assert len(values) == len(_REFERENCE_S)
    expected = Slerp(_SOURCE_S, rotations)(_REFERENCE_S)
    error_deg = np.rad2deg((Rotation.from_quat(values[:, 1:]) * expected.inv()).magnitude())
    assert error_deg.max() < 0.01
    np.testing.assert_allclose(np.linalg.norm(values[:, 1:], axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(values[:, 0], np.interp(_REFERENCE_S, _SOURCE_S, position), atol=1e-6)


def test_angle_across_the_wrap_interpolates_the_short_way(tmp_path):
    """A yaw angle that wraps from +180° to -180° between samples interpolates along the short arc."""
    # Arrange: yaw turns at 90°/s through +pi and is recorded wrapped to [-pi, pi).
    true_yaw = 2.5 + _SOURCE_S * np.pi / 2
    df = _topic_df({"yaw": (true_yaw + np.pi) % (2 * np.pi) - np.pi})

    # Act
    values = _convert(tmp_path, stream="""
        - key: observation.state
          topic: /pose
          type: string_typed_msg
          selector: {names: [yaw]}
          align: {method: linear, tolerance_ms: 150}
          transforms:
            - {type: unwrap_angle, stage: pre}
            - {type: wrap_angle}
    """, df=df)

    # Assert
    expected = np.interp(_REFERENCE_S, _SOURCE_S, true_yaw)
    error = (values[:, 0] - expected + np.pi) % (2 * np.pi) - np.pi
    assert np.abs(error).max() < 1e-5
    assert values[:, 0].min() >= -np.pi and values[:, 0].max() < np.pi


def test_polar_vector_interpolates_in_cartesian(tmp_path):
    """A polar vector whose angle wraps between samples interpolates as a cartesian vector,
    keeping its direction, while the stream's other columns stay as recorded."""
    # Arrange: a unit vector turning at 90°/s through +-180°, stored as [r, theta] in degrees.
    true_theta = 150.0 + _SOURCE_S * 90.0
    depth = np.linspace(5.0, 7.0, len(_SOURCE_S))
    df = _topic_df({"depth": depth, "r": np.ones_like(_SOURCE_S), "theta": (true_theta + 180) % 360 - 180})

    # Act
    values = _convert(tmp_path, stream="""
        - key: observation.state
          topic: /pose
          type: string_typed_msg
          selector: {names: [depth, r, theta], lerobot_names: [depth, x, y]}
          align: {method: linear, tolerance_ms: 150}
          transforms:
            - {type: polar_to_cartesian, stage: pre, degrees: true, columns: [r, theta]}
    """, df=df)

    # Assert
    direction_deg = np.rad2deg(np.arctan2(values[:, 2], values[:, 1]))
    error = (direction_deg - np.interp(_REFERENCE_S, _SOURCE_S, true_theta) + 180) % 360 - 180
    assert np.abs(error).max() < 0.1
    np.testing.assert_allclose(values[:, 0], np.interp(_REFERENCE_S, _SOURCE_S, depth), atol=1e-5)


def test_post_transform_does_not_treat_the_sides_of_a_gap_as_neighbours(tmp_path):
    """A stream with a hole longer than its tolerance loses the frames in the hole, and a
    frame-to-frame difference on the rest never spans the hole."""
    # Arrange: a ramp of 1 per second, without its samples between 0.75 s and 1.35 s.
    df = _topic_df({"px": _SOURCE_S.copy()})
    df = df[(_SOURCE_S < 0.75) | (_SOURCE_S > 1.35)].reset_index(drop=True)

    # Act
    values = _convert(tmp_path, stream="""
        - key: observation.state
          topic: /pose
          type: string_typed_msg
          selector: {names: [px]}
          align: {method: linear, tolerance_ms: 150}
          transforms:
            - {type: finite_difference}
    """, df=df)

    # Assert: no frame's difference exceeds one frame's rise. Frames just past
    # a run's last sample repeat it (align: linear), so theirs is zero.
    step = _STEP_NS / 1e9
    assert 0 < len(values) < len(_REFERENCE_S)
    assert values[:, 0].max() < step * 1.001
    assert np.median(values[:, 0]) == pytest.approx(step, rel=1e-4)


@pytest.mark.parametrize(
    ("transform", "expected"),
    [
        pytest.param("{type: finite_difference}", 0.0, id="finite_difference"),
        pytest.param("{type: butterworth_lowpass, cutoff_hz: 5}", 4.0, id="butterworth_lowpass"),
    ],
)
def test_post_transform_converts_a_short_run_between_gaps(tmp_path, transform, expected):
    """A few frames left between two holes in a stream are converted, however short the run."""
    # Arrange: a constant stream with samples only at 0-0.5 s, 1.0 s and 1.5-2 s.
    keep = (_SOURCE_S < 0.55) | np.isclose(_SOURCE_S, 1.0) | (_SOURCE_S > 1.45)
    df = _topic_df({"px": np.full(len(_SOURCE_S), 4.0)})[keep].reset_index(drop=True)

    # Act
    values = _convert(tmp_path, stream=f"""
        - key: observation.state
          topic: /pose
          type: string_typed_msg
          selector: {{names: [px]}}
          align: {{method: hold, tolerance_ms: 40}}
          transforms:
            - {transform}
    """, df=df)

    # Assert
    assert 0 < len(values) < len(_REFERENCE_S)
    np.testing.assert_allclose(values[:, 0], expected, atol=1e-5)


@pytest.mark.parametrize(("section", "key"), [("observations", "observation.state"), ("actions", "action")])
def test_topic_with_only_numeric_fields_aligns(tmp_path, section, key):
    """A topic whose fields are all numbers aligns onto the timeline like any other, as an
    observation or as an action."""
    # Arrange
    position = np.linspace(0.0, 2.0, len(_SOURCE_S))
    df = _topic_df({"px": position}, text_field=False)

    # Act
    values = _convert(tmp_path, stream=f"""
        - key: {key}
          topic: /pose
          type: string_typed_msg
          selector: {{names: [px]}}
          align: {{method: linear, tolerance_ms: 150}}
    """, df=df, section=section, key=key)

    # Assert
    assert len(values) == len(_REFERENCE_S)
    np.testing.assert_allclose(values[:, 0], np.interp(_REFERENCE_S, _SOURCE_S, position), atol=1e-6)
