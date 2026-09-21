"""Contract-loading tests for the hub-launch breaking-change batch.

Covers three schema changes, all enforced at :func:`load_contract` time:

1. ``tolerance_ms`` semantics unified with the live runtime's bounded
   default (see ``AlignSpec`` / ``_as_align`` in ``contract_utils``):
   omitted -> auto-bounded default, ``null`` -> unlimited, ``0`` -> error.
2. Legacy contract forms (``strategy``/``tol_ms`` align aliases, the
   nested ``publish:`` action form) are rejected outright.
3. An ``image.depth:`` block is rejected at load time — depth images are
   not supported yet, and the error points at the repo for anyone who
   wants to contribute support.

Plus the error-message quality pass: duplicate ``lerobot_names`` name both
conflicting specs, and a missing/wrong-typed field in one spec raises a
``ValueError`` naming that spec's index and key instead of a bare
``KeyError``/``TypeError``.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest
from roboto_to_lerobot.contract_utils import AlignSpec, load_contract


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "contract.yaml"
    path.write_text(dedent(body))
    return path


# ---------------------------------------------------------------------------
# AlignSpec dataclass guard (direct construction, bypassing load_contract)
# ---------------------------------------------------------------------------


def test_align_spec_allows_none_tolerance():
    spec = AlignSpec(method="hold", tolerance_ms=None)
    assert spec.tolerance_ms is None


def test_align_spec_rejects_zero_tolerance():
    with pytest.raises(ValueError, match="tolerance_ms"):
        AlignSpec(method="hold", tolerance_ms=0)


def test_align_spec_rejects_negative_tolerance():
    with pytest.raises(ValueError, match="tolerance_ms"):
        AlignSpec(method="hold", tolerance_ms=-5)


# ---------------------------------------------------------------------------
# tolerance_ms semantics at contract-load time
# ---------------------------------------------------------------------------


def test_align_block_omitted_uses_auto_bounded_default(tmp_path):
    """No ``align:`` at all -> max(2/fps, 50ms), same formula the live
    runtime uses to bound an unlimited tolerance."""
    path = _write(tmp_path, """
        name: c
        version: 1
        fps: 20
        observations:
          - key: observation.state
            topic: /robot/joint_states
            type: sensor_msgs/msg/JointState
            selector: {names: [j1]}
        actions: []
    """)
    contract = load_contract(path)
    align = contract.observations[0].align
    assert align.method == "hold"
    # 2/20 = 100ms > 50ms floor
    assert align.tolerance_ms == pytest.approx(100.0)


def test_align_block_present_without_tolerance_ms_uses_auto_bounded_default(tmp_path):
    """``align: {method: nearest}`` with no ``tolerance_ms`` still gets the
    bounded default, not unlimited."""
    path = _write(tmp_path, """
        name: c
        version: 1
        fps: 100
        observations:
          - key: observation.state
            topic: /robot/joint_states
            type: sensor_msgs/msg/JointState
            selector: {names: [j1]}
            align: {method: nearest}
        actions: []
    """)
    contract = load_contract(path)
    align = contract.observations[0].align
    assert align.method == "nearest"
    # 2/100 = 20ms < 50ms floor -> floor wins
    assert align.tolerance_ms == pytest.approx(50.0)


def test_tolerance_ms_null_is_unlimited(tmp_path):
    path = _write(tmp_path, """
        name: c
        version: 1
        fps: 20
        observations:
          - key: observation.state
            topic: /robot/joint_states
            type: sensor_msgs/msg/JointState
            selector: {names: [j1]}
            align: {method: hold, tolerance_ms: null}
        actions: []
    """)
    contract = load_contract(path)
    assert contract.observations[0].align.tolerance_ms is None


def test_tolerance_ms_zero_raises(tmp_path):
    path = _write(tmp_path, """
        name: c
        version: 1
        fps: 20
        observations:
          - key: observation.state
            topic: /robot/joint_states
            type: sensor_msgs/msg/JointState
            selector: {names: [j1]}
            align: {method: hold, tolerance_ms: 0}
        actions: []
    """)
    with pytest.raises(ValueError, match="tolerance_ms: null"):
        load_contract(path)


def test_tolerance_ms_positive_value_unchanged(tmp_path):
    path = _write(tmp_path, """
        name: c
        version: 1
        fps: 20
        observations:
          - key: observation.state
            topic: /robot/joint_states
            type: sensor_msgs/msg/JointState
            selector: {names: [j1]}
            align: {method: hold, tolerance_ms: 250}
        actions: []
    """)
    contract = load_contract(path)
    assert contract.observations[0].align.tolerance_ms == 250.0


# ---------------------------------------------------------------------------
# Legacy forms are rejected outright
# ---------------------------------------------------------------------------


def test_legacy_strategy_key_raises(tmp_path):
    path = _write(tmp_path, """
        name: c
        version: 1
        fps: 20
        observations:
          - key: observation.state
            topic: /robot/joint_states
            type: sensor_msgs/msg/JointState
            selector: {names: [j1]}
            align: {strategy: hold, tolerance_ms: 100}
        actions: []
    """)
    with pytest.raises(ValueError, match="strategy"):
        load_contract(path)


def test_legacy_tol_ms_key_raises(tmp_path):
    path = _write(tmp_path, """
        name: c
        version: 1
        fps: 20
        observations:
          - key: observation.state
            topic: /robot/joint_states
            type: sensor_msgs/msg/JointState
            selector: {names: [j1]}
            align: {method: hold, tol_ms: 100}
        actions: []
    """)
    with pytest.raises(ValueError, match="tol_ms"):
        load_contract(path)


def test_legacy_publish_block_raises(tmp_path):
    path = _write(tmp_path, """
        name: c
        version: 1
        fps: 20
        observations: []
        actions:
          - key: action
            publish:
              topic: /robot/joint_commands
              type: sensor_msgs/msg/JointState
    """)
    with pytest.raises(ValueError, match="publish"):
        load_contract(path)


def test_legacy_publish_with_role_no_longer_bypasses_exclusivity_check(tmp_path):
    """Regression: ``publish:`` + ``role:`` used to silently skip the
    topic/role exclusivity check because ``publish:`` short-circuited
    before ``_topic_or_role`` ran. Now ``publish:`` is rejected outright,
    so this combination surfaces the (now single) error path."""
    path = _write(tmp_path, """
        name: c
        version: 1
        fps: 20
        observations: []
        actions:
          - key: action
            role: left_arm
            publish:
              topic: /robot/joint_commands
              type: sensor_msgs/msg/JointState
    """)
    with pytest.raises(ValueError, match="publish"):
        load_contract(path)


# ---------------------------------------------------------------------------
# image.depth: block is rejected at load time
# ---------------------------------------------------------------------------


def test_image_depth_block_raises(tmp_path):
    path = _write(tmp_path, """
        name: c
        version: 1
        fps: 20
        observations:
          - key: observation.images.depth
            topic: /camera/depth/image_raw
            type: sensor_msgs/msg/Image
            image:
              resize: [480, 640]
              depth: {range: [0.1, 5.0]}
        actions: []
    """)
    with pytest.raises(ValueError, match="depth"):
        load_contract(path)


def test_image_without_depth_block_still_loads(tmp_path):
    """Non-depth image specs are unaffected by the depth rejection."""
    path = _write(tmp_path, """
        name: c
        version: 1
        fps: 20
        observations:
          - key: observation.images.exo
            topic: /camera/exo/image_raw/compressed
            type: sensor_msgs/msg/CompressedImage
            image:
              resize: [480, 640]
        actions: []
    """)
    contract = load_contract(path)
    assert contract.videos[0].image == {"resize": [480, 640]}


# ---------------------------------------------------------------------------
# Error-message quality: duplicate lerobot_names names both specs
# ---------------------------------------------------------------------------


def test_duplicate_lerobot_names_names_both_conflicting_specs(tmp_path):
    path = _write(tmp_path, """
        name: c
        version: 1
        fps: 20
        observations:
          - key: observation.state
            topic: /arm_a/state
            type: sensor_msgs/msg/JointState
            selector: {names: [j1], lerobot_names: [shared_name]}
          - key: observation.other
            topic: /arm_b/state
            type: sensor_msgs/msg/JointState
            selector: {names: [j1], lerobot_names: [shared_name]}
        actions: []
    """)
    with pytest.raises(ValueError) as exc:
        load_contract(path)
    message = str(exc.value)
    assert "shared_name" in message
    # Both conflicting specs must be named: key and topic/role for each.
    assert "observation.state" in message
    assert "observation.other" in message
    assert "/arm_a/state" in message
    assert "/arm_b/state" in message


def test_lerobot_names_length_mismatch_prints_both_lists(tmp_path):
    path = _write(tmp_path, """
        name: c
        version: 1
        fps: 20
        observations:
          - key: observation.state
            topic: /robot/joint_states
            type: sensor_msgs/msg/JointState
            selector: {names: [j1, j2], lerobot_names: [only_one]}
        actions: []
    """)
    with pytest.raises(ValueError) as exc:
        load_contract(path)
    message = str(exc.value)
    assert "['j1', 'j2']" in message
    assert "['only_one']" in message


# ---------------------------------------------------------------------------
# Missing/wrong-typed field in one spec -> indexed ValueError, not a bare
# KeyError/TypeError.
# ---------------------------------------------------------------------------


def test_missing_key_field_in_observation_raises_indexed_valueerror(tmp_path):
    path = _write(tmp_path, """
        name: c
        version: 1
        fps: 20
        observations:
          - key: observation.state
            topic: /robot/joint_states
            type: sensor_msgs/msg/JointState
            selector: {names: [j1]}
          - topic: /robot/joint_states_2
            type: sensor_msgs/msg/JointState
        actions: []
    """)
    with pytest.raises(ValueError) as exc:
        load_contract(path)
    message = str(exc.value)
    assert "observations[1]" in message


def test_missing_type_field_in_action_raises_indexed_valueerror_with_key(tmp_path):
    path = _write(tmp_path, """
        name: c
        version: 1
        fps: 20
        observations: []
        actions:
          - key: action
            topic: /robot/joint_commands
            selector: {names: [j1]}
    """)
    with pytest.raises(ValueError) as exc:
        load_contract(path)
    message = str(exc.value)
    assert "actions[0]" in message
    assert "action" in message
