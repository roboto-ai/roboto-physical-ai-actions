"""Schema tests for the two codegen-only contract fields.

The contract carries a per-stream ``qos:`` block and a per-action
``safety_behavior:`` field to the contract. The converter never reads
either field — both are consumed only by codegen — so the schema-level
contract is just "parse, validate, default; round-trip an existing
contract unchanged."
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest
from roboto_to_lerobot.contract_utils import (
    QosSpec,
    load_contract,
)


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "contract.yaml"
    path.write_text(dedent(body))
    return path


# ---------------------------------------------------------------------------
# QosSpec dataclass guards
# ---------------------------------------------------------------------------


def test_qos_spec_default_is_best_effort_keep_last_1():
    spec = QosSpec()
    assert (spec.reliability, spec.history, spec.depth) == (
        "BEST_EFFORT", "KEEP_LAST", 1,
    )


def test_qos_spec_rejects_unknown_reliability():
    with pytest.raises(ValueError, match="reliability"):
        QosSpec(reliability="MAYBE")


def test_qos_spec_rejects_unknown_history():
    with pytest.raises(ValueError, match="history"):
        QosSpec(history="KEEP_SOME")


def test_qos_spec_rejects_negative_depth():
    with pytest.raises(ValueError, match="depth"):
        QosSpec(depth=-1)


def test_qos_spec_rejects_zero_depth_for_keep_last():
    # A KEEP_LAST queue of depth 0 is a zero-capacity queue DDS silently
    # drops every message into — the exact failure the schema guards.
    with pytest.raises(ValueError, match="depth"):
        QosSpec(history="KEEP_LAST", depth=0)


def test_qos_spec_allows_zero_depth_for_keep_all():
    # KEEP_ALL ignores depth, so depth 0 is harmless there.
    spec = QosSpec(history="KEEP_ALL", depth=0)
    assert spec.depth == 0


# ---------------------------------------------------------------------------
# Contract-level: existing contracts load unchanged (backward compat)
# ---------------------------------------------------------------------------


def test_existing_contract_loads_without_qos_or_safety_behavior(tmp_path):
    """Older contracts must load identically — qos None, default safety."""
    path = _write(tmp_path, """
        name: legacy
        version: 1
        fps: 30
        observations:
          - key: observation.state
            topic: /robot/joint_states
            type: sensor_msgs/msg/JointState
            selector: {names: [j1, j2]}
        actions:
          - key: action
            topic: /teleop/action
            type: sensor_msgs/msg/JointState
            selector: {names: [j1, j2]}
    """)
    contract = load_contract(path)

    assert contract.observations[0].qos is None
    assert contract.actions[0].qos is None
    assert contract.actions[0].safety_behavior == "publish_nothing"


# ---------------------------------------------------------------------------
# qos: explicit parsing
# ---------------------------------------------------------------------------


def test_qos_block_on_observation_parses(tmp_path):
    path = _write(tmp_path, """
        name: c
        version: 1
        fps: 30
        observations:
          - key: observation.state
            topic: /robot/joint_states
            type: sensor_msgs/msg/JointState
            selector: {names: [j1]}
            qos: {reliability: RELIABLE, history: KEEP_LAST, depth: 10}
        actions: []
    """)
    contract = load_contract(path)

    assert contract.observations[0].qos == QosSpec(
        reliability="RELIABLE", history="KEEP_LAST", depth=10,
    )


def test_qos_block_on_action_parses(tmp_path):
    path = _write(tmp_path, """
        name: c
        version: 1
        fps: 30
        observations: []
        actions:
          - key: action
            topic: /teleop/action
            type: sensor_msgs/msg/JointState
            selector: {names: [j1]}
            qos: {reliability: RELIABLE, depth: 5}
    """)
    contract = load_contract(path)

    # depth filled, history defaulted, reliability honored.
    assert contract.actions[0].qos == QosSpec(
        reliability="RELIABLE", history="KEEP_LAST", depth=5,
    )


def test_qos_accepts_lowercase_yaml_spelling(tmp_path):
    """YAML authors commonly write enum values in lowercase; normalize at parse."""
    path = _write(tmp_path, """
        name: c
        version: 1
        fps: 30
        observations:
          - key: observation.state
            topic: /robot/joint_states
            type: sensor_msgs/msg/JointState
            selector: {names: [j1]}
            qos: {reliability: reliable, history: keep_all, depth: 0}
        actions: []
    """)
    contract = load_contract(path)
    assert contract.observations[0].qos == QosSpec(
        reliability="RELIABLE", history="KEEP_ALL", depth=0,
    )


def test_qos_invalid_value_surfaces_at_load_time(tmp_path):
    path = _write(tmp_path, """
        name: c
        version: 1
        fps: 30
        observations:
          - key: observation.state
            topic: /robot/joint_states
            type: sensor_msgs/msg/JointState
            selector: {names: [j1]}
            qos: {reliability: MAYBE}
        actions: []
    """)
    with pytest.raises(ValueError, match="reliability"):
        load_contract(path)


# ---------------------------------------------------------------------------
# safety_behavior
# ---------------------------------------------------------------------------


def test_safety_behavior_explicit_value_passes_through(tmp_path):
    """Schema does not reject unsupported modes — codegen handles refusals later."""
    path = _write(tmp_path, """
        name: c
        version: 1
        fps: 30
        observations: []
        actions:
          - key: action
            topic: /teleop/action
            type: sensor_msgs/msg/JointState
            selector: {names: [j1]}
            safety_behavior: hold_last
    """)
    contract = load_contract(path)
    assert contract.actions[0].safety_behavior == "hold_last"
