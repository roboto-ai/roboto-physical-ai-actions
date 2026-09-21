"""Tests for ``codegen.cli`` — argparse surface + codegen-time refusals.

The CLI ships the executable surface and the validation pass that
rejects features the live runtime cannot honor; these tests assert
that refusals fire before any file is written.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest
from roboto_to_lerobot.codegen.cli import (
    REFUSED_ALIGN_METHODS,
    REFUSED_TRANSFORMS,
    GenNodeError,
    build_parser,
    main,
    validate_contract_for_codegen,
)
from roboto_to_lerobot.runtime.contract_io import load_contract

_OBS_BLOCK = """
        observations:
          - key: observation.state
            topic: /robot/joint_states
            type: sensor_msgs/msg/JointState
            selector: {names: [j1, j2]}
"""

_ACTION_BLOCK = """
        actions:
          - key: action
            topic: /teleop/action
            type: sensor_msgs/msg/JointState
            selector: {names: [j1, j2]}
"""


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "contract.yaml"
    path.write_text(dedent(body))
    return path


# ---------------------------------------------------------------------------
# Refuse lists
# ---------------------------------------------------------------------------


def test_refuse_lists_match_handoff():
    assert REFUSED_TRANSFORMS == frozenset({
        "butterworth_lowpass", "resample_uniform",
    })
    assert REFUSED_ALIGN_METHODS == frozenset({"linear", "none"})


# ---------------------------------------------------------------------------
# validate_contract_for_codegen
# ---------------------------------------------------------------------------


def test_valid_contract_passes_validation(tmp_path):
    path = _write(tmp_path, f"""
        name: ok
        version: 1
        fps: 30
        {_OBS_BLOCK}
        {_ACTION_BLOCK}
    """)
    contract = load_contract(path)
    validate_contract_for_codegen(contract)  # does not raise


def test_linear_align_refused_with_clear_error(tmp_path):
    """Linear interpolation is interpolating — not realizable in a live node."""
    path = _write(tmp_path, """
        name: c
        version: 1
        fps: 30
        observations:
          - key: observation.state
            topic: /robot/joint_states
            type: sensor_msgs/msg/JointState
            selector: {names: [j1]}
            align: {method: linear, tolerance_ms: 100}
        actions: []
    """)
    contract = load_contract(path)
    with pytest.raises(GenNodeError, match="linear"):
        validate_contract_for_codegen(contract)


def test_none_align_refused(tmp_path):
    path = _write(tmp_path, f"""
        name: c
        version: 1
        fps: 30
        observations:
          - key: observation.state
            topic: /robot/joint_states
            type: sensor_msgs/msg/JointState
            selector: {{names: [j1]}}
            align: {{method: none}}
        {_ACTION_BLOCK}
    """)
    contract = load_contract(path)
    with pytest.raises(GenNodeError, match="none"):
        validate_contract_for_codegen(contract)


def test_butterworth_lowpass_transform_refused(tmp_path):
    path = _write(tmp_path, f"""
        name: c
        version: 1
        fps: 30
        {_OBS_BLOCK}
        actions:
          - key: action
            topic: /teleop/action
            type: sensor_msgs/msg/JointState
            selector: {{names: [j1, j2]}}
            transforms:
              - type: butterworth_lowpass
                stage: pre
                cutoff_hz: 20
                order: 2
    """)
    contract = load_contract(path)
    with pytest.raises(GenNodeError, match="butterworth_lowpass"):
        validate_contract_for_codegen(contract)


def test_resample_uniform_transform_refused(tmp_path):
    path = _write(tmp_path, f"""
        name: c
        version: 1
        fps: 30
        {_OBS_BLOCK}
        actions:
          - key: action
            topic: /teleop/action
            type: sensor_msgs/msg/JointState
            selector: {{names: [j1, j2]}}
            transforms:
              - type: resample_uniform
                stage: pre
    """)
    contract = load_contract(path)
    with pytest.raises(GenNodeError, match="resample_uniform"):
        validate_contract_for_codegen(contract)


def test_unsupported_safety_behavior_refused(tmp_path):
    """hold_last and safe_pose are not implemented; refuse cleanly."""
    path = _write(tmp_path, f"""
        name: c
        version: 1
        fps: 30
        {_OBS_BLOCK}
        actions:
          - key: action
            topic: /teleop/action
            type: sensor_msgs/msg/JointState
            selector: {{names: [j1, j2]}}
            safety_behavior: hold_last
    """)
    contract = load_contract(path)
    with pytest.raises(GenNodeError, match="hold_last"):
        validate_contract_for_codegen(contract)


def test_out_of_enum_safety_behavior_rejected_at_load(tmp_path):
    """A safety_behavior outside VALID_SAFETY_BEHAVIORS is a schema error.

    Caught at load time (not deferred to codegen) so a typo surfaces as a
    clear schema message rather than the generic gen-node refusal.
    """
    path = _write(tmp_path, f"""
        name: c
        version: 1
        fps: 30
        {_OBS_BLOCK}
        actions:
          - key: action
            topic: /teleop/action
            type: sensor_msgs/msg/JointState
            selector: {{names: [j1, j2]}}
            safety_behavior: hold_lastt
    """)
    with pytest.raises(ValueError, match="hold_lastt"):
        load_contract(path)


def test_action_align_method_not_refused(tmp_path):
    """The runtime never aligns actions, so a refused align.method on an
    action must NOT be refused at codegen (would over-refuse a contract
    the live runtime accepts). Regression guard for the action-side
    align-check removal."""
    path = _write(tmp_path, f"""
        name: c
        version: 1
        fps: 30
        {_OBS_BLOCK}
        actions:
          - key: action
            topic: /teleop/action
            type: sensor_msgs/msg/JointState
            selector: {{names: [j1, j2]}}
            align: {{method: linear, tolerance_ms: 100}}
    """)
    contract = load_contract(path)
    validate_contract_for_codegen(contract)  # does not raise


def test_non_listed_transform_refused(tmp_path):
    """A transform outside both the causal allow-list and the non-causal
    named set (e.g. finite_difference) is still refused with the generic
    message. Mirrors LiveAdapter._validate_transforms."""
    path = _write(tmp_path, f"""
        name: c
        version: 1
        fps: 30
        observations:
          - key: observation.state
            topic: /robot/joint_states
            type: sensor_msgs/msg/JointState
            selector: {{names: [j1, j2]}}
            transforms:
              - type: finite_difference
                stage: pre
        {_ACTION_BLOCK}
    """)
    contract = load_contract(path)
    with pytest.raises(GenNodeError, match="finite_difference"):
        validate_contract_for_codegen(contract)


def test_observation_without_decoder_refused(tmp_path):
    """An observation type with no registered decoder is refused at codegen,
    mirroring LiveAdapter._make_buffer_entry's boot-time gate."""
    path = _write(tmp_path, f"""
        name: c
        version: 1
        fps: 30
        observations:
          - key: observation.temp
            topic: /sensor/temp
            type: sensor_msgs/msg/Temperature
        {_ACTION_BLOCK}
    """)
    contract = load_contract(path)
    with pytest.raises(GenNodeError, match="no decoder registered"):
        validate_contract_for_codegen(contract)


def test_causal_lowpass_transform_on_observation_is_allowed(tmp_path):
    """butterworth_lowpass_causal is the one transform gen-node can run
    live; a stage:pre instance with fs_hz declared must pass validation."""
    path = _write(tmp_path, f"""
        name: c
        version: 1
        fps: 30
        observations:
          - key: observation.state
            topic: /robot/joint_states
            type: sensor_msgs/msg/JointState
            selector: {{names: [j1, j2]}}
            transforms:
              - type: butterworth_lowpass_causal
                stage: pre
                cutoff_hz: 5.0
                fs_hz: 100.0
        {_ACTION_BLOCK}
    """)
    contract = load_contract(path)
    validate_contract_for_codegen(contract)  # does not raise


def test_causal_lowpass_pre_stage_without_fs_hz_refused(tmp_path):
    path = _write(tmp_path, f"""
        name: c
        version: 1
        fps: 30
        observations:
          - key: observation.state
            topic: /robot/joint_states
            type: sensor_msgs/msg/JointState
            selector: {{names: [j1, j2]}}
            transforms:
              - type: butterworth_lowpass_causal
                stage: pre
                cutoff_hz: 5.0
        {_ACTION_BLOCK}
    """)
    contract = load_contract(path)
    with pytest.raises(GenNodeError, match="fs_hz"):
        validate_contract_for_codegen(contract)


def test_causal_lowpass_post_stage_matching_fps_is_allowed(tmp_path):
    path = _write(tmp_path, f"""
        name: c
        version: 1
        fps: 30
        observations:
          - key: observation.state
            topic: /robot/joint_states
            type: sensor_msgs/msg/JointState
            selector: {{names: [j1, j2]}}
            transforms:
              - type: butterworth_lowpass_causal
                stage: post
                cutoff_hz: 5.0
                fs_hz: 30.0
        {_ACTION_BLOCK}
    """)
    contract = load_contract(path)
    validate_contract_for_codegen(contract)  # does not raise


def test_causal_lowpass_post_stage_contradicting_fps_refused(tmp_path):
    path = _write(tmp_path, f"""
        name: c
        version: 1
        fps: 30
        observations:
          - key: observation.state
            topic: /robot/joint_states
            type: sensor_msgs/msg/JointState
            selector: {{names: [j1, j2]}}
            transforms:
              - type: butterworth_lowpass_causal
                stage: post
                cutoff_hz: 5.0
                fs_hz: 60.0
        {_ACTION_BLOCK}
    """)
    contract = load_contract(path)
    with pytest.raises(GenNodeError, match="contradicts"):
        validate_contract_for_codegen(contract)


def test_causal_lowpass_transform_on_action_allowed_without_fs_hz(tmp_path):
    """Actions accept butterworth_lowpass_causal unconditionally — the
    runtime passes the policy's action output through unfiltered, so
    there's no fs_hz to design a live filter for."""
    path = _write(tmp_path, f"""
        name: c
        version: 1
        fps: 30
        {_OBS_BLOCK}
        actions:
          - key: action
            topic: /teleop/action
            type: sensor_msgs/msg/JointState
            selector: {{names: [j1, j2]}}
            transforms:
              - type: butterworth_lowpass_causal
                stage: pre
                cutoff_hz: 5.0
    """)
    contract = load_contract(path)
    validate_contract_for_codegen(contract)  # does not raise


def test_causal_lowpass_transform_on_video_refused(tmp_path):
    """Videos never accept transforms, causal or not."""
    path = _write(tmp_path, f"""
        name: c
        version: 1
        fps: 30
        observations:
          - key: observation.cam
            topic: /camera/compressed
            type: sensor_msgs/msg/CompressedImage
            image: {{resize: [64, 64]}}
            transforms:
              - type: butterworth_lowpass_causal
                stage: post
                cutoff_hz: 5.0
        {_ACTION_BLOCK}
    """)
    contract = load_contract(path)
    with pytest.raises(GenNodeError, match="video"):
        validate_contract_for_codegen(contract)


def test_file_backed_observation_type_refused(tmp_path):
    """Bare file-backed/offline-only types (no 'pkg/msg/Type' shape) have no
    live ROS message class — refuse rather than emit `from builtins import video`."""
    path = _write(tmp_path, f"""
        name: c
        version: 1
        fps: 30
        observations:
          - key: observation.cam
            topic: /camera/frames
            type: video
        {_ACTION_BLOCK}
    """)
    contract = load_contract(path)
    with pytest.raises(GenNodeError, match="file-backed"):
        validate_contract_for_codegen(contract)


# ---------------------------------------------------------------------------
# CLI surface — argparse + entry-point behaviour
# ---------------------------------------------------------------------------


def _required_argv(contract: Path, out: Path, manifest: Path) -> list[str]:
    return [
        "gen-node", str(contract),
        "--policy-module", "my_pkg.policies.act",
        "--manifest", str(manifest),
        "--out", str(out),
    ]


def test_parser_requires_subcommand(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_parser_requires_required_flags(capsys, tmp_path):
    parser = build_parser()
    with pytest.raises(SystemExit):
        # --manifest / --out / --policy-module all required
        parser.parse_args(["gen-node", str(tmp_path / "c.yaml")])


def test_parser_parses_full_invocation(tmp_path):
    parser = build_parser()
    args = parser.parse_args([
        "gen-node",
        "contract.yaml",
        "--policy-module", "pkg.policies.act",
        "--manifest", "manifest.json",
        "--out", "node.py",
        "--force",
        "--node-name", "MyNode",
    ])
    assert args.command == "gen-node"
    assert args.contract == Path("contract.yaml")
    assert args.policy_module == "pkg.policies.act"
    assert args.manifest == Path("manifest.json")
    assert args.out == Path("node.py")
    assert args.force is True
    assert args.node_name == "MyNode"


def test_main_valid_contract_returns_0(tmp_path):
    contract = _write(tmp_path, f"""
        name: ok
        version: 1
        fps: 30
        {_OBS_BLOCK}
        {_ACTION_BLOCK}
    """)
    out = tmp_path / "node.py"
    code = main(_required_argv(contract, out, tmp_path / "manifest.json"))
    assert code == 0
    assert out.exists()


def test_main_refusal_does_not_write_output(tmp_path):
    """A refused contract must not leave a partial file behind.

    Refusals run before rendering, so the file should never be touched
    — important because step 4 will add an overwrite guard, and a half-
    written file from a refused run would poison subsequent attempts.
    """
    contract = _write(tmp_path, """
        name: c
        version: 1
        fps: 30
        observations:
          - key: observation.state
            topic: /robot/joint_states
            type: sensor_msgs/msg/JointState
            selector: {names: [j1]}
            align: {method: linear, tolerance_ms: 100}
        actions: []
    """)
    out = tmp_path / "node.py"
    code = main(_required_argv(contract, out, tmp_path / "manifest.json"))
    assert code == 2
    assert not out.exists()


def test_main_refusal_returns_2(tmp_path, capsys):
    contract = _write(tmp_path, """
        name: c
        version: 1
        fps: 30
        observations:
          - key: observation.state
            topic: /robot/joint_states
            type: sensor_msgs/msg/JointState
            selector: {names: [j1]}
            align: {method: linear, tolerance_ms: 100}
        actions: []
    """)
    code = main(_required_argv(
        contract, tmp_path / "node.py", tmp_path / "manifest.json",
    ))
    assert code == 2
    err = capsys.readouterr().err
    assert "linear" in err
