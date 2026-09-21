"""CLI-level refusal tests for ``roboto-to-lerobot gen-node``.

Covers the refuse-overwrite guard plus a
representative CLI-exit-code path through each of the codegen-time
refuse lists (linear/none policies, non-causal transforms). The
in-process ``validate_contract_for_codegen`` tests already live in
``test_gen_node_cli.py``; these tests assert that the CLI surfaces the
refusal with a non-zero exit code, a usable error message, and no
partially-written output file.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from roboto_to_lerobot.codegen.cli import main

_MINIMAL_OK = """
    name: ok
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
"""


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "contract.yaml"
    path.write_text(dedent(body))
    return path


def _argv(contract: Path, out: Path, tmp_path: Path, *extra: str) -> list[str]:
    return [
        "gen-node", str(contract),
        "--policy-module", "my_pkg.policies.act",
        "--manifest", str(tmp_path / "manifest.json"),
        "--out", str(out),
        *extra,
    ]


# ---------------------------------------------------------------------------
# Refuse-overwrite
# ---------------------------------------------------------------------------


def test_refuse_overwrite_preserves_existing_file(tmp_path, capsys):
    contract = _write(tmp_path, _MINIMAL_OK)
    out = tmp_path / "node.py"
    out.write_text("# user-edited file — keep me\n")
    sentinel = out.read_text()

    code = main(_argv(contract, out, tmp_path))

    assert code == 2
    assert out.read_text() == sentinel, "refused gen-node must not touch the file"
    err = capsys.readouterr().err
    assert "Refusing to overwrite" in err
    assert "--force" in err


def test_force_overwrites_existing_file(tmp_path, capsys):
    contract = _write(tmp_path, _MINIMAL_OK)
    out = tmp_path / "node.py"
    out.write_text("# this gets clobbered\n")

    code = main(_argv(contract, out, tmp_path, "--force"))

    assert code == 0
    new = out.read_text()
    assert "this gets clobbered" not in new
    assert "class OkInferenceNode" in new


def test_no_existing_file_writes_unconditionally(tmp_path):
    """No file at --out ⇒ no guard fires; this is the green-field common case."""
    contract = _write(tmp_path, _MINIMAL_OK)
    out = tmp_path / "node.py"

    code = main(_argv(contract, out, tmp_path))

    assert code == 0
    assert out.exists()


# ---------------------------------------------------------------------------
# CLI exit codes for refuse-list contracts
# ---------------------------------------------------------------------------


def test_refuse_linear_policy_at_cli(tmp_path, capsys):
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
    code = main(_argv(contract, tmp_path / "node.py", tmp_path))
    assert code == 2
    assert "linear" in capsys.readouterr().err


def test_refuse_unsupported_action_type_at_cli(tmp_path, capsys):
    """Action with no registered encoder must refuse before render.

    Mirrors what `LiveAdapter.__init__` would refuse at boot — moved
    to gen-node time so the user sees the unsupported-type error
    before the generated file is ever written.
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
        actions:
          - key: action
            topic: /cmd_vel
            type: geometry_msgs/msg/Twist
    """)
    code = main(_argv(contract, tmp_path / "node.py", tmp_path))
    assert code == 2
    err = capsys.readouterr().err
    assert "geometry_msgs/msg/Twist" in err
    assert "Supported action types" in err
    assert not (tmp_path / "node.py").exists()


def test_refuse_multi_spec_action_without_selector_names(tmp_path, capsys):
    """Two specs sharing a base key need selector.names on every spec.

    Mirrors LiveAdapter's multi-spec selector-width check. Moving the
    refusal forward means the user sees the missing-selector
    message at gen-node time rather than at adapter __init__.
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
        actions:
          - key: action
            topic: /robot_a/cmd
            type: std_msgs/msg/Float64MultiArray
            selector: {names: [a1, a2]}
          - key: action
            topic: /robot_b/cmd
            type: std_msgs/msg/Float64MultiArray
    """)
    code = main(_argv(contract, tmp_path / "node.py", tmp_path))
    assert code == 2
    err = capsys.readouterr().err
    assert "selector.names" in err
    assert "/robot_b/cmd" in err
    assert not (tmp_path / "node.py").exists()


def test_refuse_butterworth_transform_at_cli(tmp_path, capsys):
    contract = _write(tmp_path, """
        name: c
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
            transforms:
              - type: butterworth_lowpass
                stage: pre
                cutoff_hz: 20
                order: 2
    """)
    code = main(_argv(contract, tmp_path / "node.py", tmp_path))
    assert code == 2
    err = capsys.readouterr().err
    assert "butterworth_lowpass" in err
    assert not (tmp_path / "node.py").exists()


def test_refuse_resample_uniform_transform_at_cli(tmp_path, capsys):
    contract = _write(tmp_path, """
        name: c
        version: 1
        fps: 30
        observations:
          - key: observation.state
            topic: /robot/joint_states
            type: sensor_msgs/msg/JointState
            selector: {names: [j1, j2]}
            transforms:
              - type: resample_uniform
                stage: pre
        actions:
          - key: action
            topic: /teleop/action
            type: sensor_msgs/msg/JointState
            selector: {names: [j1, j2]}
    """)
    code = main(_argv(contract, tmp_path / "node.py", tmp_path))
    assert code == 2
    err = capsys.readouterr().err
    assert "resample_uniform" in err
    assert not (tmp_path / "node.py").exists()


def test_causal_lowpass_transform_at_cli_generates_node(tmp_path, capsys):
    """butterworth_lowpass_causal is the one transform gen-node can run
    live; a correctly-configured contract must gen-node successfully."""
    contract = _write(tmp_path, """
        name: ok
        version: 1
        fps: 30
        observations:
          - key: observation.state
            topic: /robot/joint_states
            type: sensor_msgs/msg/JointState
            selector: {names: [j1, j2]}
            transforms:
              - type: butterworth_lowpass_causal
                stage: pre
                cutoff_hz: 5.0
                fs_hz: 100.0
        actions:
          - key: action
            topic: /teleop/action
            type: sensor_msgs/msg/JointState
            selector: {names: [j1, j2]}
    """)
    code = main(_argv(contract, tmp_path / "node.py", tmp_path))
    assert code == 0
    assert (tmp_path / "node.py").exists()


def test_refuse_causal_lowpass_pre_stage_without_fs_hz_at_cli(tmp_path, capsys):
    contract = _write(tmp_path, """
        name: c
        version: 1
        fps: 30
        observations:
          - key: observation.state
            topic: /robot/joint_states
            type: sensor_msgs/msg/JointState
            selector: {names: [j1, j2]}
            transforms:
              - type: butterworth_lowpass_causal
                stage: pre
                cutoff_hz: 5.0
        actions:
          - key: action
            topic: /teleop/action
            type: sensor_msgs/msg/JointState
            selector: {names: [j1, j2]}
    """)
    code = main(_argv(contract, tmp_path / "node.py", tmp_path))
    assert code == 2
    err = capsys.readouterr().err
    assert "fs_hz" in err
    assert not (tmp_path / "node.py").exists()


def test_refuse_causal_lowpass_post_stage_contradicting_fps_at_cli(tmp_path, capsys):
    contract = _write(tmp_path, """
        name: c
        version: 1
        fps: 30
        observations:
          - key: observation.state
            topic: /robot/joint_states
            type: sensor_msgs/msg/JointState
            selector: {names: [j1, j2]}
            transforms:
              - type: butterworth_lowpass_causal
                stage: post
                cutoff_hz: 5.0
                fs_hz: 60.0
        actions:
          - key: action
            topic: /teleop/action
            type: sensor_msgs/msg/JointState
            selector: {names: [j1, j2]}
    """)
    code = main(_argv(contract, tmp_path / "node.py", tmp_path))
    assert code == 2
    err = capsys.readouterr().err
    assert "contradicts" in err
    assert not (tmp_path / "node.py").exists()


def test_refuse_causal_lowpass_transform_on_video_at_cli(tmp_path, capsys):
    contract = _write(tmp_path, """
        name: c
        version: 1
        fps: 30
        observations:
          - key: observation.cam
            topic: /camera/compressed
            type: sensor_msgs/msg/CompressedImage
            image: {resize: [64, 64]}
            transforms:
              - type: butterworth_lowpass_causal
                stage: post
                cutoff_hz: 5.0
        actions:
          - key: action
            topic: /teleop/action
            type: sensor_msgs/msg/JointState
            selector: {names: [j1, j2]}
    """)
    code = main(_argv(contract, tmp_path / "node.py", tmp_path))
    assert code == 2
    err = capsys.readouterr().err
    assert "video" in err
    assert not (tmp_path / "node.py").exists()


def test_refuse_non_listed_transform_at_cli(tmp_path, capsys):
    """A transform outside the non-causal list (e.g. finite_difference) is
    still refused — the live runtime applies no transforms at all."""
    contract = _write(tmp_path, """
        name: c
        version: 1
        fps: 30
        observations:
          - key: observation.state
            topic: /robot/joint_states
            type: sensor_msgs/msg/JointState
            selector: {names: [j1, j2]}
            transforms:
              - type: finite_difference
                stage: pre
        actions:
          - key: action
            topic: /teleop/action
            type: sensor_msgs/msg/JointState
            selector: {names: [j1, j2]}
    """)
    code = main(_argv(contract, tmp_path / "node.py", tmp_path))
    assert code == 2
    err = capsys.readouterr().err
    assert "finite_difference" in err
    assert not (tmp_path / "node.py").exists()


def test_refuse_observation_without_decoder_at_cli(tmp_path, capsys):
    """An observation type with no registered decoder refuses before render,
    mirroring LiveAdapter's boot-time decoder gate."""
    contract = _write(tmp_path, """
        name: c
        version: 1
        fps: 30
        observations:
          - key: observation.temp
            topic: /sensor/temp
            type: sensor_msgs/msg/Temperature
        actions:
          - key: action
            topic: /teleop/action
            type: sensor_msgs/msg/JointState
            selector: {names: [j1, j2]}
    """)
    code = main(_argv(contract, tmp_path / "node.py", tmp_path))
    assert code == 2
    err = capsys.readouterr().err
    assert "no decoder registered" in err
    assert not (tmp_path / "node.py").exists()


def test_refuse_compressed_video_observation_at_cli(tmp_path, capsys):
    """A live node cannot decode compressed video: each message is one encoded
    access unit that needs its GOP prefix, and the node keeps no decoder state
    between messages. The offline converter supports these topics, so the
    refusal has to be explicit — the type does have a registered decoder."""
    contract = _write(tmp_path, """
        name: c
        version: 1
        fps: 30
        observations:
          - key: observation.images.hand
            topic: /camera/hand/video
            type: foxglove_msgs/msg/CompressedVideo
            image: {resize: [64, 64]}
        actions:
          - key: action
            topic: /teleop/action
            type: sensor_msgs/msg/JointState
            selector: {names: [j1, j2]}
    """)
    code = main(_argv(contract, tmp_path / "node.py", tmp_path))
    assert code == 2
    err = capsys.readouterr().err
    assert "compressed-video type" in err
    assert "offline conversion only" in err
    assert not (tmp_path / "node.py").exists()
