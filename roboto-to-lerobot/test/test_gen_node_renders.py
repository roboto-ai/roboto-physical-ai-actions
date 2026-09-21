"""Render-side tests for codegen.render — turn contracts into source.

Asserts the rendered Python source parses cleanly via :func:`ast.parse`
and that the load-bearing structural elements survive: the verbatim
runtime import surface the generated node pins (``LiveAdapter,
load_contract, verify_manifest``), the ``CONTRACT_SHA256`` constant
baked in for boot-time drift detection, and per-spec subscriptions /
publishers.

Three contracts cover the common shapes a user will throw at the
generator: a minimal vector contract, a multi-camera contract that
exercises sensor/state QoS dispatch, and one that uses an explicit
``qos:`` override to verify inline-profile emission.
"""

from __future__ import annotations

import ast
from pathlib import Path
from textwrap import dedent

from roboto_to_lerobot.codegen.render import render_node
from roboto_to_lerobot.runtime.contract_io import load_contract

REQUIRED_RUNTIME_IMPORT = (
    "from roboto_to_lerobot.runtime import LiveAdapter, load_contract, verify_manifest"
)


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "contract.yaml"
    path.write_text(dedent(body))
    return path


def _render(tmp_path: Path, body: str, **kwargs) -> tuple[str, Path]:
    path = _write(tmp_path, body)
    contract = load_contract(path)
    source = render_node(
        contract,
        policy_module=kwargs.pop("policy_module", "my_pkg.policies.act"),
        manifest_path=kwargs.pop("manifest_path", tmp_path / "manifest.json"),
        node_class_name=kwargs.pop("node_class_name", None),
        apply_black=kwargs.pop("apply_black", True),
    )
    return source, path


# ---------------------------------------------------------------------------
# Minimal vector contract — happy path, ast-parses, has required structure
# ---------------------------------------------------------------------------


_MINIMAL = """
    name: minimal
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


def test_minimal_contract_renders_and_parses(tmp_path):
    source, _ = _render(tmp_path, _MINIMAL)
    ast.parse(source)  # raises SyntaxError if the template emitted garbage


def test_minimal_contract_has_required_runtime_import(tmp_path):
    source, _ = _render(tmp_path, _MINIMAL)
    assert REQUIRED_RUNTIME_IMPORT in source


def test_minimal_contract_bakes_in_contract_sha(tmp_path):
    source, path = _render(tmp_path, _MINIMAL)
    import hashlib

    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert f'CONTRACT_SHA256 = "{expected}"' in source


def test_minimal_contract_imports_policy_module(tmp_path):
    source, _ = _render(
        tmp_path, _MINIMAL, policy_module="my_team.policies.fancy",
    )
    assert "from my_team.policies.fancy import load_policy" in source


def test_minimal_contract_default_node_class_name(tmp_path):
    source, _ = _render(tmp_path, _MINIMAL)
    assert "class MinimalInferenceNode(Node)" in source


def test_minimal_contract_custom_node_class_name(tmp_path):
    source, _ = _render(tmp_path, _MINIMAL, node_class_name="MyCustomNode")
    assert "class MyCustomNode(Node)" in source


# ---------------------------------------------------------------------------
# Multi-camera contract — sensor vs reliable QoS dispatch, per-spec callbacks
# ---------------------------------------------------------------------------


_MULTI_CAM = """
    name: multi_cam
    version: 1
    fps: 30
    observations:
      - key: observation.images.left
        topic: /camera/left/image_raw/compressed
        type: sensor_msgs/msg/CompressedImage
        image: {resize: [240, 320]}
        align: {method: nearest, tolerance_ms: 100}
      - key: observation.images.right
        topic: /camera/right/image_raw/compressed
        type: sensor_msgs/msg/CompressedImage
        image: {resize: [240, 320]}
        align: {method: nearest, tolerance_ms: 100}
      - key: observation.state
        topic: /robot/joint_states
        type: sensor_msgs/msg/JointState
        selector: {names: [j1, j2, j3]}
        align: {method: hold, tolerance_ms: 500}
    actions:
      - key: action
        topic: /teleop/action
        type: sensor_msgs/msg/JointState
        selector: {names: [j1, j2, j3]}
"""


def test_multi_cam_contract_renders_and_parses(tmp_path):
    source, _ = _render(tmp_path, _MULTI_CAM)
    ast.parse(source)


def test_multi_cam_emits_one_subscription_per_topic(tmp_path):
    source, _ = _render(tmp_path, _MULTI_CAM)
    # Two camera topics + one joint_states = 3 subscriptions.
    assert source.count("self.create_subscription") == 3


def test_multi_cam_subscribes_with_sensor_qos_for_images(tmp_path):
    source, _ = _render(tmp_path, _MULTI_CAM)
    # Count the QoS token *as an argument* (trailing comma) so the shared
    # profile definitions ("qos_sensor = QoSProfile(...)") don't inflate the
    # count: a definition is "qos_sensor =", a use is "qos_sensor,". Two image
    # subscriptions take qos_sensor; the joint_states sub + the action
    # publisher take qos_reliable. If an image sub regressed to the reliable
    # bucket these counts would shift, so they pin per-spec dispatch.
    assert source.count("qos_sensor,") == 2
    assert source.count("qos_reliable,") == 2


def test_multi_cam_publishes_with_reliable_qos_for_action(tmp_path):
    source, _ = _render(tmp_path, _MULTI_CAM)
    assert "self.create_publisher" in source
    # The action is a JointState ⇒ reliable bucket. Assert the QoS argument
    # lands *inside* the create_publisher(...) call rather than merely
    # somewhere in the file.
    start = source.index("create_publisher(")
    call = source[start : source.index(")", start)]
    assert "qos_reliable" in call, f"publisher call missing reliable QoS: {call!r}"


def test_multi_cam_callback_method_per_subscription(tmp_path):
    source, _ = _render(tmp_path, _MULTI_CAM)
    # Sanitized callback names derived from topic.
    assert "_on_camera_left_image_raw_compressed" in source
    assert "_on_camera_right_image_raw_compressed" in source
    assert "_on_robot_joint_states" in source


# ---------------------------------------------------------------------------
# Identifier-collision disambiguation — distinct topics that sanitize to the
# same Python identifier must not collapse onto one callback/publisher.
# ---------------------------------------------------------------------------


_COLLIDING_TOPICS = """
    name: collide
    version: 1
    fps: 30
    observations:
      - key: observation.a
        topic: /robot/joint_states
        type: sensor_msgs/msg/JointState
        selector: {names: [j1]}
      - key: observation.b
        topic: /robot/joint-states
        type: sensor_msgs/msg/JointState
        selector: {names: [j2]}
    actions:
      - key: action.a
        topic: /arm/cmd
        type: sensor_msgs/msg/JointState
        selector: {names: [j1]}
      - key: action.b
        topic: /arm-cmd
        type: sensor_msgs/msg/JointState
        selector: {names: [j2]}
"""


def test_colliding_subscription_topics_get_distinct_callbacks(tmp_path):
    source, _ = _render(tmp_path, _COLLIDING_TOPICS)
    ast.parse(source)
    # Both topics sanitize to `_on_robot_joint_states`; the second must be
    # suffixed so each distinct topic keeps its own callback. Two distinct
    # `def` callbacks and two subscriptions, not one of each.
    assert source.count("def _on_robot_joint_states(") == 1
    assert source.count("def _on_robot_joint_states_2(") == 1
    assert source.count("self.create_subscription") == 2


def test_colliding_publisher_topics_get_distinct_attrs(tmp_path):
    source, _ = _render(tmp_path, _COLLIDING_TOPICS)
    # `/arm/cmd` and `/arm-cmd` both sanitize to `_pub_arm_cmd`; the second
    # must be suffixed so each action topic gets its own publisher.
    assert "self._pub_arm_cmd " in source
    assert "self._pub_arm_cmd_2 " in source
    assert source.count("self.create_publisher") == 2


# ---------------------------------------------------------------------------
# Explicit QoS override — inline QoSProfile emission
# ---------------------------------------------------------------------------


_QOS_OVERRIDE = """
    name: qos_override
    version: 1
    fps: 30
    observations:
      - key: observation.state
        topic: /robot/joint_states
        type: sensor_msgs/msg/JointState
        selector: {names: [j1]}
        qos: {reliability: BEST_EFFORT, history: KEEP_LAST, depth: 5}
    actions:
      - key: action
        topic: /teleop/action
        type: sensor_msgs/msg/JointState
        selector: {names: [j1]}
        qos: {reliability: RELIABLE, history: KEEP_ALL, depth: 50}
"""


def test_qos_override_renders_and_parses(tmp_path):
    source, _ = _render(tmp_path, _QOS_OVERRIDE)
    ast.parse(source)


def test_qos_override_emits_inline_qosprofile(tmp_path):
    source, _ = _render(tmp_path, _QOS_OVERRIDE)
    # Both inline calls present with the customised depth values.
    assert "depth=5" in source
    assert "depth=50" in source
    assert "ReliabilityPolicy.BEST_EFFORT" in source
    assert "ReliabilityPolicy.RELIABLE" in source
    assert "HistoryPolicy.KEEP_ALL" in source


# ---------------------------------------------------------------------------
# CLI end-to-end — gen-node writes a file that ast-parses
# ---------------------------------------------------------------------------


def test_cli_main_writes_parseable_file(tmp_path):
    from roboto_to_lerobot.codegen.cli import main

    contract = _write(tmp_path, _MULTI_CAM)
    out = tmp_path / "node.py"
    code = main([
        "gen-node", str(contract),
        "--policy-module", "my_pkg.policies.act",
        "--manifest", str(tmp_path / "manifest.json"),
        "--out", str(out),
    ])
    assert code == 0
    assert out.exists()
    ast.parse(out.read_text())
