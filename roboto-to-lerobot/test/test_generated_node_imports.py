"""End-to-end smoke test: render a node, monkey-patch rclpy, drive it.

The full e2e gate lives in the docker bag-replay smoke because real
``rclpy`` does not install cleanly from PyPI — it ships as part of a
sourced ROS 2 distro. This test is the cheap CI layer
that catches everything *except* ROS-specific bugs: contract wiring,
encoder dispatch, the tick path's policy invocation, the
publisher-by-topic lookup. If this passes, the generated file is
well-formed Python and the data-path through the runtime kernel
works; if it fails on a stock dev box, the bug is in our codegen,
not in ROS.

The test installs fakes for ``rclpy`` (Node, qos enums, init/spin
no-ops), ``sensor_msgs.msg`` (JointState constructor that captures
``**kwargs``), and the user-side ``my_pkg.policies.act``
(``load_policy`` returning a recorder fn). It then renders a tiny
contract, importlib-loads the rendered file into a private module
name, instantiates the node, drives one subscription callback to
push an observation, runs the timer once, and asserts the publisher
saw a constructed message.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from textwrap import dedent
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from roboto_to_lerobot.codegen.render import render_node
from roboto_to_lerobot.runtime.contract_io import load_contract

# ---------------------------------------------------------------------------
# Fakes — installed into sys.modules before importing the generated file
# ---------------------------------------------------------------------------


class _FakeNode:
    """Minimal stand-in for ``rclpy.node.Node``.

    Records the wiring the generated ``__init__`` does (subscriptions,
    publishers, timer) so the test body can drive them directly. The
    real Node would also start an executor; this stub never spins.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._params: dict[str, Any] = {}
        self.subscriptions: list[tuple[str, Any]] = []
        self.publishers: dict[str, MagicMock] = {}
        self.timer_callback: Any = None
        self._clock_ns = 1_000_000_000  # arbitrary non-zero start

    def declare_parameter(self, name: str, default: Any = None):
        self._params[name] = default

    def get_parameter(self, name: str):
        return types.SimpleNamespace(value=self._params.get(name))

    def create_subscription(self, msg_cls, topic, callback, qos):
        self.subscriptions.append((topic, callback))

    def create_publisher(self, msg_cls, topic, qos):
        pub = MagicMock()
        self.publishers[topic] = pub
        return pub

    def create_timer(self, period_s: float, callback):
        self.timer_callback = callback

    def get_clock(self):
        clock = MagicMock()
        clock.now.return_value = types.SimpleNamespace(nanoseconds=self._clock_ns)
        return clock

    def destroy_node(self) -> None:
        pass


class _FakeQosEnum:
    """All four ROS QoS enum values flattened onto one shim class.

    The template emits ``ReliabilityPolicy.BEST_EFFORT`` /
    ``HistoryPolicy.KEEP_LAST`` etc.; both classes resolve to this
    type, which carries every name as a class attribute. Simpler than
    minting two separate stubs that look identical.
    """

    BEST_EFFORT = "BEST_EFFORT"
    RELIABLE = "RELIABLE"
    KEEP_LAST = "KEEP_LAST"
    KEEP_ALL = "KEEP_ALL"


class _FakeMsg:
    """Generic ``MessageType(**payload)`` shim.

    Every ROS message ``MyType(**payload)`` call in the generated
    publish path lands here; payloads are captured on the instance so
    assertions can inspect what was published.
    """

    def __init__(self, **payload):
        self.__dict__.update(payload)


def _install_rclpy_fakes() -> list[str]:
    """Plant ``rclpy*`` / ``sensor_msgs.msg`` / ``my_pkg.*`` in sys.modules.

    Returns the list of installed module names so a fixture can wipe
    them after the test; leaving them in ``sys.modules`` would poison
    other tests in the same pytest session.
    """
    installed: list[str] = []

    def _put(name: str, module: types.ModuleType) -> None:
        sys.modules[name] = module
        installed.append(name)

    rclpy_mod = types.ModuleType("rclpy")
    rclpy_mod.init = MagicMock()
    rclpy_mod.spin = MagicMock()
    rclpy_mod.shutdown = MagicMock()
    _put("rclpy", rclpy_mod)

    rclpy_node_mod = types.ModuleType("rclpy.node")
    rclpy_node_mod.Node = _FakeNode
    _put("rclpy.node", rclpy_node_mod)

    rclpy_qos_mod = types.ModuleType("rclpy.qos")
    rclpy_qos_mod.QoSProfile = MagicMock
    rclpy_qos_mod.ReliabilityPolicy = _FakeQosEnum
    rclpy_qos_mod.HistoryPolicy = _FakeQosEnum
    _put("rclpy.qos", rclpy_qos_mod)

    sensor_msgs_mod = types.ModuleType("sensor_msgs")
    sensor_msgs_msg_mod = types.ModuleType("sensor_msgs.msg")
    sensor_msgs_msg_mod.JointState = _FakeMsg
    sensor_msgs_msg_mod.CompressedImage = _FakeMsg
    sensor_msgs_msg_mod.Image = _FakeMsg
    _put("sensor_msgs", sensor_msgs_mod)
    _put("sensor_msgs.msg", sensor_msgs_msg_mod)

    std_msgs_mod = types.ModuleType("std_msgs")
    std_msgs_msg_mod = types.ModuleType("std_msgs.msg")
    std_msgs_msg_mod.Float64MultiArray = _FakeMsg
    std_msgs_msg_mod.Float64 = _FakeMsg
    _put("std_msgs", std_msgs_mod)
    _put("std_msgs.msg", std_msgs_msg_mod)

    # The user-side policy module the generated file imports.
    # `load_policy` returns a recorder so the test can inspect the
    # observation dict the policy was called with.
    policy_module = types.ModuleType("fake_policies")

    def load_policy(_path: str):
        return _RecordingPolicy()

    policy_module.load_policy = load_policy
    _put("fake_policies", policy_module)

    return installed


class _RecordingPolicy:
    """Policy stub that captures its calls and returns a constant action."""

    def __init__(self) -> None:
        self.calls: list[dict[str, np.ndarray]] = []

    def __call__(self, obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        self.calls.append(obs)
        # Match the action selector ([j1, j2]) in the test contract.
        return {"action": np.array([0.5, 0.75])}


# ---------------------------------------------------------------------------
# Fixture: render + import + cleanup
# ---------------------------------------------------------------------------


@pytest.fixture
def generated_module(tmp_path: Path):
    """Render a small contract, install fakes, importlib-load the file."""
    installed_names = _install_rclpy_fakes()
    try:
        contract_path = tmp_path / "contract.yaml"
        contract_path.write_text(dedent("""
            name: smoke
            version: 1
            fps: 30
            observations:
              - key: observation.state
                topic: /robot/joint_states
                type: sensor_msgs/msg/JointState
                selector: {names: [j1, j2]}
                align: {method: hold, tolerance_ms: 500}
            actions:
              - key: action
                topic: /teleop/action
                type: sensor_msgs/msg/JointState
                selector: {names: [j1, j2]}
        """))

        # Bake a manifest that matches the contract sha so verify_manifest
        # passes at boot — otherwise __init__ raises before we can drive it.
        import hashlib
        import json
        sha = hashlib.sha256(contract_path.read_bytes()).hexdigest()
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps({"contract": {"sha256": sha}}))

        contract = load_contract(contract_path)
        source = render_node(
            contract,
            policy_module="fake_policies",
            manifest_path=manifest_path,
        )
        out = tmp_path / "generated_node.py"
        out.write_text(source)

        spec = importlib.util.spec_from_file_location("_smoke_generated_node", out)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        yield module, contract_path, manifest_path
    finally:
        for name in installed_names:
            sys.modules.pop(name, None)
        sys.modules.pop("_smoke_generated_node", None)


# ---------------------------------------------------------------------------
# The smoke test itself
# ---------------------------------------------------------------------------


def test_generated_node_imports_and_constructs(generated_module):
    """The rendered file imports under stock Python and constructs the Node."""
    module, _, _ = generated_module
    assert hasattr(module, "SmokeInferenceNode")
    node = module.SmokeInferenceNode()
    # Subscriptions + publishers wired during __init__.
    assert ("/robot/joint_states", node._on_robot_joint_states) in node.subscriptions
    assert "/teleop/action" in node.publishers


def test_generated_node_tick_calls_policy_and_publishes(generated_module):
    """Drive a callback → run the timer → assert policy and publisher fired."""
    module, _, _ = generated_module
    node = module.SmokeInferenceNode()

    # Drive a synthetic JointState message into the only subscription.
    msg = types.SimpleNamespace(
        header=types.SimpleNamespace(stamp=types.SimpleNamespace(sec=1, nanosec=0)),
        name=["j1", "j2"],
        position=[0.1, 0.2],
    )
    node._on_robot_joint_states(msg)

    # Tick now: with the buffer freshly populated, sample() must not be stale.
    node._on_tick()

    # The recording policy captured the observation dict.
    assert isinstance(node._policy, _RecordingPolicy)
    assert len(node._policy.calls) == 1
    assert "observation.state" in node._policy.calls[0]
    # Observations are float32 for byte-parity with the converter, so the
    # round-tripped values differ from the float64 inputs at float32 epsilon.
    obs_state = node._policy.calls[0]["observation.state"]
    assert obs_state.dtype == np.float32
    np.testing.assert_allclose(obs_state, [0.1, 0.2], rtol=1e-6)

    # The publisher received exactly one constructed message with the
    # policy's encoded payload.
    pub = node.publishers["/teleop/action"]
    pub.publish.assert_called_once()
    published_msg = pub.publish.call_args.args[0]
    assert isinstance(published_msg, _FakeMsg)
    assert published_msg.name == ["j1", "j2"]
    assert published_msg.position == [0.5, 0.75]


def test_generated_node_tick_skips_publish_when_no_observation(generated_module):
    """publish_nothing: a tick with an empty buffer must not publish anything."""
    module, _, _ = generated_module
    node = module.SmokeInferenceNode()

    # No callback driven ⇒ buffer empty ⇒ sample() returns None at the
    # empty-buffer guard (before the staleness path).
    node._on_tick()

    pub = node.publishers["/teleop/action"]
    pub.publish.assert_not_called()
    assert isinstance(node._policy, _RecordingPolicy)
    assert node._policy.calls == []


def test_generated_node_tick_skips_publish_when_sample_is_stale(generated_module):
    """publish_nothing on the *staleness* path: a buffer holding a sample
    older than tolerance must skip publish, not just the empty buffer.

    The previous test never populates the buffer, so it stops at the
    empty-buffer guard and never reaches the staleness/tolerance check.
    Here we push a sample, advance the node clock well past the 500 ms
    tolerance, then tick — sample() must return None and the policy must
    never run.
    """
    module, _, _ = generated_module
    node = module.SmokeInferenceNode()

    msg = types.SimpleNamespace(
        header=types.SimpleNamespace(stamp=types.SimpleNamespace(sec=1, nanosec=0)),
        name=["j1", "j2"],
        position=[0.1, 0.2],
    )
    node._on_robot_joint_states(msg)  # pushed at the node's current clock

    # Advance arrival clock 10 s — far beyond the contract's 500 ms tolerance.
    node._clock_ns += 10_000_000_000
    node._on_tick()

    pub = node.publishers["/teleop/action"]
    pub.publish.assert_not_called()
    assert node._policy.calls == [], "stale sample must not reach the policy"


def test_generated_node_refuses_on_contract_drift(generated_module):
    """Boot-time drift guard: a contract whose bytes changed since codegen
    must raise before the adapter is built (the baked CONTRACT_SHA256 check)."""
    module, contract_path, _ = generated_module
    # Mutate the on-disk contract so its sha no longer matches the baked one.
    contract_path.write_text(contract_path.read_text() + "\n# edited after codegen\n")

    with pytest.raises(RuntimeError, match="drift"):
        module.SmokeInferenceNode()


def test_generated_node_refuses_on_foreign_manifest(generated_module):
    """Boot-time manifest guard: a manifest recording a different contract
    sha must raise (the contract itself is untouched, so drift passes first)."""
    import json

    module, _, manifest_path = generated_module
    manifest_path.write_text(json.dumps({"contract": {"sha256": "deadbeef"}}))

    with pytest.raises(ValueError, match="sha"):
        module.SmokeInferenceNode()


def test_generated_node_handles_headerless_message_type(tmp_path):
    """A header-less observation type must flow through the callback.

    The callback keys samples on arrival time (``get_clock().now()``), not
    ``msg.header.stamp``, so a message with no ``header`` attribute — e.g.
    ``std_msgs/Float64MultiArray`` — does not raise ``AttributeError``. The
    JointState smoke contract hides this: its synthetic message carries a
    header, so the crash path is structurally invisible there.
    """
    import hashlib
    import json

    installed_names = _install_rclpy_fakes()
    try:
        contract_path = tmp_path / "contract.yaml"
        contract_path.write_text(dedent("""
            name: headerless
            version: 1
            fps: 30
            observations:
              - key: observation.state
                topic: /state
                type: std_msgs/msg/Float64MultiArray
                align: {method: hold, tolerance_ms: 500}
            actions:
              - key: action
                topic: /teleop/action
                type: sensor_msgs/msg/JointState
                selector: {names: [j1, j2]}
        """))
        sha = hashlib.sha256(contract_path.read_bytes()).hexdigest()
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps({"contract": {"sha256": sha}}))

        contract = load_contract(contract_path)
        source = render_node(
            contract,
            policy_module="fake_policies",
            manifest_path=manifest_path,
        )
        out = tmp_path / "generated_node.py"
        out.write_text(source)

        spec = importlib.util.spec_from_file_location("_headerless_generated_node", out)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        node = module.HeaderlessInferenceNode()
        # Header-LESS message: no ``.header`` attribute at all.
        node._on_state(types.SimpleNamespace(data=[0.1, 0.2]))
        node._on_tick()  # would raise AttributeError if the callback read a header

        assert len(node._policy.calls) == 1
        node.publishers["/teleop/action"].publish.assert_called_once()
    finally:
        for name in installed_names:
            sys.modules.pop(name, None)
        sys.modules.pop("_headerless_generated_node", None)
