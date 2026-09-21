"""Fast CI gate for the `runtime/` submodule.

Complements the hosted-compute byte-equivalence verifier
(`scripts/verify_byte_equivalence.py`). Two jobs:

1. The action-facing primitives (`runtime.encoders`, `runtime.converters`)
   import without the converter's heavy deps (pandas/mcap/roboto), so a live
   ROS 2 node can use them — asserted in a subprocess (an in-process
   `sys.modules` check is meaningless once any test imports pandas).
2. The shims, registries, image ops, encoders, and contract_io behave.

`runtime.image` / `runtime.decoders` / `runtime.contract_io` still pull
`extract` / `contract_utils` (pandas/cv2/roboto) today; making those light is
future work that splits those modules.
"""
from __future__ import annotations

import importlib
import subprocess
import sys
import textwrap


def _import_in_clean_subprocess(body: str) -> subprocess.CompletedProcess:
    """Run `body` in a fresh interpreter and return the completed process.

    Used for checks that depend on a pristine import state — heavy-dep
    isolation and import-time registration — which cannot be observed in this
    process once another test has imported pandas / populated the registry.
    """
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True,
        text=True,
    )


def test_runtime_package_importable():
    runtime = importlib.import_module("roboto_to_lerobot.runtime")
    assert runtime.__name__ == "roboto_to_lerobot.runtime"


def test_light_runtime_modules_import_without_heavy_deps():
    """`runtime.encoders` and `runtime.converters` must import without
    pandas/mcap/roboto so a live ROS node lacking the converter stack can use
    them. That is the headline goal; checked per-module in a subprocess."""
    for module in (
        "roboto_to_lerobot.runtime.encoders",
        "roboto_to_lerobot.runtime.converters",
    ):
        result = _import_in_clean_subprocess(
            f"""
            import importlib, sys
            importlib.import_module({module!r})
            leaked = [m for m in ("pandas", "mcap", "roboto") if m in sys.modules]
            if leaked:
                print(",".join(leaked))
                sys.exit(1)
            """
        )
        assert result.returncode == 0, (
            f"importing {module} leaked heavy deps: "
            f"{result.stdout.strip()!r} (stderr: {result.stderr.strip()!r})"
        )


def test_runtime_package_all_surface_exposed():
    """``runtime.__all__`` is the public surface the generated-node template
    imports from. Any drift between the declared ``__all__`` and the
    actually-exported names is a contract bug — codegen pins these names."""
    runtime = importlib.import_module("roboto_to_lerobot.runtime")
    expected = {
        "Contract",
        "LiveAdapter",
        "ReplayAdapter",
        "StreamBuffer",
        "auto_bound_tolerance_ns",
        "load_contract",
        "verify_manifest",
    }
    assert set(runtime.__all__) == expected
    for name in expected:
        assert hasattr(runtime, name), f"runtime.{name} should be exported"


def test_runtime_stream_buffer_importable_directly():
    rt_sb = importlib.import_module("roboto_to_lerobot.runtime.stream_buffer")
    assert hasattr(rt_sb, "StreamBuffer")
    assert hasattr(rt_sb, "auto_bound_tolerance_ns")
    assert rt_sb.SUPPORTED_METHODS == ("hold", "nearest")


def test_runtime_live_adapter_importable_directly():
    rt_la = importlib.import_module("roboto_to_lerobot.runtime.live_adapter")
    assert hasattr(rt_la, "LiveAdapter")


def test_runtime_replay_adapter_importable_directly():
    """The replay module imports. ReplayAdapter defers ``read_ros2_messages``
    to call time, so this module declares no top-level mcap_ros2 dependency —
    but importing it still drags mcap_ros2 into ``sys.modules`` transitively
    via ``live_adapter`` -> ``contract_utils`` -> ``roboto``, so mcap_ros2
    isolation can't be asserted here. The lazy import only starts paying off
    once the heavy-dep split (future work) removes that transitive pull."""
    rt_ra = importlib.import_module("roboto_to_lerobot.runtime.replay_adapter")
    assert hasattr(rt_ra, "ReplayAdapter")


def test_runtime_converters_importable_directly():
    """`runtime.converters` is the authoritative source of DECODERS / register_decoder."""
    rt_converters = importlib.import_module("roboto_to_lerobot.runtime.converters")
    assert hasattr(rt_converters, "DECODERS")
    assert hasattr(rt_converters, "register_decoder")
    assert hasattr(rt_converters, "decode_value")


def test_converters_shim_reexports_runtime():
    """Top-level `converters` is a re-export shim; identity must be preserved
    so decoders registered via either import path land in the same registry."""
    rt_converters = importlib.import_module("roboto_to_lerobot.runtime.converters")
    top_converters = importlib.import_module("roboto_to_lerobot.converters")
    assert top_converters.DECODERS is rt_converters.DECODERS
    assert top_converters.register_decoder is rt_converters.register_decoder
    assert top_converters.decode_value is rt_converters.decode_value


def test_decoders_register_into_runtime_registry():
    """Importing the decoder shim populates the runtime registry. Run in a
    clean subprocess: in this process the registry is already populated by
    other tests importing contract_utils (which imports decoders for its side
    effect), so an in-process `len > 0` would pass even if this import path
    were broken."""
    result = _import_in_clean_subprocess(
        """
        import importlib
        rt = importlib.import_module("roboto_to_lerobot.runtime.converters")
        assert rt.DECODERS == {}, "registry should start empty before import"
        importlib.import_module("roboto_to_lerobot.decoders")
        for key in ("sensor_msgs/msg/JointState", "sensor_msgs/msg/Image"):
            assert key in rt.DECODERS, f"{key} not registered"
        """
    )
    assert result.returncode == 0, result.stderr


def test_runtime_decoders_importable_directly():
    """`runtime.decoders` is the authoritative module; importing it directly
    (not via the shim) registers into the runtime registry. Verified in a
    clean subprocess for the same registry-already-populated reason."""
    result = _import_in_clean_subprocess(
        """
        import importlib
        rt = importlib.import_module("roboto_to_lerobot.runtime.converters")
        assert rt.DECODERS == {}, "registry should start empty before import"
        importlib.import_module("roboto_to_lerobot.runtime.decoders")
        assert "sensor_msgs/msg/JointState" in rt.DECODERS
        """
    )
    assert result.returncode == 0, result.stderr


def test_runtime_image_resize_noop_fast_path():
    """resize_image returns the input untouched when the shape already
    matches — converter relies on this for native-resolution streams."""
    import numpy as np

    rt_image = importlib.import_module("roboto_to_lerobot.runtime.image")
    img = np.zeros((4, 8, 3), dtype=np.uint8)
    assert rt_image.resize_image(img, 4, 8) is img


def test_runtime_image_resize_changes_shape():
    import numpy as np

    rt_image = importlib.import_module("roboto_to_lerobot.runtime.image")
    img = np.zeros((4, 8, 3), dtype=np.uint8)
    out = rt_image.resize_image(img, 2, 4)
    assert out.shape == (2, 4, 3)


def test_runtime_image_reexports_depth_helper():
    """``depth_to_uint8_rgb`` is the same callable whether imported from
    ``runtime.image`` (the live-runtime path) or ``extract`` (the converter
    path) — both paths must reach the identical implementation so the
    eventual depth-stream parity story is single-source."""
    rt_image = importlib.import_module("roboto_to_lerobot.runtime.image")
    extract = importlib.import_module("roboto_to_lerobot.extract")
    assert rt_image.depth_to_uint8_rgb is extract.depth_to_uint8_rgb


def test_runtime_encoders_importable():
    """`runtime.encoders` is purely additive (no caller in the
    converter pipeline) and exposes the registry surface. The no-heavy-dep
    guarantee is checked separately in
    test_light_runtime_modules_import_without_heavy_deps."""
    rt_encoders = importlib.import_module("roboto_to_lerobot.runtime.encoders")
    assert hasattr(rt_encoders, "ENCODERS")
    assert hasattr(rt_encoders, "register_encoder")
    assert hasattr(rt_encoders, "encode_value")
    # All three target message types are registered out of the box.
    for type_str in (
        "sensor_msgs/msg/JointState",
        "std_msgs/msg/Float64MultiArray",
        "std_msgs/msg/Float64",
    ):
        assert type_str in rt_encoders.ENCODERS


def test_joint_state_encode_payload():
    """The encoder maps an action vector to the minimal JointState payload,
    in selector order."""
    import numpy as np
    from roboto_to_lerobot.contract_utils import ActionSpec
    from roboto_to_lerobot.runtime.encoders import encode_value

    spec = ActionSpec(
        key="action",
        topic="/teleop/action",
        type="sensor_msgs/msg/JointState",
        selector={"names": ["joint_a", "joint_b", "joint_c"]},
    )
    action = np.array([0.1, -0.2, 1.5], dtype=np.float64)
    payload = encode_value(action, spec)
    assert payload == {
        "name": ["joint_a", "joint_b", "joint_c"],
        "position": [0.1, -0.2, 1.5],
    }


def test_joint_state_encode_decode_round_trip():
    """A genuine round trip for the position field: decode a JointState
    message to an action vector, encode it back, and re-decode the payload —
    the result must match. Covers only the no-transform position path, the
    only case the encoders claim to invert."""
    import numpy as np
    import pandas as pd
    from roboto_to_lerobot.contract_utils import ActionSpec
    from roboto_to_lerobot.runtime.converters import decode_value
    from roboto_to_lerobot.runtime.encoders import encode_value

    spec = ActionSpec(
        key="action",
        topic="/teleop/action",
        type="sensor_msgs/msg/JointState",
        selector={"names": ["joint_a", "joint_b", "joint_c"]},
    )
    # A JointState message as the converter sees it (pandas Series).
    msg = pd.Series({"name": ["joint_a", "joint_b", "joint_c"],
                     "position": [0.1, -0.2, 1.5]})
    decoded = decode_value(msg, spec)             # array in selector order
    payload = encode_value(decoded, spec)         # back to a JointState dict
    redecoded = decode_value(pd.Series(payload), spec)
    assert np.allclose(decoded, redecoded)
    assert payload["name"] == ["joint_a", "joint_b", "joint_c"]
    assert np.allclose(payload["position"], [0.1, -0.2, 1.5])


def test_joint_state_encode_rejects_non_position_field():
    """Selectors like 'velocity.joint_a' decode fine but can't be
    encoded — the action vector has no signal for which field each entry
    represents, so the encoder must refuse rather than silently mislabel."""
    import numpy as np
    from roboto_to_lerobot.contract_utils import ActionSpec
    from roboto_to_lerobot.runtime.encoders import encode_value

    spec = ActionSpec(
        key="action",
        topic="/teleop/action",
        type="sensor_msgs/msg/JointState",
        selector={"names": ["velocity.joint_a"]},
    )
    try:
        encode_value(np.array([0.5]), spec)
    except ValueError as exc:
        assert "position" in str(exc)
    else:
        raise AssertionError("expected ValueError on non-position selector")


def test_float64_multiarray_encode():
    import numpy as np
    from roboto_to_lerobot.contract_utils import ActionSpec
    from roboto_to_lerobot.runtime.encoders import encode_value

    spec = ActionSpec(
        key="action",
        topic="/action_vec",
        type="std_msgs/msg/Float64MultiArray",
    )
    payload = encode_value(np.array([1.0, 2.0, 3.0]), spec)
    assert payload == {"data": [1.0, 2.0, 3.0]}


def test_float64_scalar_encode_accepts_zero_d():
    import numpy as np
    from roboto_to_lerobot.contract_utils import ActionSpec
    from roboto_to_lerobot.runtime.encoders import encode_value

    spec = ActionSpec(
        key="action",
        topic="/action_scalar",
        type="std_msgs/msg/Float64",
    )
    assert encode_value(np.float64(0.7), spec) == {"data": 0.7}
    assert encode_value(np.array([0.7]), spec) == {"data": 0.7}


def test_encode_value_refuses_unregistered_type():
    import numpy as np
    from roboto_to_lerobot.contract_utils import ActionSpec
    from roboto_to_lerobot.runtime.encoders import encode_value

    spec = ActionSpec(
        key="action",
        topic="/cmd_vel",
        type="geometry_msgs/msg/Twist",
    )
    try:
        encode_value(np.array([1.0, 0.0]), spec)
    except ValueError as exc:
        assert "geometry_msgs/msg/Twist" in str(exc)
    else:
        raise AssertionError("expected ValueError on unregistered type")


# ----------------------------------------------------------------------------
# runtime/contract_io — load_contract wrapper, Contract.sha256, verify_manifest
# ----------------------------------------------------------------------------


def _write_minimal_contract(tmp_path):
    """Write a minimal but valid contract YAML; returns the file path."""
    p = tmp_path / "contract.yaml"
    # version is deliberately non-default (load_contract defaults to 1) so the
    # assertion below proves the field is read from the file, not defaulted.
    p.write_text(
        "name: test_contract\n"
        "version: 2\n"
        "fps: 30\n"
        "observations: []\n"
        "actions: []\n"
        "tasks: []\n",
        encoding="utf-8",
    )
    return p


def test_runtime_contract_io_importable():
    rt_ci = importlib.import_module("roboto_to_lerobot.runtime.contract_io")
    assert hasattr(rt_ci, "load_contract")
    assert hasattr(rt_ci, "Contract")
    assert hasattr(rt_ci, "verify_manifest")


def test_load_contract_wraps_inner_dataclass(tmp_path):
    """Wrapper delegates attribute access to the underlying contract
    dataclass; live-runtime callers see name/version/fps unchanged."""
    from roboto_to_lerobot.runtime.contract_io import load_contract

    src = _write_minimal_contract(tmp_path)
    contract = load_contract(src)
    assert contract.name == "test_contract"
    assert contract.version == 2
    assert contract.fps == 30
    assert contract.source_path == src


def test_contract_sha256_matches_converter_definition(tmp_path):
    """The runtime SHA must be byte-identical to the converter's
    `hashlib.sha256(contract_path.read_bytes()).hexdigest()` so the
    manifest-baked value compares equal."""
    import hashlib

    from roboto_to_lerobot.runtime.contract_io import load_contract

    src = _write_minimal_contract(tmp_path)
    expected = hashlib.sha256(src.read_bytes()).hexdigest()
    contract = load_contract(src)
    assert contract.sha256() == expected


def test_verify_manifest_passes_on_matching_sha(tmp_path):
    import json

    from roboto_to_lerobot.runtime.contract_io import load_contract, verify_manifest

    src = _write_minimal_contract(tmp_path)
    contract = load_contract(src)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"contract": {"sha256": contract.sha256()}}))
    verify_manifest(manifest, contract)  # must not raise


def test_verify_manifest_refuses_on_drift(tmp_path):
    import json

    from roboto_to_lerobot.runtime.contract_io import load_contract, verify_manifest

    src = _write_minimal_contract(tmp_path)
    contract = load_contract(src)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"contract": {"sha256": "deadbeef" * 8}}))
    try:
        verify_manifest(manifest, contract)
    except ValueError as exc:
        assert "sha" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError on sha mismatch")


def test_contract_getattr_guard_no_recursion():
    """A Contract built without __init__ (as copy/pickle would, bypassing the
    constructor on a __slots__ class) must raise AttributeError, not recurse
    forever, when an attribute is accessed."""
    from roboto_to_lerobot.runtime.contract_io import Contract

    obj = Contract.__new__(Contract)  # slots unset, __init__ bypassed
    try:
        _ = obj.name  # the access itself is the assertion
    except AttributeError:
        pass
    except RecursionError as exc:
        raise AssertionError("__getattr__ guard missing: access recursed") from exc
    else:
        raise AssertionError("expected AttributeError on uninitialized Contract")


def test_verify_manifest_refuses_non_dict_contract_section(tmp_path):
    """A manifest whose `contract` field is not an object must raise
    ValueError (the documented failure), not AttributeError from `.get`."""
    import json

    from roboto_to_lerobot.runtime.contract_io import load_contract, verify_manifest

    src = _write_minimal_contract(tmp_path)
    contract = load_contract(src)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"contract": "some/path.yaml"}))
    try:
        verify_manifest(manifest, contract)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on non-dict contract section")


def test_verify_manifest_refuses_when_sha_missing(tmp_path):
    """A manifest with no `contract.sha256` field cannot offer a drift
    guarantee — refusing is safer than passing silently."""
    import json

    from roboto_to_lerobot.runtime.contract_io import load_contract, verify_manifest

    src = _write_minimal_contract(tmp_path)
    contract = load_contract(src)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"contract": {}}))
    try:
        verify_manifest(manifest, contract)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on missing sha")


