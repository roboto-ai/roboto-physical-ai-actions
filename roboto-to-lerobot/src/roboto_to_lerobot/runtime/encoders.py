"""Action encoders: the inverse of the per-type decoders' field-selection step.

A live-inference node hands a policy output (numpy array, shape and
ordering dictated by ``ActionSpec.selector.names``) back through one of
these encoders to produce a ROS message payload dict. The dict shape is
chosen so that downstream wiring (``LiveAdapter.encode_action``)
can construct concrete ROS messages with ``MessageType(**payload)`` once
rclpy is in the picture — keeping this module rclpy-free means it stays
importable inside the converter image and the CI test environment.

Scope of the "inverse": these encoders invert the decoder's selector-driven
field extraction only. They do NOT invert any ``ActionSpec.transforms`` the
converter applies on the decode path, so ``encode(decode(msg))`` reproduces
the original wire values only for transform-free specs. Transform inversion
is not implemented.

Coverage:

* ``sensor_msgs/msg/JointState`` — inverts the converter's selector-driven
  decoder for the ``position`` field. Any selector that asks for
  ``velocity``/``effort`` round-trips through decode but cannot be encoded
  without a richer action-space model, which is not implemented.
* ``std_msgs/msg/Float64MultiArray`` — generic vector encoder.
* ``std_msgs/msg/Float64`` — scalar action.

Exotic action types (Twist, JointTrajectory, custom messages) are
explicitly out of scope and will refuse to encode at runtime — the
live adapter surfaces that as a contract error. Add them when a real
contract needs them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..contract_utils import ActionSpec


EncoderFn = Callable[[np.ndarray, "ActionSpec"], dict[str, Any]]


__all__ = ("ENCODERS", "EncoderFn", "encode_value", "register_encoder")


# Registry populated by ``@register_encoder``. Keys are ROS message type
# strings; values are the encoder callables. Parallel structure to
# ``runtime.converters.DECODERS`` — kept as a separate dict because
# encoders and decoders are independently discoverable; not every type
# needs both directions.
ENCODERS: dict[str, EncoderFn] = {}


def register_encoder(type_str: str, dtype: str):
    """Register an encoder for a ROS message type.

    ``dtype`` is declarative only, mirroring ``register_decoder``: it states
    the numeric dtype the encoder expects on the input array, but the value
    is not stored on the registry and no caller reads it back today (kept as
    a documentation hook for a future decoder/encoder dtype-agreement check).

    Example:
        @register_encoder("sensor_msgs/msg/JointState", dtype="float64")
        def encode_joint_state(action, spec):
            return {"name": [...], "position": [...]}
    """

    def _wrap(fn: EncoderFn):
        ENCODERS[type_str] = fn
        return fn

    return _wrap


def encode_value(action: np.ndarray, spec: ActionSpec) -> dict[str, Any]:
    """Encode an action array using the registered encoder for ``spec.type``.

    Raises ``ValueError`` if no encoder is registered for the message type —
    the live adapter surfaces this as a contract-time error so a malformed
    contract fails fast rather than silently dropping actions.
    """
    fn = ENCODERS.get(spec.type)
    if not fn:
        raise ValueError(
            f"No encoder registered for message type '{spec.type}' "
            f"(action key '{spec.key}')."
        )
    return fn(action, spec)


# =============================================================================
# JointState — inverts runtime.decoders._dec_joint_state (position field only)
# =============================================================================


@register_encoder("sensor_msgs/msg/JointState", dtype="float64")
def _enc_joint_state(action: np.ndarray, spec: ActionSpec) -> dict[str, Any]:
    """Encode a JointState action.

    ``action`` is a 1-D array whose entries align positionally with
    ``spec.selector.names``. Each selector name may be either a bare
    joint name (interpreted as the ``position`` field, matching the
    decoder) or ``position.<joint>`` explicitly. Other fields
    (``velocity``, ``effort``) are rejected — encoding them requires a
    full per-joint mapping that the single-position action vector
    cannot supply.

    Returns ``{"name": [...], "position": [...]}`` — the minimal payload
    the LiveAdapter needs to construct a ``sensor_msgs/msg/JointState``.
    """
    selector_names = (spec.selector or {}).get("names", [])
    if not selector_names:
        raise ValueError(
            f"JointState encoder requires spec.selector.names to map action "
            f"vector entries to joint names (action key '{spec.key}', "
            f"type '{spec.type}')."
        )

    flat = np.asarray(action, dtype=np.float64).reshape(-1)
    if flat.shape[0] != len(selector_names):
        raise ValueError(
            f"Action vector length {flat.shape[0]} does not match selector "
            f"names length {len(selector_names)} (action key '{spec.key}', "
            f"type '{spec.type}')."
        )

    names = []
    positions = []
    for selector, value in zip(selector_names, flat, strict=True):
        if "." in selector:
            field, joint_name = selector.split(".", 1)
            if field != "position":
                raise ValueError(
                    f"JointState encoder only supports the 'position' field; "
                    f"selector '{selector}' requests '{field}' (action key "
                    f"'{spec.key}', type '{spec.type}')."
                )
        else:
            joint_name = selector
        names.append(joint_name)
        positions.append(float(value))

    return {"name": names, "position": positions}


# =============================================================================
# Generic vector / scalar encoders
# =============================================================================


@register_encoder("std_msgs/msg/Float64MultiArray", dtype="float64")
def _enc_float64_multiarray(action: np.ndarray, spec: ActionSpec) -> dict[str, Any]:
    """Encode a vector action as Float64MultiArray.

    The contract's selector names are not consulted — the array is shipped
    as ``{"data": [...]}`` in declared order. Callers that care about per-
    channel labelling should pick a structured message type (JointState).
    """
    flat = np.asarray(action, dtype=np.float64).reshape(-1)
    return {"data": [float(v) for v in flat]}


@register_encoder("std_msgs/msg/Float64", dtype="float64")
def _enc_float64(action: np.ndarray, spec: ActionSpec) -> dict[str, Any]:
    """Encode a scalar action as Float64.

    Accepts a 0-D array, a 1-element 1-D array, or a Python scalar — the
    live-adapter does not promise any particular shape on the way back
    from the policy, but the wire format is one float.
    """
    arr = np.asarray(action, dtype=np.float64)
    flat = arr.reshape(-1)
    if flat.shape[0] != 1:
        raise ValueError(
            f"Float64 encoder expects a single-element action, got shape "
            f"{arr.shape} (action key '{spec.key}', type '{spec.type}')."
        )
    return {"data": float(flat[0])}
