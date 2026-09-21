"""Stub policy module for the docker bag-replay smoke.

The generated node imports ``load_policy`` from whatever module
``--policy-module`` names; the smoke wires it to
``inference_node.policy`` (this file).

The returned callable maps the observation's ``observation.state``
vector through a deterministic transform — first ``_ACTION_WIDTH``
entries plus ``_STUB_OFFSET`` — so the action is *input-dependent*.
``verify_action.py`` captures the whole ``/robot/joint_states``
stream in parallel and searches it for the obs whose first
``_ACTION_WIDTH`` values are closest to ``action - _STUB_OFFSET``;
a pass requires a float-precision match in the common case, or a
match within the verifier's tolerance (~0.2 rad) when the obs echo's
DDS subscriber discovery loses the head of the stream. A runtime bug
that let the policy run on defaults (observations never reaching it)
would yield ``action == _STUB_OFFSET`` across the board, which no
real obs in the stream would match — the older zero-fingerprint check
could not catch that.

Action width 8 is pinned in FOUR coupled places: ``_ACTION_WIDTH``
here, ``contract.yaml``'s ``/teleop/action`` selector,
``_ACTION_WIDTH`` in ``verify_action.py``, and the four-place note
in ``run_in_container.sh``. ``_STUB_OFFSET`` is mirrored in
``verify_action.py`` (with the verifier tolerance). All change
together if the contract widens or the transform is altered.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_ACTION_WIDTH = 8  # mirrored in contract.yaml + verify_action.py + run_in_container.sh
_STUB_OFFSET = 100.0  # mirrored in verify_action.py


def _policy(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    # Input-dependent stub: action = obs.observation.state[:8] + 100.0.
    # verify_action.py searches the captured obs stream for the obs
    # the policy actually sampled, rather than relying on tolerance
    # against a single time-misaligned obs. A single --once capture
    # failed because end-effector joints drift ~1 rad/sec, and the
    # one captured frame was several hundred ms away from the frame
    # the policy used at its first tick — far beyond any defensible
    # tolerance. encode_action down-casts the action through float32
    # (live_adapter) before publishing it as JointState (float64 on
    # the wire per runtime/encoders), so the policy never controls
    # the serialized type.
    state = obs["observation.state"]
    return {"action": state[:_ACTION_WIDTH] + _STUB_OFFSET}


def load_policy(_path: Any):
    """Return the stub policy callable.

    Signature mirrors the user-side ``load_policy(path)`` that real
    policies expose; ``path`` is whatever ``policy_path`` parameter the
    node was started with, ignored here.
    """
    return _policy
