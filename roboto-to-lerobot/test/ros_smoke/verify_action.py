"""Verifier for the docker bag-replay smoke's input-dependent stub.

Reads two YAML files captured by ``ros2 topic echo``:
  --obs-log    : a *continuous* capture of ``/robot/joint_states`` (one
                 YAML doc per message, separated by ``---``).
  --action-log : a single ``--once`` capture of ``/teleop/action``.

**Precondition (enforced by the caller, not this script).** The
captured action must come from the live generated node's publisher, not
from the bag's recorded teleop. ``run_in_container.sh`` ensures this
by remapping the bag's ``/teleop/action`` away
(``--remap /teleop/action:=/teleop/action_bag_recorded``) before the
echo subscribes. Without that remap, the verifier would compare the
bag's recorded teleop to the bag's obs and the obs[:8] + offset
invariant would fail — masking real policy bugs behind a false
negative, or, on a coincidentally-matching trajectory, false-passing.

The stub policy in ``ros_pkg/inference_node/policy.py`` computes
``action = obs["observation.state"][:_ACTION_WIDTH] + _STUB_OFFSET``.
The ``observation.state`` array that reaches the policy is
*selector-ordered*: ``LiveAdapter`` (see
``src/roboto_to_lerobot/runtime/live_adapter.py``) assembles it from
the contract's ``observations[].selector.names``, not from the bag's
``JointState.name[]`` order. So this verifier:

1. Reads the contract to discover the observation selector names.
2. For each captured ``/robot/joint_states`` message, reorders
   ``position[]`` by selector name (looking up each ``selector.name``
   in the message's ``name[]`` → ``position[]`` pairing).
3. Searches the reordered obs stream for the message whose first
   ``_ACTION_WIDTH`` selector entries best match
   ``action - _STUB_OFFSET``.

Indexing the bag's raw ``position[]`` by integer would assume the
recorder happened to write joints in declaration order; that pin is
fragile to fixture rotation and would silently false-pass on a
permuted bag.

The captured obs stream is also resilient to a truncated trailing
YAML document: ``ros2 topic echo`` is SIGTERMed mid-stream by the
harness, so the file's last record may be incomplete. The parser
iterates ``safe_load_all`` with per-doc error handling and stops on
the first ``YAMLError`` (only the trailing doc can ever be partial).

Exits 0 on success, 1 with a diagnostic listing the closest obs found.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Mirrored from ros_pkg/inference_node/policy.py — see the four-place
# pinning note there before changing either.
_ACTION_WIDTH = 8
_STUB_OFFSET = 100.0
# rad; sized to absorb fast end-effector drift over the DDS
# subscriber-discovery window. The bag publishes /robot/joint_states
# at ~50 Hz (sim) — wall-clock ~100 Hz under --rate 2 — so a discovery
# delay of 50–100 ms wall-clock costs the obs echo ~5–10 head frames.
# Across that window, end-effector joints moving ~1.5 rad/sec sim
# drift ~0.15–0.3 rad. Still ~5× tighter than realistic joint
# excursions (~±π): a "stale-default" bug where the policy never sees
# obs would produce per-joint deltas an order of magnitude beyond
# this and trip the gate. Empirically across 25 stability runs: 24
# delivered max |delta| ~4 µrad (full obs stream captured); two
# outliers landed at 0.126 and 0.152 rad (early frames lost to
# discovery, closest match offset by the gap above).
_TOLERANCE = 0.2

# Contract observation key that the stub policy reads
# (`obs["observation.state"]` in policy.py).
_OBS_KEY = "observation.state"


def _load_selector_names(contract_path: Path, section: str, key: str) -> list[str]:
    """Return the ``selector.names`` list for the contract entry matching
    ``key`` in ``section`` (e.g. ``observations``)."""
    with contract_path.open() as fh:
        contract = yaml.safe_load(fh)
    for entry in (contract or {}).get(section, []):
        if entry.get("key") == key:
            return list(entry["selector"]["names"])
    raise SystemExit(
        f"verify_action: FAIL — no entry in {contract_path} "
        f"{section!r} with key={key!r}"
    )


def _reorder_by_selector(
    doc: dict, selector_names: list[str]
) -> list[float] | None:
    """Return the doc's positions in ``selector_names`` order, or None
    if the doc is missing any selector name or has mismatched lengths.

    ``LiveAdapter`` builds ``obs["observation.state"]`` by selecting
    positions by name from the incoming JointState; this mirrors that
    so we compare what the policy actually saw, not what the recorder
    happened to write in declaration order.
    """
    names = doc.get("name")
    positions = doc.get("position")
    if not isinstance(names, list) or not isinstance(positions, list):
        return None
    if len(positions) < len(names):
        return None
    # strict=False: the guard above only rejects positions SHORTER than names;
    # a longer positions array is valid and is intentionally truncated here.
    name_to_pos = dict(zip(names, positions, strict=False))
    out: list[float] = []
    for sname in selector_names:
        if sname not in name_to_pos:
            return None
        out.append(float(name_to_pos[sname]))
    return out


def _read_obs_stream(
    yaml_path: Path, selector_names: list[str]
) -> list[list[float]]:
    """Parse the obs YAML stream into one selector-ordered position list
    per message. Documents missing any selector name, or with truncated
    YAML (the harness SIGTERMs the echo mid-stream), are skipped.
    """
    out: list[list[float]] = []
    with yaml_path.open() as fh:
        loader = yaml.safe_load_all(fh)
        while True:
            try:
                doc = next(loader)
            except StopIteration:
                break
            except yaml.YAMLError:
                # Only the trailing doc can be partial (SIGTERM mid-write);
                # parser state isn't reliably recoverable mid-error, so stop.
                break
            if not isinstance(doc, dict):
                continue
            ordered = _reorder_by_selector(doc, selector_names)
            if ordered is not None:
                out.append(ordered)
    return out


def _read_action_positions(yaml_path: Path) -> list[float] | None:
    """Parse the single-message action capture. Returns ``position[]`` as
    published; ``encode_action`` writes in selector order so integer
    indexing matches the policy's output positions. None if no message
    is parseable.
    """
    with yaml_path.open() as fh:
        try:
            for doc in yaml.safe_load_all(fh):
                if isinstance(doc, dict) and isinstance(doc.get("position"), list):
                    return [float(v) for v in doc["position"]]
        except yaml.YAMLError:
            pass
    return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Verify the input-dependent stub policy's invariant."
    )
    ap.add_argument("--obs-log", required=True, type=Path)
    ap.add_argument("--action-log", required=True, type=Path)
    ap.add_argument(
        "--contract",
        required=True,
        type=Path,
        help="Contract YAML; provides the observation selector names so we "
        "reorder the bag's raw JointState positions into the order the "
        "policy actually saw.",
    )
    args = ap.parse_args()

    selector_names = _load_selector_names(args.contract, "observations", _OBS_KEY)
    if len(selector_names) < _ACTION_WIDTH:
        print(
            f"verify_action: FAIL — contract's '{_OBS_KEY}' selector has "
            f"{len(selector_names)} names; need at least {_ACTION_WIDTH}",
            file=sys.stderr,
        )
        return 1

    action = _read_action_positions(args.action_log)
    if action is None:
        print(
            f"verify_action: FAIL — {args.action_log} has no parseable "
            "JointState message",
            file=sys.stderr,
        )
        return 1
    if len(action) != _ACTION_WIDTH:
        print(
            f"verify_action: FAIL — action.position has length "
            f"{len(action)}, expected {_ACTION_WIDTH}",
            file=sys.stderr,
        )
        return 1

    obs_stream = _read_obs_stream(args.obs_log, selector_names)
    if not obs_stream:
        print(
            f"verify_action: FAIL — {args.obs_log} has no obs message with a "
            f"complete selector set ({selector_names})",
            file=sys.stderr,
        )
        return 1

    best_obs: list[float] = []
    best_max_delta = float("inf")
    for obs in obs_stream:
        max_delta = max(
            abs(action[i] - (obs[i] + _STUB_OFFSET)) for i in range(_ACTION_WIDTH)
        )
        if max_delta < best_max_delta:
            best_max_delta = max_delta
            best_obs = obs

    if best_max_delta <= _TOLERANCE:
        print(
            f"verify_action: OK — action matches obs + {_STUB_OFFSET} within "
            f"{best_max_delta:.6f} rad "
            f"(searched {len(obs_stream)} observations, tolerance={_TOLERANCE}, "
            f"selector={selector_names[:_ACTION_WIDTH]})"
        )
        return 0

    expected = [best_obs[i] + _STUB_OFFSET for i in range(_ACTION_WIDTH)]
    print(
        f"verify_action: FAIL — no obs in the {len(obs_stream)}-msg stream "
        f"matches the invariant (best max |delta|={best_max_delta:.6f}, "
        f"tolerance={_TOLERANCE} rad).",
        file=sys.stderr,
    )
    print("  Closest obs (selector-ordered):", file=sys.stderr)
    for i in range(_ACTION_WIDTH):
        print(
            f"    [{selector_names[i]}]  obs={best_obs[i]:+.6f}  "
            f"action={action[i]:+.6f}  "
            f"expected={expected[i]:+.6f}  "
            f"delta={action[i] - expected[i]:+.6f}",
            file=sys.stderr,
        )
    print(f"  full closest obs (selector-ordered): {best_obs}", file=sys.stderr)
    print(f"  full action.position              : {action}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
