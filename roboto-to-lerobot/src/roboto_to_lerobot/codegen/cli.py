"""``roboto-to-lerobot`` CLI — currently exposes only the ``gen-node`` subcommand.

This is the package's only executable surface. Wiring goes through
``[project.scripts] roboto-to-lerobot = "roboto_to_lerobot.codegen.cli:main"``
in ``pyproject.toml``.

The CLI is intentionally thin: parse args, load the contract through
the runtime facade, run codegen-time validations (a superset of the
runtime's boot-time gates — align methods, transforms, decoders,
encoders, safety behavior, selector widths), and hand off to the
renderer. The validations fail before any source file is touched, so
a user sees an unsupported-feature message at gen-node time rather
than from a crash inside the live node.

Refuse-list location. ``REFUSED_TRANSFORMS`` and
``REFUSED_ALIGN_METHODS`` are hardcoded constants rather than metadata
carried on a registry alongside the transforms themselves. The
hardcoded approach is deliberate: the refusal is operator-facing and
stable, the list is short, and storing it next to the codegen step
keeps every "why won't this contract gen?" answer in one file.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from pathlib import Path

from ..runtime import decoders as _decoders  # noqa: F401 — populate DECODERS
from ..runtime.contract_io import Contract, load_contract
from ..runtime.converters import DECODERS
from ..runtime.encoders import ENCODERS
from ..video import is_compressed_video_schema
from .render import RenderError, render_node

# The one transform the live runtime can run: it's the only transform in
# the registry that's causal (forward-only), so it's the only one that can
# be reproduced sample-by-sample as messages arrive. Kept as a module
# constant so this file and ``runtime/live_adapter.py``'s ``_LIVE_TRANSFORM``
# are visibly reading the same name in a diff — this gate must never
# *accept* something the adapter would refuse.
LIVE_TRANSFORM: str = "butterworth_lowpass_causal"

# Every other transform is refused. This set gets a tailored "non-causal"
# message (they need future samples, so they can never run live even once
# more transform support lands); anything outside both this set and
# ``LIVE_TRANSFORM`` gets the generic refusal.
# TODO: swap for ``TransformEntry.refuse_in_runtime`` metadata once a
# transform registry carries it. Hardcoded here so a contract author
# gets a refusal at gen-node time rather than a silent drop or a
# confused live-runtime error several layers down.
REFUSED_TRANSFORMS: frozenset[str] = frozenset({
    "butterworth_lowpass",
    "resample_uniform",
})

# Alignment methods ``StreamBuffer`` does not implement. Listed
# here (not in ``runtime.stream_buffer``) so the refusal lands at
# codegen time, before any source file is written; the runtime would
# also refuse, but later in the workflow.
REFUSED_ALIGN_METHODS: frozenset[str] = frozenset({"linear", "none"})


class GenNodeError(Exception):
    """Raised when a contract cannot be turned into a generated node.

    Carries a single operator-facing message; the CLI catches it, prints
    the message to stderr, and exits non-zero. Distinct from generic
    ``ValueError`` so callers (and tests) can match on type.
    """


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_contract_for_codegen(contract: Contract) -> None:
    """Refuse contracts that use features the live runtime rejects.

    These refusals are a conservative *superset* of ``LiveAdapter``'s
    boot-time gates: everything the adapter would reject is rejected here
    too — moved forward so the user sees the message *before* any
    source file is written — which gives the one-directional guarantee
    that a contract gen-node accepts will boot. Codegen also refuses a
    few node-level shapes the adapter's gates can't see (it builds
    buffers, it doesn't import/subscribe ROS types), so it never
    *accepts* something the runtime rejects.

    - alignment method on observations/videos: the ``StreamBuffer`` only
      implements ``hold`` and ``nearest`` (no ``linear`` / ``none``).
      Actions are *not* checked — the runtime never aligns the action
      path (see :meth:`LiveAdapter._build_action_specs`).
    - transforms on any stream: ``LiveAdapter._validate_transforms`` allows
      only ``butterworth_lowpass_causal`` — the sole transform in the
      registry that's causal and so the only one reproducible sample-by-
      sample live — and refuses everything else on obs/actions. We mirror
      that here, plus the same ``fs_hz`` requirements the adapter enforces
      on observations (a ``stage: pre`` causal transform needs a declared
      ``fs_hz``; a ``stage: post`` one may not declare an ``fs_hz`` that
      contradicts ``contract.fps``). Actions accept the causal transform
      unconditionally — the runtime passes the policy's action output
      through unfiltered (see ``LiveAdapter.encode_action``'s docstring),
      so there's no ``fs_hz`` to validate. Videos never accept any
      transform, causal or not (the runtime would silently ignore a video
      transform; refusing is safer).
    - observation/video decoder: mirror ``LiveAdapter._make_buffer_entry``
      so a type with no registered decoder refuses here. We additionally
      refuse file-backed/offline-only types (e.g. ``video``) that the
      adapter's decoder gate lets through but the generated node has no
      ROS message class to import or subscribe with, and compressed-video
      types, whose registered decoder only passes through frames the offline
      converter decoded a whole GOP at a time.
    - action safety_behavior / encoder / multi-spec selector widths.

    Raises:
        GenNodeError: with a message naming the offending stream key,
            the unsupported feature, and what to use instead.
    """
    for spec in contract.observations:
        _refuse_align_method(spec.key, spec.align.method)
        _refuse_transforms(spec.key, spec.transforms, fps=contract.fps, is_action=False)
        _refuse_unsupported_observation(spec.key, spec.topic, spec.type)

    for spec in contract.videos:
        _refuse_align_method(spec.key, spec.align.method)
        _refuse_video_transforms(spec.key, spec.transforms)
        _refuse_unsupported_observation(spec.key, spec.topic, spec.type)

    for spec in contract.actions:
        # No align check: the runtime never aligns actions, so refusing
        # an action's align.method would over-refuse a contract the
        # runtime accepts.
        _refuse_transforms(spec.key, spec.transforms, fps=contract.fps, is_action=True)
        _refuse_safety_behavior(spec.key, spec.safety_behavior)
        _refuse_missing_encoder(spec.key, spec.topic, spec.type)

    _refuse_multi_spec_without_selector_names(contract)


def _refuse_align_method(key: str, method: str) -> None:
    if method in REFUSED_ALIGN_METHODS:
        raise GenNodeError(
            f"Stream '{key}' uses align.method='{method}', which is not "
            "supported. The StreamBuffer implements 'hold' and 'nearest' "
            "only; 'linear' and 'none' are not implemented. Change the "
            "contract to one of the supported methods or wait for the "
            "v2 release."
        )


def _refuse_transforms(key: str, transforms, *, fps: float, is_action: bool) -> None:
    """Mirror ``LiveAdapter._validate_transforms`` for observations/actions.

    Allows only ``LIVE_TRANSFORM`` (``butterworth_lowpass_causal``) — the
    one transform in the registry that's causal and so the only one
    reproducible sample-by-sample live. Everything else refuses: the
    non-causal set in ``REFUSED_TRANSFORMS`` gets a tailored message (they
    need future samples, so they can never run live even once more
    transform support lands); anything outside both sets gets the generic
    refusal below.

    For observations, the causal transform additionally must declare
    ``fs_hz`` when ``stage: pre`` (the raw topic rate training measured is
    otherwise unknowable to the live runtime), and must not declare an
    ``fs_hz`` that contradicts ``fps`` when ``stage: post`` (post-stage
    transforms always run on the contract-fps timeline). Actions accept
    the causal transform unconditionally — no ``fs_hz`` check — because
    the runtime passes the policy's action output through unfiltered
    rather than designing a live filter for it.
    """
    if not transforms:
        return
    for transform in transforms:
        if transform.type == LIVE_TRANSFORM:
            if is_action:
                continue  # pass-through — see LiveAdapter.encode_action.
            if transform.stage == "pre" and "fs_hz" not in transform.params:
                raise GenNodeError(
                    f"Stream '{key}' uses stage:pre '{LIVE_TRANSFORM}' "
                    "without 'fs_hz'. The live runtime cannot know the "
                    "training-measured raw topic rate — declare fs_hz in "
                    "the contract so training and deployment design "
                    "identical filter coefficients."
                )
            if transform.stage == "post":
                fs_hz = transform.params.get("fs_hz")
                if fs_hz is not None and not math.isclose(
                    float(fs_hz), fps, rel_tol=1e-9, abs_tol=1e-9
                ):
                    raise GenNodeError(
                        f"Stream '{key}' uses stage:post '{LIVE_TRANSFORM}' "
                        f"with fs_hz={fs_hz}, which contradicts "
                        f"contract.fps={fps}. Post-stage transforms run on "
                        "the contract-fps reference timeline; drop fs_hz "
                        "or set it to contract.fps."
                    )
            continue
        if transform.type in REFUSED_TRANSFORMS:
            raise GenNodeError(
                f"Stream '{key}' uses transform '{transform.type}', "
                "which gen-node refuses to codegen: it is non-causal "
                "(needs future samples) and cannot run faithfully in a "
                f"live node. Retrain with a causal preprocessing step "
                f"(e.g. '{LIVE_TRANSFORM}', which the live runtime can "
                "run directly), or wait for --allow-causal-approximation "
                "in the v2 release."
            )
        raise GenNodeError(
            f"Stream '{key}' declares transform '{transform.type}', which "
            f"the live runtime cannot run: only '{LIVE_TRANSFORM}' is "
            "supported live. LiveAdapter refuses everything else at boot; "
            "gen-node refuses here so no node file is written."
        )


def _refuse_video_transforms(key: str, transforms) -> None:
    """Videos never accept transforms — mirror ``LiveAdapter._validate_transforms``.

    Unlike observations/actions, videos refuse *every* transform,
    including the causal low-pass: the runtime treats video frames as
    opaque decoded arrays, and the converter itself only applies
    transforms to observation/action vector streams.
    """
    if not transforms:
        return
    raise GenNodeError(
        f"Stream '{key}' is a video/image stream declaring "
        f"{len(transforms)} transform(s); the live runtime never applies "
        "transforms to video/image streams, causal or not."
    )


def _refuse_unsupported_observation(key: str, topic: str, type_str: str) -> None:
    """Mirror ``LiveAdapter._make_buffer_entry``'s decoder gate at codegen time.

    Three ways an observation/video type fails to run live:

    1. No decoder is registered for it — the adapter refuses at boot.
    2. The type is a file-backed/offline-only shape (``video`` /
       ``avi_video`` / ``mp4_video`` / ``string_typed_msg``) that *is*
       in ``DECODERS`` but has no ``pkg/msg/Type`` ROS message class, so
       the generated node has nothing to ``import`` or subscribe with.
       These only appear in offline-conversion contracts.
    3. The type is compressed video. It has both a real ROS message class and
       a registered decoder, but that decoder is a pass-through over frames the
       offline converter already GOP-decoded — a live node keeps no decoder
       state across messages and cannot turn one access unit into a frame.

    Either way the file gen-node would write cannot boot, so refuse here.
    """
    if is_compressed_video_schema(type_str):
        raise GenNodeError(
            f"Observation/video '{key}' (topic={topic}) declares compressed-video type "
            f"'{type_str}'. Each message is one encoded video access unit, decodable only "
            "together with the preceding frames of its group of pictures, so a live node "
            "cannot decode it message by message. Compressed video is supported for "
            "offline conversion only; subscribe to a per-frame image topic "
            "(sensor_msgs/msg/CompressedImage or sensor_msgs/msg/Image) for live inference."
        )
    if "/" not in type_str:
        raise GenNodeError(
            f"Observation/video '{key}' (topic={topic}) declares type "
            f"'{type_str}', which is a file-backed/offline-only shape with "
            "no live ROS message class. The generated node subscribes to a "
            "real ROS topic, so it cannot consume this type. Use a "
            "'pkg/msg/Type' message type for live inference."
        )
    if type_str not in DECODERS:
        raise GenNodeError(
            f"Observation/video '{key}' (topic={topic}) declares type "
            f"'{type_str}', but gen-node has no decoder registered for it. "
            f"Supported types: {sorted(t for t in DECODERS if '/' in t)}. "
            "Other types are not supported."
        )


def _refuse_safety_behavior(key: str, behavior: str) -> None:
    if behavior == "publish_nothing":
        return
    raise GenNodeError(
        f"Action '{key}' sets safety_behavior='{behavior}', which is "
        "not supported. The generated node only implements "
        "'publish_nothing' (downstream controller times out and applies "
        "its own stop). 'hold_last' and 'safe_pose' are not implemented."
    )


def _refuse_missing_encoder(key: str, topic: str, type_str: str) -> None:
    """Mirror ``LiveAdapter.__init__``'s encoder check at codegen time.

    The adapter would refuse a missing encoder at boot — moving the
    refusal to ``gen-node`` time means the user sees the
    unsupported-type message before the generated file is written,
    not after they ``ros2 run`` it. Encoder coverage is intentionally
    narrow (``JointState``, ``Float64MultiArray``, ``Float64``); other
    types refuse with the registered list so the user knows what to
    substitute.
    """
    if type_str in ENCODERS:
        return
    raise GenNodeError(
        f"Action '{key}' (topic={topic}) declares type '{type_str}', "
        f"but gen-node has no encoder registered for it. Supported "
        f"action types: {sorted(ENCODERS)}. Other types are not "
        "supported."
    )


def _refuse_multi_spec_without_selector_names(contract: Contract) -> None:
    """Multi-spec action keys need ``selector.names`` on every spec.

    ``LiveAdapter.encode_action`` slices the policy's per-key ndarray
    by ``len(selector.names)`` in declaration order when a base key
    fans out across topics. A missing selector means the adapter has
    no width to slice with; it refuses at ``__init__`` time. Catching
    it here makes the refusal visible at ``gen-node`` instead of
    several layers down inside ``LiveAdapter``.
    """
    by_key: dict[str, list] = {}
    for spec in contract.actions:
        by_key.setdefault(spec.key, []).append(spec)

    for key, group in by_key.items():
        if len(group) <= 1:
            continue
        for spec in group:
            names = (spec.selector or {}).get("names")
            if not names:
                raise GenNodeError(
                    f"Action base key '{key}' has {len(group)} specs (one per topic) "
                    f"but spec (topic={spec.topic}) is missing 'selector.names'. "
                    "Multi-topic actions need selector names on every spec so the "
                    "live adapter can slice the policy output across topics."
                )


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser.

    Single subcommand today (``gen-node``); structured as a subcommand
    tree from day 1 so additions (``gen-launch``, ``selfcheck``, etc.)
    can land without breaking the executable's name.
    """
    parser = argparse.ArgumentParser(
        prog="roboto-to-lerobot",
        description="Roboto → LeRobot tools, including the live-runtime codegen.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser(
        "gen-node",
        help="Generate a runnable rclpy.Node from a contract.yaml.",
    )
    gen.add_argument(
        "contract",
        type=Path,
        help="Path to the contract YAML.",
    )
    gen.add_argument(
        "--policy-module",
        required=True,
        help=(
            "Dotted Python import path that exposes ``load_policy(path)``. "
            "The generated node imports ``load_policy`` from this module."
        ),
    )
    gen.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help=(
            "Path to the converter-produced manifest.json. Used at "
            "boot to verify the contract SHA the manifest was produced "
            "against matches the contract the node loaded."
        ),
    )
    gen.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Where to write the generated node file.",
    )
    gen.add_argument(
        "--force",
        action="store_true",
        help="Overwrite --out if it exists (step 4).",
    )
    gen.add_argument(
        "--node-name",
        default=None,
        help=(
            "Python class name for the generated node "
            "(default: derived from contract.name)."
        ),
    )

    return parser


def gen_node_command(args: argparse.Namespace) -> int:
    """Validate a contract, render the node, and write it to ``--out``.

    Refuses to overwrite an existing ``--out`` unless ``--force`` is
    passed. There is no three-way merge for regen (no ``--upgrade``
    mode); the refusal is the safety net that prevents an accidental
    ``gen-node`` from clobbering hand-edits made on top of an earlier
    rendering.
    """
    contract = load_contract(args.contract)
    validate_contract_for_codegen(contract)
    if args.out.exists() and not args.force:
        raise GenNodeError(
            f"Refusing to overwrite existing {args.out}. Pass --force to "
            "overwrite, or generate to a new path and merge manually."
        )
    source = render_node(
        contract,
        policy_module=args.policy_module,
        manifest_path=args.manifest,
        node_class_name=args.node_name,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(source, encoding="utf-8")
    print(
        f"gen-node: wrote {args.out} ({len(source.splitlines())} lines) "
        f"from contract '{contract.name}'.",
        file=sys.stderr,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point — returns a Unix-style exit code.

    ``argv`` is exposed for testing; ``main()`` with no argument falls
    back to ``sys.argv[1:]`` like a normal entry point.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "gen-node":
        try:
            return gen_node_command(args)
        except (GenNodeError, RenderError) as e:
            print(f"gen-node: {e}", file=sys.stderr)
            return 2

    # argparse with ``required=True`` on the subparsers already
    # rejects this branch, but keep an explicit fallback so a future
    # subcommand miswire surfaces as a real error.
    parser.error(f"Unknown command: {args.command}")
    return 1  # unreachable; parser.error raises


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
