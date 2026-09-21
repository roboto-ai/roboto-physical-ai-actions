"""Live-runtime adapter: messages in, observation tensors out.

A generated ROS 2 node owns one :class:`LiveAdapter` and routes every
subscription callback through :meth:`on_message`; on each policy tick it
calls :meth:`sample` to get the dict the policy expects. :meth:`encode_action`
turns a policy output back into ``(topic, payload)`` pairs the node publishes.

The adapter holds no rclpy objects, so it runs under stock Python with
synthetic messages and the generated node stays the only place rclpy lives.
"rclpy-free" is the only isolation claim here: importing this module still
pulls the converter's heavy deps (numpy, pandas, cv2, roboto, yaml) through
``.image`` / ``.contract_io`` / ``..contract_utils``. Trimming that
dependency surface is future work.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..contract_utils import ActionSpec, ObservationSpec
from ..logger import logger

# Side-effect imports — populate the decoder/encoder registries used
# below. Mirrors how ``contract_utils`` triggers decoder registration via
# its top-level ``from . import decoders`` in the converter pipeline; the
# encoder registry has no other entry point, so importing it here ensures
# the live runtime always sees a populated ``ENCODERS`` dict regardless of
# whether the application code touched ``runtime.encoders`` first.
from . import decoders as _decoders  # noqa: F401
from . import encoders as _encoders  # noqa: F401
from .causal_lowpass import CausalLowpass
from .contract_io import Contract
from .converters import DECODERS, decode_value
from .encoders import ENCODERS, encode_value
from .image import resize_image
from .stream_buffer import StreamBuffer, auto_bound_tolerance_ns

# The only transform the live runtime can reproduce sample-by-sample. Every
# other registered transform (``butterworth_lowpass``, ``resample_uniform``,
# ``finite_difference``, ...) either needs future samples (non-causal) or
# changes the time axis, neither of which a single-pass, one-message-at-a-
# time adapter can do. Kept as a module constant (rather than inline string
# literals scattered through the gate below) so ``codegen/cli.py``'s mirror
# gate and this one are visibly reading the same name in a diff.
_LIVE_TRANSFORM = "butterworth_lowpass_causal"


# ``BufferEntry`` pairs the originating spec with the per-stream buffer.
# Kept as a tuple rather than a dataclass to stay introspectable in tests
# without adding another import surface.
BufferEntry = tuple[ObservationSpec, StreamBuffer]


class LiveAdapter:
    """Stateful glue between ROS callbacks and a policy.

    The adapter does not own any rclpy objects — the generated node
    creates publishers/subscriptions and forwards messages here. This
    keeps the adapter testable in CI without a ROS install and lets a
    future ``ReplayAdapter`` (step 4) reuse the same instance against a
    bag file.

    Threading: not thread-safe. ``on_message`` mutates the per-stream
    buffers that ``sample`` iterates, with no internal lock, so the
    generated node MUST drive a single :class:`LiveAdapter` from a
    single-threaded executor (callbacks and the tick loop on one thread).
    Under a ``MultiThreadedExecutor`` a callback ``push`` concurrent with
    a tick ``sample`` can raise ``RuntimeError: deque mutated during
    iteration``. ``ReplayAdapter`` is single-threaded by construction.

    Args:
        contract: a :class:`runtime.contract_io.Contract` (the facade,
            not the underlying ``contract_utils.Contract`` dataclass);
            type-hinting the facade matches what generated nodes pass.
    """

    def __init__(self, contract: Contract) -> None:
        self._contract = contract
        self._fps = float(contract.fps)

        # Buffers grouped two ways: by topic (for ``on_message``
        # dispatch — one ROS message may feed multiple specs) and by
        # base key (for ``sample`` — multi-topic observations under the
        # same key concat in declaration order, matching
        # ``Contract.get_lerobot_features``).
        self._topic_entries: dict[str, list[BufferEntry]] = {}
        self._obs_key_entries: dict[str, list[BufferEntry]] = {}
        self._video_key_entries: dict[str, BufferEntry] = {}

        # Action groups: base_key → list of specs in declaration order.
        # Each spec is paired with the encoder resolved at __init__ so a
        # missing encoder surfaces at adapter boot, not at first tick.
        self._action_groups: dict[str, list[ActionSpec]] = {}

        self._validate_transforms(contract)
        # Causal low-pass filters, keyed by ``ObservationSpec.unique_key``
        # (topic-qualified, so two specs sharing a base key never collide).
        # Built before the buffers below so a Nyquist-invalid ``cutoff_hz``
        # (raised from ``CausalLowpass.__init__``) fails at adapter boot,
        # matching the "fail at __init__, not first tick" contract every
        # other gate in this class follows.
        self._pre_filters: dict[str, list[CausalLowpass]] = {}
        self._post_filters: dict[str, list[CausalLowpass]] = {}
        self._build_pre_filters(contract.observations)
        self._build_post_filters(contract.observations)
        self._build_observation_buffers(contract.observations)
        self._build_video_buffers(contract.videos)
        self._build_action_specs(contract.actions)

    def _validate_transforms(self, contract: Contract) -> None:
        """Refuse contracts that use transforms the live runtime cannot honour.

        Product decision: only ``butterworth_lowpass_causal`` may run live.
        Every other registered transform (``butterworth_lowpass``,
        ``resample_uniform``, ``finite_difference``, ...) either needs
        future samples (non-causal) or changes the time axis — neither of
        which a single-pass, one-message-at-a-time adapter can do.
        ``codegen/cli.py``'s ``_refuse_transforms`` mirrors this gate as a
        conservative superset: everything refused here must also be
        refused there, so a contract ``gen-node`` accepts is guaranteed to
        boot.

        Video specs never accept transforms, causal or not — the runtime
        treats video frames as opaque decoded arrays, and the converter
        only ever applies transforms to observation/action *vector*
        streams.

        On observations, ``butterworth_lowpass_causal`` additionally
        requires:
          - ``stage: pre`` → a declared ``fs_hz`` param. Pre-stage
            transforms run on the raw stream at whatever rate training
            happened to measure (``lerobot.py``'s ``raw_fs``) — a number
            the live runtime has no way to reconstruct on its own.
            Declaring ``fs_hz`` in the contract is what lets training and
            deployment design *identical* filter coefficients; see
            ``transforms.py``'s ``_butterworth_lowpass_causal`` docstring
            for the full parity argument.
          - ``stage: post`` → if ``fs_hz`` is declared, it must equal
            ``contract.fps`` (within float tolerance). Post-stage
            transforms always run on the contract-fps reference timeline,
            so a differing ``fs_hz`` is a contradiction in the contract —
            refuse it rather than silently pick one interpretation.

        Actions accept ``butterworth_lowpass_causal`` unconditionally, with
        no ``fs_hz`` requirement: :meth:`encode_action` never re-applies
        it to the policy's output (see that method's docstring for why a
        pass-through is correct), so there is no live filter to design
        coefficients for.
        """
        for spec in contract.videos:
            if spec.transforms:
                raise ValueError(
                    f"Spec key={spec.key!r} topic={spec.topic!r} is a "
                    f"video/image stream declaring {len(spec.transforms)} "
                    "transform(s); the live runtime never applies transforms "
                    "to video/image streams."
                )

        for spec in contract.observations:
            self._validate_stream_transforms(spec, is_action=False)
        for spec in contract.actions:
            self._validate_stream_transforms(spec, is_action=True)

    def _validate_stream_transforms(
        self, spec: ObservationSpec | ActionSpec, *, is_action: bool
    ) -> None:
        for transform in spec.transforms or []:
            if transform.type != _LIVE_TRANSFORM:
                raise ValueError(
                    f"Spec key={spec.key!r} topic={spec.topic!r} declares "
                    f"transform {transform.type!r}, which the live runtime "
                    f"cannot run: only {_LIVE_TRANSFORM!r} can be reproduced "
                    "sample-by-sample online. Retrain with a causal "
                    "preprocessing step, or drop the transform for live "
                    "deployment."
                )
            if is_action:
                # Pass-through — see encode_action's docstring. Actions
                # never get an fs_hz requirement because the runtime never
                # designs (or applies) a filter for the action path.
                continue
            if transform.stage == "pre" and "fs_hz" not in transform.params:
                raise ValueError(
                    f"Spec key={spec.key!r} topic={spec.topic!r} declares a "
                    f"stage:pre {_LIVE_TRANSFORM!r} transform without "
                    "'fs_hz'. The live runtime cannot know the training-"
                    "measured raw topic rate — declare fs_hz in the "
                    "contract so training and deployment design identical "
                    "filter coefficients."
                )
            if transform.stage == "post":
                fs_hz = transform.params.get("fs_hz")
                if fs_hz is not None and not math.isclose(
                    float(fs_hz), self._fps, rel_tol=1e-9, abs_tol=1e-9
                ):
                    raise ValueError(
                        f"Spec key={spec.key!r} topic={spec.topic!r} "
                        f"declares a stage:post {_LIVE_TRANSFORM!r} "
                        f"transform with fs_hz={fs_hz}, which contradicts "
                        f"contract.fps={self._fps}. Post-stage transforms "
                        "run on the contract-fps reference timeline; drop "
                        "fs_hz or set it to contract.fps."
                    )

    # ------------------------------------------------------------------
    # Causal filter construction (observation path only — see
    # encode_action for why actions are pass-through)
    # ------------------------------------------------------------------

    def _build_pre_filters(self, specs: list[ObservationSpec]) -> None:
        """One :class:`CausalLowpass` per stage:pre causal transform.

        Designed with the contract's declared ``fs_hz`` (validated present
        by ``_validate_stream_transforms``) — the training-measured raw
        rate, not whatever the live topic happens to publish at. See the
        module-level parity discussion in ``causal_lowpass.py``.
        """
        for spec in specs:
            filters = [
                CausalLowpass(
                    cutoff_hz=float(t.params["cutoff_hz"]),
                    order=int(t.params.get("order", 2)),
                    fs=float(t.params["fs_hz"]),
                )
                for t in (spec.transforms or [])
                if t.type == _LIVE_TRANSFORM and t.stage == "pre"
            ]
            if filters:
                self._pre_filters[spec.unique_key] = filters

    def _build_post_filters(self, specs: list[ObservationSpec]) -> None:
        """One :class:`CausalLowpass` per stage:post causal transform.

        Designed with ``contract.fps`` — post-stage transforms run on the
        aligned reference timeline, and ``_validate_stream_transforms``
        already refused any declared ``fs_hz`` that contradicts it.
        """
        for spec in specs:
            filters = [
                CausalLowpass(
                    cutoff_hz=float(t.params["cutoff_hz"]),
                    order=int(t.params.get("order", 2)),
                    fs=self._fps,
                )
                for t in (spec.transforms or [])
                if t.type == _LIVE_TRANSFORM and t.stage == "post"
            ]
            if filters:
                self._post_filters[spec.unique_key] = filters

    # ------------------------------------------------------------------
    # Buffer construction
    # ------------------------------------------------------------------

    def _build_observation_buffers(
        self, specs: list[ObservationSpec]
    ) -> None:
        for spec in specs:
            entry = self._make_buffer_entry(spec, is_video=False)
            self._topic_entries.setdefault(spec.topic, []).append(entry)
            self._obs_key_entries.setdefault(spec.key, []).append(entry)

    def _build_video_buffers(self, specs: list[ObservationSpec]) -> None:
        for spec in specs:
            if spec.key in self._video_key_entries:
                # ``load_contract`` already rejects duplicate video keys
                # at parse time; belt-and-suspenders so an internally
                # constructed Contract can't silently overwrite.
                raise ValueError(
                    f"Duplicate video spec key '{spec.key}' — videos must be 1:1 with key."
                )
            entry = self._make_buffer_entry(spec, is_video=True)
            self._topic_entries.setdefault(spec.topic, []).append(entry)
            self._video_key_entries[spec.key] = entry

    def _make_buffer_entry(
        self, spec: ObservationSpec, *, is_video: bool
    ) -> BufferEntry:
        if spec.type not in DECODERS:
            # Symmetric with the encoder guard in ``_build_action_specs`` —
            # surface a contract that names an undecodable observation/video
            # type at boot rather than at the first ``on_message``. (This
            # presence check does not assert live-readiness: some registered
            # decoders still assume the converter's pandas-row input shape;
            # reconciling that with real ROS messages is future work.)
            raise ValueError(
                f"No decoder registered for {'video' if is_video else 'observation'} "
                f"type {spec.type!r} (key={spec.key}, topic={spec.topic}). "
                f"Registered types: {sorted(DECODERS)}."
            )
        tolerance_ns = auto_bound_tolerance_ns(
            spec.align.tolerance_ms, fps=self._fps
        )
        if spec.align.tolerance_ms is None:
            # Make the divergence from converter semantics visible at boot —
            # see ``stream_buffer.auto_bound_tolerance_ns`` for the reasoning.
            logger.warning(
                "Stream key=%s topic=%s has tolerance_ms=None (converter's "
                "'unlimited'); auto-bounded to %d ns for the live runtime.",
                spec.key,
                spec.topic,
                tolerance_ns,
            )
        buffer = StreamBuffer(
            method=spec.align.method,
            tolerance_ns=tolerance_ns,
        )
        return spec, buffer

    def _build_action_specs(self, specs: list[ActionSpec]) -> None:
        for spec in specs:
            if spec.type not in ENCODERS:
                # Surface the missing encoder at adapter boot — a contract
                # that lists an action we cannot encode is a config bug, and
                # discovering it at first-tick time loses both wall-clock and
                # debugging signal.
                raise ValueError(
                    f"No encoder registered for action type {spec.type!r} "
                    f"(key={spec.key}, topic={spec.topic}). "
                    f"Registered types: {sorted(ENCODERS)}."
                )
            self._action_groups.setdefault(spec.key, []).append(spec)

        for key, group in self._action_groups.items():
            if len(group) <= 1:
                continue
            # Multi-spec base keys need selector widths so the policy
            # output can be sliced back out. The converter relies on the
            # same widths via Contract.get_lerobot_features.
            for spec in group:
                names = (spec.selector or {}).get("names")
                if not names:
                    raise ValueError(
                        f"Action base key {key!r} has multiple specs but spec "
                        f"(topic={spec.topic}) is missing 'selector.names' — "
                        "required to slice the policy output across multi-topic actions."
                    )

    # ------------------------------------------------------------------
    # Public API — observation path
    # ------------------------------------------------------------------

    def on_message(self, topic: str, msg: Any, ts_ns: int) -> None:
        """Decode ``msg`` for every spec subscribed to ``topic`` and buffer it.

        Raises ``ValueError`` for unknown topics: a live driver
        publishing on the wrong topic is a contract bug, not a missing
        sample — failing loud is safer than silently dropping.

        Pre-stage causal filters (``self._pre_filters``) run here, on the
        decoded value, *before* ``buffer.push`` — matching the converter,
        which applies ``stage: pre`` transforms to the raw stream before
        alignment (see ``lerobot.py``'s pre-alignment-transform block).
        Applying the filter after alignment would feed it timestamps that
        have already been resampled/held, which is not what training saw.
        """
        entries = self._topic_entries.get(topic)
        if entries is None:
            raise ValueError(
                f"on_message received unknown topic {topic!r}. "
                f"Known topics: {sorted(self._topic_entries)}."
            )
        for spec, buffer in entries:
            value = decode_value(msg, spec)
            if spec.image and "resize" in (spec.image or {}):
                expected_h, expected_w = spec.image["resize"]
                value = resize_image(value, expected_h, expected_w)
            for pre_filter in self._pre_filters.get(spec.unique_key, ()):
                value = pre_filter.push(value)
            buffer.push(ts_ns, value)

    def sample(self, now_ns: int) -> dict[str, np.ndarray] | None:
        """Materialise the dict the policy sees at this tick.

        Returns ``None`` if any buffer for any key is stale — the policy
        expects every input, so partial observations are unsafe to feed.

        Stateful: post-stage causal filters (``self._post_filters``) carry
        ``zi`` across calls, so this method must be called exactly once
        per policy tick, in tick order. Calling it more than once for the
        same logical tick, skipping a tick, or calling it out of order
        desyncs the filter state from what an equivalent offline batch
        run would have produced.

        Implementation note: sampling happens in two passes so a stale
        tick (``None`` return) never partially advances filter state. Pass
        1 pulls every raw buffer value and bails on the first staleness,
        before any filter is touched. Pass 2 — reached only once every
        buffer for this tick is known-fresh — applies the post-filters and
        assembles the output. Without this split, a run of buffers where
        the first N filter successfully but the (N+1)th is stale would
        have already advanced N filters' ``zi`` for a tick whose result is
        discarded, corrupting them relative to an offline batch run over
        the same messages.
        """
        # Pass 1: gather raw samples; the first staleness aborts before any
        # post-filter state is touched.
        raw_obs: dict[str, list[np.ndarray]] = {}
        for key, entries in self._obs_key_entries.items():
            parts: list[np.ndarray] = []
            for _spec, buffer in entries:
                value = buffer.sample(now_ns)
                if value is None:
                    return None
                parts.append(np.asarray(value))
            raw_obs[key] = parts

        video_frames: dict[str, np.ndarray] = {}
        for key, (_spec, buffer) in self._video_key_entries.items():
            frame = buffer.sample(now_ns)
            if frame is None:
                return None
            video_frames[key] = frame

        # Pass 2: every stream for this tick is fresh — safe to advance
        # post-filter state and assemble the policy-facing dict.
        out: dict[str, np.ndarray] = {}
        for key, entries in self._obs_key_entries.items():
            filtered_parts: list[np.ndarray] = []
            for (spec, _buffer), value in zip(entries, raw_obs[key], strict=True):
                for post_filter in self._post_filters.get(spec.unique_key, ()):
                    value = post_filter.push(value)
                filtered_parts.append(value)
            assembled = (
                filtered_parts[0]
                if len(filtered_parts) == 1
                else np.concatenate(filtered_parts)
            )
            # Match the converter's frame assembly, which casts every
            # observation vector to float32; the decoders emit float64, so
            # without this the policy would see float64 where it trained on
            # float32. Videos stay uint8 and are handled separately below.
            out[key] = assembled.astype(np.float32, copy=False)

        out.update(video_frames)
        return out

    # ------------------------------------------------------------------
    # Public API — action path
    # ------------------------------------------------------------------

    def encode_action(
        self, action: dict[str, np.ndarray], now_ns: int
    ) -> list[tuple[str, dict[str, Any]]]:
        """Turn the policy's per-key action dict into ``(topic, payload)`` pairs.

        For base keys with a single action spec, the full ndarray is
        handed to the encoder unchanged. For base keys with multiple
        specs (same key, different topics), the ndarray is sliced by
        each spec's ``selector.names`` width — declaration order pins
        the slice boundaries to match what the converter wrote into the
        training feature.

        A ``butterworth_lowpass_causal`` transform declared on an action
        spec is intentionally a no-op here — this method never looks at
        ``spec.transforms`` at all. The converter filtered the *recorded*
        action stream to build the training targets, so the policy was
        trained to emit values that already live in the filtered target
        space; its output is the thing the filter's output would have
        been. Filtering it again would double-apply the transform. A
        low-pass filter also has no stable causal inverse to "undo" a
        filter that was never applied to the policy's output in the first
        place, so there is nothing to invert even if we wanted to. The
        adapter's transform gate (``_validate_transforms``) still requires
        the transform type to be ``butterworth_lowpass_causal`` — only the
        one transform proven reproducible live is allowed to appear on an
        action spec at all — but no filter is constructed for the action
        path (contrast ``self._pre_filters`` / ``self._post_filters``,
        which are observation-only).

        ``now_ns`` is currently unused; the generated node fills the
        message header. Kept on the signature so codegen can wire
        it without API churn.
        """
        del now_ns  # reserved for header timestamp handling

        out: list[tuple[str, dict[str, Any]]] = []

        for key, group in self._action_groups.items():
            if key not in action:
                raise KeyError(
                    f"Policy output missing required action key {key!r}. "
                    f"Got: {sorted(action)}; expected: {sorted(self._action_groups)}."
                )
            # Cast to float32 to match the converter's action feature dtype,
            # so the vector the adapter encodes is byte-consistent with the
            # float32 the policy was trained to emit.
            array = np.asarray(action[key], dtype=np.float32).reshape(-1)

            if len(group) == 1:
                payload = encode_value(array, group[0])
                out.append((group[0].topic, payload))
                continue

            offset = 0
            for spec in group:
                width = len((spec.selector or {}).get("names", []))
                slice_ = array[offset : offset + width]
                if slice_.shape[0] != width:
                    raise ValueError(
                        f"Action {key!r} is shorter than its declared specs require: "
                        f"have {array.shape[0]} entries, need >= {offset + width} "
                        f"to cover spec (topic={spec.topic})."
                    )
                payload = encode_value(slice_, spec)
                out.append((spec.topic, payload))
                offset += width

        return out

    # ------------------------------------------------------------------
    # Introspection (test helpers)
    # ------------------------------------------------------------------

    @property
    def topics(self) -> list[str]:
        """All ROS topics this adapter subscribes to. Ordering is insertion."""
        return list(self._topic_entries)

    @property
    def action_topics(self) -> list[str]:
        """All ROS topics this adapter publishes to, in declaration order."""
        return [spec.topic for group in self._action_groups.values() for spec in group]
