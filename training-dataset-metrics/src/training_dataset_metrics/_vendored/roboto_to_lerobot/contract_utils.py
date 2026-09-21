# Copyright 2025 Isaac Blankenau (Rosetta)
# Copyright 2025 Roboto AI (modifications for roboto-to-lerobot)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Contract model and loader for roboto-to-lerobot.

Derived from Rosetta's ``rosetta/common/contract_utils.py`` at commit a0c312a:
https://github.com/iblnkn/rosetta/blob/a0c312aed8f7901c8a819e708f437502924ae9ce/rosetta/common/contract_utils.py
License text: ``runtime/LICENSE-rosetta``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Any, Literal, NamedTuple

import pandas as pd
import roboto
import yaml

from . import decoders  # noqa: F401
from .converters import DECODERS, decode_value
from .extract import (
    _add_timestamp_column,
    _sanitize_topic_for_key,
    keep_monotonic_timestamps,
)
from .video import (
    COMPRESSED_VIDEO_MESSAGE_PATHS,
    COMPRESSED_VIDEO_SCHEMAS,
    decode_video_stream_rows,
    is_compressed_video_schema,
    is_compressed_video_topic,
)

# Maximum threads per ``DataCollection`` for in-event topic fetches. Each
# fetch is socket-I/O-bound (signed-URL fetch + pyarrow read) and releases
# the GIL, so threading parallelises well. Callers nest this inside a
# process pool, making the total socket budget
# ``pool_size × _TOPIC_FETCH_THREADS`` — keep the cap modest.
_TOPIC_FETCH_THREADS = 16


def fps_to_time_step_ns(fps: float) -> int:
    """Convert a contract fps into nanoseconds-per-frame using round-to-nearest.

    Truncating (``int(1e9 / fps)``) loses up to one nanosecond per frame for
    fractional rates such as 29.97 or 23.976, drift that accumulates into a
    visible offset over a long episode. Rounding caps the per-frame error at
    half a nanosecond and is consistent across the main process, every
    worker, and ``generate_frames``'s ``action_lead`` arithmetic.
    """
    return round(1e9 / fps)


# ---------- Contract datamodel ----------


# Contract ``type:`` values that route to the file-backed video loader
# (``topic.get_data()`` with ``content_dict["image"]``). ``video`` is the
# preferred spelling; ``avi_video`` / ``mp4_video`` are kept for older
# contracts. The per-frame decode dispatches on the same ``type`` string,
# so each entry here also has a matching ``register_decoder(...)`` alias
# in ``decoders.py``.
_IMAGE_VIDEO_TYPES = frozenset({"video", "avi_video", "mp4_video"})


def _empty_video_df() -> pd.DataFrame:
    """Empty video DataFrame whose columns have the dtypes the merge expects."""
    return pd.DataFrame({
        "timestamp": pd.Series([], dtype="int64"),
        "format":    pd.Series([], dtype="object"),
        "data":      pd.Series([], dtype="object"),
    })


def _empty_video_stream_df() -> pd.DataFrame:
    """Empty compressed-video DataFrame: one already-decoded ``frame`` per row.

    Compressed-video streams are decoded a whole range at a time (see
    :py:mod:`roboto_to_lerobot.video`), so their loaded shape carries decoded
    HWC uint8 RGB arrays rather than the encoded ``format``/``data`` pair the
    per-message image paths carry.
    """
    return pd.DataFrame({
        "timestamp": pd.Series([], dtype="int64"),
        "frame":     pd.Series([], dtype="object"),
    })


def image_resize(spec: ObservationSpec) -> tuple[int, int] | None:
    """The ``(height, width)`` an image/video spec's frames must be resized to, if any.

    Returns the contract's values verbatim, deliberately: the compressed-video
    path resizes at decode time to bound memory, and ``lerobot.generate_frames``
    resizes again just before ``add_frame`` reading ``image["resize"]`` directly.
    The second call is a no-op only because both see the same target shape, so
    this must not normalise what the other side does not.
    """
    if spec.image and "resize" in spec.image:
        height, width = spec.image["resize"]
        return height, width
    return None


def video_spec_kind(video: ObservationSpec, topic_list: list[Any]) -> str:
    """Which fetch/load shape a video spec uses, cross-checked against its topics.

    One decision, consulted by both the work-list builder and the loader, so the
    bucket a spec's data is written to and the bucket it is read from can never
    disagree:

    - ``video_file`` — file-backed video (``video``/``avi_video``/``mp4_video``);
      ``topic.get_data()`` yields one still frame per message.
    - ``video_stream`` — compressed video (``foxglove_msgs/msg/CompressedVideo``
      and friends); a whole range is GOP-decoded at fetch time.
    - ``video_msgs`` — per-message stills (``CompressedImage``, ``Image``).

    The topics are consulted, not just the declared ``type:``, because a
    compressed-video topic declared as ``CompressedImage`` would otherwise hand
    single H.264 access units to ``cv2.imdecode``. Detection reads the signals
    Roboto ingestion registers — the representation format and the schema name
    (see :py:func:`roboto_to_lerobot.video.is_compressed_video_topic`).

    Args:
        video: The contract's video spec.
        topic_list: Every Topic bound to the spec's topic name (a chunked
            recording exposes one per file).

    Returns:
        The ``spec_kind`` string for this spec's work items.

    Raises:
        ValueError: If the spec's topics store compressed video but the spec
            does not declare it, or if it declares a compressed-video spelling
            that has no registered decoder.
    """
    if video.type.lower() in _IMAGE_VIDEO_TYPES:
        return "video_file"

    if is_compressed_video_schema(video.type):
        if video.type not in DECODERS:
            raise ValueError(
                f"Video spec '{video.key}' declares compressed-video type '{video.type}', "
                f"which has no registered decoder. Use one of: "
                f"{', '.join(sorted(COMPRESSED_VIDEO_SCHEMAS))}."
            )
        return "video_stream"

    if any(is_compressed_video_topic(topic) for topic in topic_list):
        raise ValueError(
            f"Video spec '{video.key}' declares type '{video.type}', but topic "
            f"'{video.topic}' stores compressed video (one encoded access unit per "
            f"message, not a still image). Declare it as "
            f"'foxglove_msgs/msg/CompressedVideo' so the frames are GOP-decoded."
        )
    return "video_msgs"


VALID_ALIGNMENT_METHODS = ("hold", "nearest", "linear", "none")
"""Supported alignment methods for time-aligning data streams.

- ``hold``    – last observation carried forward (backward as-of join).
- ``nearest`` – pick the closest sample in time regardless of direction.
- ``linear``  – linearly interpolate between the two bracketing samples.
- ``none``    – only exact timestamp matches; everything else becomes NaN.
"""


VALID_QOS_RELIABILITY = ("BEST_EFFORT", "RELIABLE")
VALID_QOS_HISTORY = ("KEEP_LAST", "KEEP_ALL")


@dataclass(frozen=True, slots=True)
class QosSpec:
    """ROS 2 QoS profile for a subscription or publisher.

    Lives on observation/action specs and is consumed by codegen
    to emit a ``QoSProfile(...)`` in the generated node. Carried on the
    schema rather than synthesised at codegen time so an author can pin a
    non-default profile when the publisher requires it — a wrong QoS in a
    live node yields silent zero-message subscriptions.

    When the spec leaves ``qos:`` unset, the spec field stays ``None`` and
    codegen falls back to a type-based default (sensor topics ⇒
    ``BEST_EFFORT/KEEP_LAST/1``; state/action ⇒ ``RELIABLE/KEEP_LAST/10``).
    The offline converter never reads this field — the contract loader
    parses it and the converter ignores it.

    Field names borrow rosetta's spelling verbatim (``reliability``,
    ``history``, ``depth``); values use the ROS 2 enum names
    (``BEST_EFFORT``, ``RELIABLE``, ``KEEP_LAST``, ``KEEP_ALL``) so
    codegen can substitute them into ``QoSProfile(...)`` without
    translation.
    """

    reliability: str = "BEST_EFFORT"
    history: str = "KEEP_LAST"
    depth: int = 1

    def __post_init__(self):
        if self.reliability not in VALID_QOS_RELIABILITY:
            raise ValueError(
                f"Unknown QoS reliability '{self.reliability}'. "
                f"Valid options: {VALID_QOS_RELIABILITY}"
            )
        if self.history not in VALID_QOS_HISTORY:
            raise ValueError(
                f"Unknown QoS history '{self.history}'. "
                f"Valid options: {VALID_QOS_HISTORY}"
            )
        if self.history == "KEEP_LAST" and self.depth < 1:
            # A KEEP_LAST queue of depth 0 is degenerate: DDS treats it as
            # zero-capacity and silently drops every message — the exact
            # "silent zero-message subscription" this validation exists to
            # prevent. depth is only meaningful for KEEP_LAST (KEEP_ALL
            # ignores it), so the floor only applies there.
            raise ValueError(
                f"QoS history=KEEP_LAST requires depth >= 1, got {self.depth}: "
                "a zero-capacity queue silently drops every message."
            )
        if self.depth < 0:
            raise ValueError(
                f"QoS depth must be non-negative, got {self.depth}"
            )


VALID_SAFETY_BEHAVIORS = ("publish_nothing", "hold_last", "safe_pose")
"""Behaviours an action publisher uses when the observation buffer goes stale.

- ``publish_nothing`` — default; downstream controller times out and applies
  its own stop. The only mode codegen accepts.
- ``hold_last`` / ``safe_pose`` — parsed but not supported by codegen;
  ``gen-node`` refuses with a clear error.
"""


@dataclass(frozen=True, slots=True)
class AlignSpec:
    """Per-stream time-alignment configuration.

    method:       one of ``VALID_ALIGNMENT_METHODS``
    tolerance_ms: maximum gap (in ms) before the result is NaN.
                  ``None`` means unlimited (always carry forward / pick
                  nearest, however far). A positive float is a concrete
                  bound. This is the internal representation only — see
                  :func:`_as_align` for how contract YAML (omitted block,
                  omitted field, explicit ``null``, explicit ``0``) maps
                  onto it.
    """

    method: str = "hold"
    tolerance_ms: float | None = None

    def __post_init__(self):
        if self.method not in VALID_ALIGNMENT_METHODS:
            raise ValueError(
                f"Unknown alignment method '{self.method}'. "
                f"Valid options: {VALID_ALIGNMENT_METHODS}"
            )
        if self.tolerance_ms is not None and self.tolerance_ms <= 0:
            raise ValueError(
                f"AlignSpec.tolerance_ms must be None (unlimited) or a "
                f"positive number, got {self.tolerance_ms}. Contract YAML "
                "authors: use 'tolerance_ms: null' for unlimited, or omit "
                "'tolerance_ms'/'align' for the auto-bounded default."
            )


@dataclass(frozen=True, slots=True)
class TransformSpec:
    """Transform configuration.

    type:   transform name (matches key in the TRANSFORMS registry)
    params: free-form kwargs forwarded to the transform function
    stage:  "pre" (before alignment, on raw stream) or "post" (after alignment)
    """

    type: str
    params: dict[str, Any]
    stage: str = "post"


@dataclass(frozen=True, slots=True)
class ObservationSpec:
    """Observation stream description (image/vector), driven by AlignSpec.

    Either ``topic`` or ``role`` must be set on the YAML side. When ``role``
    is set, the ``topic`` field is populated with an internal placeholder
    (``__role__:<role>``) and gets replaced by the resolved topic name once
    :func:`resolve_role_bindings` runs against a concrete dataset. Image
    streams cannot be role-bound — see :func:`_topic_or_role`.
    """

    key: str
    topic: str
    type: str
    role: str | None = None
    selector: dict[str, Any] | None = None  # {names: [...]}
    image: dict[str, Any] | None = (
        None  # {resize:[H,W], encoding:'rgb8'|'bgr8'|'mono8'...}
    )
    align: AlignSpec = AlignSpec()
    lerobot_names: list[str] | None = None
    transforms: list[TransformSpec] | None = None
    qos: QosSpec | None = None

    @property
    def unique_key(self) -> str:
        """Generate a unique key that includes the topic name.

        This ensures that multiple observations with the same base key but
        different topics are stored separately.
        Format: {key}.{sanitized_topic_name}
        """
        sanitized_topic = _sanitize_topic_for_key(self.topic)
        return f"{self.key}.{sanitized_topic}"

    def get_prefixed_selector_names(self) -> list[str]:
        """Get selector names prefixed with the topic name.

        This ensures that features from different topics with the same
        selector names can be distinguished.
        Format: {topic}/{selector_name}
        """
        if not self.selector or "names" not in self.selector:
            return []
        return [f"{self.topic}/{name}" for name in self.selector["names"]]

    def get_lerobot_selector_names(self) -> list[str]:
        """Names for LeRobot feature metadata.

        Returns ``lerobot_names`` if specified, otherwise falls back to
        prefixed selector names (``topic/field_name``).
        """
        if self.lerobot_names:
            return list(self.lerobot_names)
        return self.get_prefixed_selector_names()


@dataclass(frozen=True, slots=True)
class ActionSpec:
    """Action stream description.

    YAML (example):
      actions:
        - key: action
          topic: /cmd_vel
          type: geometry_msgs/msg/Twist
          selector: { names: [linear.x, angular.z] }

    Either ``topic`` or ``role`` must be set on the YAML side; see
    :class:`ObservationSpec` and :func:`resolve_role_bindings` for how
    role-bound specs get resolved at runtime.

    Note: Flattening of nested arrays (e.g., JointTrajectory points) is
    handled automatically by the type-based decoder. No flatten config needed.
    """

    key: str
    topic: str
    type: str
    role: str | None = None
    selector: dict[str, Any] | None = None
    align: AlignSpec = AlignSpec()
    lerobot_names: list[str] | None = None
    transforms: list[TransformSpec] | None = None
    qos: QosSpec | None = None
    safety_behavior: str = "publish_nothing"

    @property
    def unique_key(self) -> str:
        """Generate a unique key that includes the topic name.

        This ensures that multiple actions with the same base key but
        different topics are stored separately.
        Format: {key}.{sanitized_topic_name}
        """
        sanitized_topic = _sanitize_topic_for_key(self.topic)
        return f"{self.key}.{sanitized_topic}"

    def get_prefixed_selector_names(self) -> list[str]:
        """Get selector names prefixed with the topic name.

        This ensures that features from different topics with the same
        selector names can be distinguished.
        Format: {topic}/{selector_name}
        """
        if not self.selector or "names" not in self.selector:
            return []
        return [f"{self.topic}/{name}" for name in self.selector["names"]]

    def get_lerobot_selector_names(self) -> list[str]:
        """Names for LeRobot feature metadata.

        Returns ``lerobot_names`` if specified, otherwise falls back to
        prefixed selector names (``topic/field_name``).
        """
        if self.lerobot_names:
            return list(self.lerobot_names)
        return self.get_prefixed_selector_names()


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """Optional 'task' channels (e.g., prompts)."""

    key: str
    topic: str
    type: str


@dataclass(frozen=True, slots=True)
class Contract:
    """Top-level contract describing a policy's ROS 2 I/O surface."""

    name: str
    version: int
    fps: float
    observations: list[ObservationSpec]
    videos: list[ObservationSpec]
    actions: list[ActionSpec]
    tasks: list[TaskSpec]
    robot_type: str | None = None
    action_lead_steps: int = 0

    @property
    def all_topics(self) -> list[str]:
        """Return list of all ROS topics (observations + actions)."""
        obs_topics = [o.topic for o in self.observations]
        act_topics = [a.topic for a in self.actions]
        video_topics = [v.topic for v in self.videos]
        return obs_topics + act_topics + video_topics

    def get_lerobot_features(
        self,
        resolved_features: dict[str, tuple[int, list[str]]] | None = None,
    ):
        """Return LeRobot feature dictionary for this contract.

        Args:
            resolved_features: Optional mapping of unique_key -> (num_values, names)
                discovered at runtime by DataCollection. When present, these
                override the static selector.names from the contract YAML
                (which may be empty or refer to array field names rather than
                individual joint names).

        Note:
            Multiple observations/actions with the same base key are concatenated
            into a single feature. The names list contains all prefixed selector
            names to identify which element came from which topic.

            For example, two observation.state entries from different topics
            produce a single "observation.state" feature with combined shape
            and names like ["topic1/field1", "topic1/field2", "topic2/field1", ...].
        """
        resolved = resolved_features or {}
        features = {}

        # Group observations by base key and concatenate
        obs_by_key: dict[str, list[ObservationSpec]] = {}
        for obs in self.observations:
            if obs.key not in obs_by_key:
                obs_by_key[obs.key] = []
            obs_by_key[obs.key].append(obs)

        for base_key, obs_list in obs_by_key.items():
            total_values = 0
            all_names: list[str] = []

            for obs in obs_list:
                unique_key = obs.unique_key
                if unique_key in resolved:
                    num_values, names = resolved[unique_key]
                    total_values += num_values
                    all_names.extend(names)
                elif obs.selector and "names" in obs.selector:
                    lr_names = obs.get_lerobot_selector_names()
                    total_values += len(lr_names)
                    all_names.extend(lr_names)

            if total_values > 0:
                features[base_key] = {
                    "dtype": "float32",
                    "shape": (total_values,),
                    "names": all_names,
                }

        # Videos are 1:1 with their key (uniqueness is validated at load time).
        for video in self.videos:
            if video.image and "resize" in video.image:
                features[video.key] = {
                    "dtype": "video",
                    "shape": (video.image["resize"][0], video.image["resize"][1], 3),
                    "names": ["height", "width", "channel"],
                }

        # Group actions by base key and concatenate
        act_by_key: dict[str, list[ActionSpec]] = {}
        for action in self.actions:
            if action.key not in act_by_key:
                act_by_key[action.key] = []
            act_by_key[action.key].append(action)

        for base_key, act_list in act_by_key.items():
            total_values = 0
            all_names: list[str] = []

            for action in act_list:
                unique_key = action.unique_key
                if unique_key in resolved:
                    num_values, names = resolved[unique_key]
                    total_values += num_values
                    all_names.extend(names)
                elif action.selector and "names" in action.selector:
                    lr_names = action.get_lerobot_selector_names()
                    total_values += len(lr_names)
                    all_names.extend(lr_names)

            if total_values > 0:
                features[base_key] = {
                    "dtype": "float32",
                    "shape": (total_values,),
                    "names": all_names,
                }

        return features

    def get_key_for_topic(self, topic: str) -> str:
        """Return the unique_key for a given topic."""
        for obs in self.observations:
            if obs.topic == topic:
                return obs.unique_key

        for video in self.videos:
            if video.topic == topic:
                return video.key

        for action in self.actions:
            if action.topic == topic:
                return action.unique_key

        raise ValueError(f"Topic {topic} not found in contract")


def _compute_buffer_ns(contract: Contract) -> int:
    """Time-window slack added on each side of every event window.

    The fetch window is widened by ``buffer_ns`` so as-of/nearest joins have
    bracketing samples just outside the event boundary. When any spec has
    ``tolerance_ms is None`` (AlignSpec's "unlimited carry-forward" value),
    we fall back to a 1 s slack: unbounded carry-forward needs a buffer that
    covers the source-rate gap, which can't be bounded from the contract.

    Returns:
        ``max(2 × max(tolerance_ms), floor)`` where ``floor = max(50 ms, 1/fps)``,
        or ``int(1e9)`` (1 s) when the contract has no specs or any spec has
        ``tolerance_ms is None``.
    """
    all_specs = [
        *(o.align for o in contract.observations),
        *(v.align for v in contract.videos),
        *(a.align for a in contract.actions),
    ]
    if not all_specs:
        return int(1e9)
    tolerances = [s.tolerance_ms for s in all_specs]
    if any(t is None for t in tolerances):
        return int(1e9)
    max_tol_ns = max(tolerances) * 1_000_000
    min_floor_ns = max(50_000_000, int(1e9 / contract.fps))
    return max(min_floor_ns, int(2 * max_tol_ns))


def resolve_role_bindings(
    dataset: roboto.Dataset,
    contract: Contract,
) -> Contract:
    """Replace role-bound specs with topic-bound specs for this dataset.

    For each spec with ``role`` set, find the single file in ``dataset``
    tagged with ``file.metadata["role"] == spec.role``, then pick the topic
    on that file whose ``message_paths`` cover the spec's ``selector.names``.
    Raises :class:`ValueError` on zero or ambiguous matches.

    Specs that already carry a literal ``topic`` (no ``role``) are returned
    unchanged, so legacy contracts pass through untouched.
    """
    from .logger import logger

    role_specs: list[tuple[str, Any]] = []
    for spec in contract.observations:
        if spec.role is not None:
            role_specs.append(("observations", spec))
    for spec in contract.actions:
        if spec.role is not None:
            role_specs.append(("actions", spec))

    if not role_specs:
        return contract

    # Single dataset.list_files pass: bucket every file by its role tag.
    files_by_role: dict[str, list] = {}
    for file in dataset.list_files():
        md = getattr(file, "metadata", None) or {}
        role = md.get("role") if isinstance(md, dict) else None
        if not role:
            continue
        files_by_role.setdefault(str(role), []).append(file)

    def _resolve_topic_for(spec: Any) -> str:
        matches = files_by_role.get(spec.role, [])
        if len(matches) == 0:
            raise ValueError(
                f"Role '{spec.role}' (spec key='{spec.key}') not found on any "
                f"file in dataset {dataset.dataset_id}. Tag files with a "
                f"`role` metadata field first."
            )
        if len(matches) > 1:
            raise ValueError(
                f"Role '{spec.role}' (spec key='{spec.key}') matched "
                f"{len(matches)} files in dataset {dataset.dataset_id}: "
                f"{[f.relative_path for f in matches]}. Expected exactly 1."
            )
        file = matches[0]
        required = set((spec.selector or {}).get("names", []) or [])
        candidates = []
        for topic in file.get_topics():
            paths = {mp.message_path for mp in topic.message_paths}
            if not required or required.issubset(paths):
                candidates.append(topic)
        if not candidates:
            raise ValueError(
                f"No topic on file {file.relative_path} covers selector "
                f"names {sorted(required)} for role '{spec.role}'"
            )
        if len(candidates) > 1:
            raise ValueError(
                f"Multiple topics on file {file.relative_path} cover selector "
                f"names for role '{spec.role}': "
                f"{[t.topic_name for t in candidates]}"
            )
        resolved = candidates[0].topic_name
        # Per-binding detail is only useful when debugging a misconfigured
        # contract; the success-path summary lives in
        # collect_topics_from_dataset's single INFO line.
        logger.debug(
            "Resolved role=%s key=%s -> topic=%s (file=%s)",
            spec.role, spec.key, resolved, file.relative_path,
        )
        return resolved

    new_obs = [
        dataclass_replace(s, topic=_resolve_topic_for(s)) if s.role is not None else s
        for s in contract.observations
    ]
    new_actions = [
        dataclass_replace(s, topic=_resolve_topic_for(s)) if s.role is not None else s
        for s in contract.actions
    ]
    return dataclass_replace(contract, observations=new_obs, actions=new_actions)


def collect_topics_from_dataset(
    dataset: roboto.Dataset,
    contract: Contract,
) -> tuple[dict[str, list[roboto.Topic]], list[dict]]:
    """
    Collect all topics from a dataset that are mentioned in the contract.

    Chunked recordings — same topic name across multiple files with disjoint
    time ranges — are preserved here and merged later in ``_merge_spec_dfs``
    after every chunked Topic is fetched in parallel by ``_run_topic_fetches``.
    True duplicates — same topic name with identical
    ``(start_time, end_time, message_count)`` across multiple files — are
    deduplicated here: one winner (lowest ``file_id``) survives, the rest are
    dropped. Without this, ``concat+sort`` would produce per-row timestamp
    multiplicity and trip the degenerate-timestamp guard in ``generate_frames``.

    Near-duplicates (same window but ``message_count`` differs by 1-2) are
    intentionally NOT caught — the signature must match exactly. The
    degenerate-timestamp guard in ``generate_frames`` remains the safety net
    for malformed single-file streams.

    Returns:
        Tuple of:
          - Dictionary mapping topic name to list of Topic objects (post-dedup).
          - List of dedup-group records (one per `(name, signature)` with >1
            file), for manifest reporting.

    Raises:
        ValueError: If a required topic from the contract is not found in any file
    """
    from .logger import logger

    contract_topics = set(contract.all_topics)
    # name -> list of (topic, file_relative_path); the path is only needed for
    # manifest reporting so we keep it alongside the Topic.
    raw: dict[str, list[tuple[roboto.Topic, str]]] = {}
    files_scanned = 0

    for file in dataset.list_files():
        files_scanned += 1
        for topic in file.get_topics():
            if topic.name in contract_topics:
                raw.setdefault(topic.name, []).append((topic, file.relative_path))
                # Per-(topic, file) finds are visible at DEBUG when a contract
                # mismatch needs unpicking; the success-path summary log lives
                # at the bottom of this function.
                logger.debug(
                    "Found topic '%s' in file '%s'",
                    topic.name, file.relative_path,
                )

    missing_topics = contract_topics - set(raw.keys())
    if missing_topics:
        raise ValueError(
            f"The following topics from the contract were not found in the dataset: "
            f"{sorted(missing_topics)}"
        )

    topics: dict[str, list[roboto.Topic]] = {}
    dedup_groups: list[dict] = []
    for name, entries in raw.items():
        by_sig: dict[tuple, list[tuple[roboto.Topic, str]]] = {}
        for t, path in entries:
            by_sig.setdefault((t.start_time, t.end_time, t.message_count), []).append((t, path))

        kept: list[roboto.Topic] = []
        for sig, group in by_sig.items():
            if len(group) == 1:
                kept.append(group[0][0])
                continue
            # Deterministic winner so repeat runs produce identical output.
            group_sorted = sorted(group, key=lambda x: x[0].file_id)
            winner, winner_path = group_sorted[0]
            dropped = group_sorted[1:]
            logger.warning(
                "Topic '%s' duplicated across %d files (start=%s end=%s count=%s); "
                "keeping %s (%s), dropping %s",
                name, len(group), sig[0], sig[1], sig[2],
                winner.file_id, winner_path,
                [d[0].file_id for d in dropped],
            )
            dedup_groups.append({
                "topic_name": name,
                "signature": {
                    "start_time_ns": sig[0],
                    "end_time_ns": sig[1],
                    "message_count": sig[2],
                },
                "kept": {
                    "file_id": winner.file_id,
                    "file_path": winner_path,
                    "topic_id": winner.topic_id,
                },
                "dropped": [
                    {"file_id": t.file_id, "file_path": p, "topic_id": t.topic_id}
                    for t, p in dropped
                ],
            })
            kept.append(winner)
        topics[name] = kept

    # One summary line per dataset replaces the chatty per-binding /
    # per-(topic, file) INFO output. dataset_id keeps it greppable when
    # multiple datasets share a run; chunk_count gives a hint when a
    # multi-file recording fanned out into multiple Topic objects under
    # the same name.
    chunk_count = sum(len(v) for v in topics.values())
    dedup_dropped = sum(len(g["dropped"]) for g in dedup_groups)
    suffix = f"; deduped {dedup_dropped} duplicate(s)" if dedup_dropped else ""
    logger.info(
        "Dataset %s: matched %d topic(s) (%d chunk(s)) across %d file(s)%s",
        getattr(dataset, "dataset_id", "?"),
        len(topics), chunk_count, files_scanned, suffix,
    )

    return topics, dedup_groups


class _TopicWorkItem(NamedTuple):
    """One unit of topic-data fetch dispatched by :func:`_run_topic_fetches`.

    ``spec_kind`` selects which fetch shape applies (see :data:`_FETCH_SHAPE`):
    ``"video_file"`` calls ``topic.get_data()`` and yields per-frame rows,
    ``"video_stream"`` GOP-decodes a compressed-video range into decoded-frame
    rows, and everything else calls ``topic.get_data_as_df`` and returns the
    (timestamp-normalised) DataFrame. ``submission_idx`` is a monotonic counter
    assigned in source-order so the fan-out step can reassemble results in
    deterministic source-order — that order is the stable-sort tie-break that
    keeps the post-fetch ``pd.concat`` bit-stable across runs.

    ``resize`` is only set for ``video_stream`` items, whose decode step resizes
    frames as it produces them.
    """
    spec_kind: Literal["observation", "video_msgs", "video_file", "video_stream", "action", "task"]
    spec_key: str
    topic_name: str
    topic: Any  # roboto.Topic — typed loosely so test fakes don't need to subclass
    start_ns: int
    end_ns: int
    message_paths_include: list[str] | None
    submission_idx: int
    resize: tuple[int, int] | None = None


# The result shape each ``spec_kind`` fetches. Two work items may only share one
# fetch if they agree on shape — a DataFrame handed to a loader expecting frame
# rows (or vice versa) fails far from the cause.
_FETCH_SHAPE: dict[str, str] = {
    "video_file": "image_rows",
    "video_stream": "frame_rows",
}


def _topic_intersects_window(
    topic: Any, start_time_ns: int, end_time_ns: int,
) -> bool:
    """True if the Topic's recorded range overlaps ``[start_ns, end_ns]``.

    Cheap pre-filter that avoids a signed-URL fetch for chunked files whose
    timestamps lie entirely outside the event window.
    """
    topic_start = topic.start_time
    topic_end = topic.end_time
    if topic_end is not None and topic_end < start_time_ns:
        return False
    if topic_start is not None and topic_start > end_time_ns:
        return False
    return True


def _fetch_one(item: _TopicWorkItem) -> Any:
    """Execute one work item against the roboto SDK.

    For ``video_file`` specs the SDK returns ``(timestamp, content_dict)``
    tuples; we eagerly materialise them into ``{timestamp, format, data}``
    row dicts so the result is safe to share across threads and across the
    later fan-out step. ``video_stream`` specs GOP-decode the range into
    ``{timestamp, frame}`` row dicts — compressed video has no per-message
    decode, so the decode has to happen where a whole range is in hand. For
    every other ``spec_kind`` the result is the DataFrame from
    ``get_data_as_df``, normalised to int64 ns timestamps.
    """
    from .logger import logger

    start_time = pd.Timestamp(item.start_ns, unit="ns")
    end_time = pd.Timestamp(item.end_ns, unit="ns")

    if item.spec_kind == "video_stream":
        return decode_video_stream_rows(
            item.topic,
            item.start_ns,
            item.end_ns,
            topic_name=item.topic_name,
            resize=item.resize,
        )

    if item.spec_kind == "video_file":
        camera_data = item.topic.get_data(start_time=start_time, end_time=end_time)
        rows: list[dict] = []
        for timestamp, content_dict in camera_data:
            if hasattr(timestamp, "value"):
                ts_ns = timestamp.value
            elif isinstance(timestamp, pd.Timestamp):
                ts_ns = timestamp.value
            else:
                ts_ns = int(pd.Timestamp(timestamp).value)

            image_data = content_dict.get("image")
            if image_data is None:
                logger.warning(
                    "No 'image' key in content_dict for topic %s",
                    item.topic_name,
                )
                continue

            rows.append({
                "timestamp": ts_ns,
                "format": "jpeg",
                "data": image_data,
            })
        return rows

    if item.message_paths_include:
        df = item.topic.get_data_as_df(
            start_time=start_time,
            end_time=end_time,
            message_paths_include=item.message_paths_include,
        )
    else:
        df = item.topic.get_data_as_df(
            start_time=start_time,
            end_time=end_time,
        )
    return _add_timestamp_column(df)


def _run_topic_fetches(
    items: list[_TopicWorkItem],
    *,
    max_workers: int = _TOPIC_FETCH_THREADS,
) -> dict[tuple[str, str], list[Any]]:
    """Fetch every work item concurrently and bucket by ``(spec_kind, spec_key)``.

    Items sharing ``(topic_id, start_ns, end_ns, sorted(message_paths), fetch
    shape, resize)`` collapse to a single fetch whose result fans out to every
    requester — catches the pathological case of two specs pinned to the same
    topic without paying for it in the common case. The shape and resize are
    part of the key because they change what the fetch *returns*: a DataFrame,
    encoded-image rows, or decoded frames at a particular resolution.

    The returned per-spec lists are ordered by ``submission_idx`` so the
    downstream ``pd.concat`` always sees rows in deterministic source-order.
    That order is the stable-sort tie-break for the timestamp sort that
    follows.
    """
    if not items:
        return {}

    def _dedup_key(it: _TopicWorkItem) -> tuple:
        return (
            it.topic.topic_id,
            it.start_ns,
            it.end_ns,
            tuple(sorted(it.message_paths_include or ())),
            _FETCH_SHAPE.get(it.spec_kind, "dataframe"),
            it.resize,
        )

    unique: dict[tuple, _TopicWorkItem] = {}
    for it in items:
        unique.setdefault(_dedup_key(it), it)

    # Cap threads at the number of *distinct* fetches: no point spinning up
    # idle workers, and callers nest this inside a process pool where
    # ``pool_size × _TOPIC_FETCH_THREADS`` is the global socket budget.
    workers = max(1, min(max_workers, len(unique)))
    results_by_key: dict[tuple, Any] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_fetch_one, item): key
            for key, item in unique.items()
        }
        for fut, key in futures.items():
            results_by_key[key] = fut.result()

    bucketed: dict[tuple[str, str], list[tuple[int, Any]]] = {}
    for it in items:
        result = results_by_key[_dedup_key(it)]
        bucketed.setdefault((it.spec_kind, it.spec_key), []).append(
            (it.submission_idx, result),
        )

    out: dict[tuple[str, str], list[Any]] = {}
    for spec, pairs in bucketed.items():
        pairs.sort(key=lambda p: p[0])
        out[spec] = [r for _, r in pairs]
    return out


def _merge_spec_dfs(per_topic: list[pd.DataFrame]) -> pd.DataFrame:
    """Concat + timestamp-sort the per-Topic DataFrames for one spec.

    Returns an empty DataFrame when nothing was fetched, so loaders can
    treat the empty case with ``len(df) == 0`` rather than ``None``.
    """
    if not per_topic:
        return pd.DataFrame()
    merged = pd.concat(per_topic, ignore_index=True)
    return merged.sort_values(by="timestamp").reset_index(drop=True)


def _merge_video_rows(
    per_topic_rows: list[list[dict]],
    empty: pd.DataFrame,
) -> pd.DataFrame:
    """Flatten + timestamp-sort the per-Topic row lists of a row-shaped video fetch.

    Shared by the file-backed loader (``{timestamp, format, data}`` rows) and
    the compressed-video loader (``{timestamp, frame}`` rows) — the flattening
    is identical, only the column set differs, which is what ``empty`` carries.

    Args:
        per_topic_rows: One row list per Topic bound to the spec.
        empty: The correctly-typed empty DataFrame to return when nothing was
            fetched, so the downstream ``generate_frames`` doesn't trip on
            column dtypes.
    """
    all_rows: list[dict] = []
    for rows in per_topic_rows:
        all_rows.extend(rows)
    if not all_rows:
        return empty
    df = pd.DataFrame(all_rows)
    df["timestamp"] = df["timestamp"].astype("int64")
    return df.sort_values(by="timestamp").reset_index(drop=True)


class DataCollection:
    """
    Generalized data collection from roboto ingested files based on contract specification.

    This class loads observations, videos, actions, and tasks from topic data
    according to the contract, handling different message types appropriately.
    """

    def __init__(
        self,
        contract: Contract,
        topics: dict[str, list[roboto.Topic]],
        start_time_ns: int,
        end_time_ns: int,
    ):
        """
        Initialize DataCollection by loading all topics specified in the contract.

        Args:
            contract: Contract specifying which topics to load and how to process them
            topics: Dictionary mapping topic name to list of Topic objects
            start_time_ns: Start time in nanoseconds (for filtering data)
            end_time_ns: End time in nanoseconds (for filtering data)
        """
        from .logger import logger

        self.contract = contract
        self.topics = topics
        self.start_time_ns = start_time_ns
        self.end_time_ns = end_time_ns
        self.observations: dict[str, pd.DataFrame] = {}
        self.videos: dict[str, pd.DataFrame] = {}
        self.actions: dict[str, pd.DataFrame] = {}
        self.tasks: dict[str, pd.DataFrame] = {}
        # Runtime-discovered feature info: key -> (num_values, joint_names)
        # Populated after loading when array columns are expanded.
        self.resolved_features: dict[str, tuple[int, list[str]]] = {}

        work_list = self._build_work_list(contract, topics, start_time_ns, end_time_ns)
        raw_by_spec = _run_topic_fetches(work_list)

        # Fixed loader order (videos → actions → observations → tasks) pins
        # the insertion order of ``self.{videos,actions,observations,tasks}``,
        # so any downstream code that depends on dict iteration order sees
        # a deterministic sequence.
        self._load_videos(contract, raw_by_spec, logger)
        self._load_actions(contract, raw_by_spec, logger)
        self._load_observations(contract, raw_by_spec, logger)
        self._load_tasks(contract, raw_by_spec, logger)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_work_list(
        contract: Contract,
        topics: dict[str, list[roboto.Topic]],
        start_time_ns: int,
        end_time_ns: int,
    ) -> list[_TopicWorkItem]:
        """Walk every contract spec and emit a flat list of fetch work items.

        Submission order is fixed (videos → actions → observations → tasks,
        each in declaration order) so the post-fetch ``pd.concat`` sees rows
        in a deterministic, bit-stable order across runs.
        """
        work_list: list[_TopicWorkItem] = []
        submission_idx = 0

        def _append(spec_kind, spec_key, topic_name, topic_list, message_paths, resize=None):
            nonlocal submission_idx
            for topic in topic_list:
                if not _topic_intersects_window(topic, start_time_ns, end_time_ns):
                    continue
                work_list.append(_TopicWorkItem(
                    spec_kind=spec_kind,
                    spec_key=spec_key,
                    topic_name=topic_name,
                    topic=topic,
                    start_ns=start_time_ns,
                    end_ns=end_time_ns,
                    message_paths_include=message_paths,
                    submission_idx=submission_idx,
                    resize=resize,
                ))
                submission_idx += 1

        for video in contract.videos:
            msg_type_lower = video.type.lower()
            video_topics = topics.get(video.topic, [])
            spec_kind = video_spec_kind(video, video_topics)
            if spec_kind == "video_file":
                _append("video_file", video.key, video.topic, video_topics, None)
                continue
            if spec_kind == "video_stream":
                _append(
                    "video_stream", video.key, video.topic, video_topics,
                    COMPRESSED_VIDEO_MESSAGE_PATHS, resize=image_resize(video),
                )
                continue
            if "compressedimage" in msg_type_lower:
                paths = ["header", "format", "data"]
            elif "image" in msg_type_lower:
                paths = ["header", "height", "width", "encoding", "data"]
            else:
                paths = None
            _append(
                "video_msgs", video.key, video.topic,
                topics.get(video.topic, []), paths,
            )

        for action in contract.actions:
            _append(
                "action", action.unique_key, action.topic,
                topics.get(action.topic, []), None,
            )

        for observation in contract.observations:
            _append(
                "observation", observation.unique_key, observation.topic,
                topics.get(observation.topic, []), None,
            )

        for task in (contract.tasks or []):
            _append(
                "task", task.key, task.topic,
                topics.get(task.topic, []), None,
            )

        return work_list

    # ------------------------------------------------------------------
    # Private loaders — consume already-fetched data from ``raw_by_spec``
    # ------------------------------------------------------------------

    def _load_actions(self, contract, raw_by_spec, logger):
        """Decode action streams from the pre-fetched per-Topic DataFrames.

        The decoder handles message-type-specific logic including flattening
        nested arrays (e.g., JointTrajectory points).

        Uses unique_key (key.topic) to store data, ensuring multiple actions
        with the same base key but different topics are kept separate.
        """
        for action in contract.actions:
            unique_key = action.unique_key
            logger.debug("Loading action: %s (unique_key=%s) from topic %s",
                       action.key, unique_key, action.topic)

            selector_names = (action.selector or {}).get("names", [])
            # LeRobot-facing names (may be overridden by lerobot_names)
            lr_names = action.get_lerobot_selector_names()

            try:
                raw_df = _merge_spec_dfs(raw_by_spec.get(("action", unique_key), []))

                logger.debug("Loaded %d raw messages from %s", len(raw_df), action.topic)

                if len(raw_df) == 0:
                    logger.warning("No messages found for %s", action.topic)
                    self.actions[unique_key] = pd.DataFrame({"timestamp": []})
                    continue

                # Apply decoder to each message
                all_timestamps = []
                all_values = []

                for _, row in raw_df.iterrows():
                    decoded = decode_value(row, action)

                    # Check if decoder returned flattened structure (e.g., JointTrajectory)
                    if isinstance(decoded, dict) and decoded.get('is_flattened'):
                        # Decoder flattened nested array into multiple rows
                        all_timestamps.extend(decoded['timestamps'])
                        all_values.extend(decoded['values'])
                    else:
                        # Regular decoder - one value per message
                        all_timestamps.append(row["timestamp"])
                        all_values.append(decoded)

                # Create DataFrame with timestamp + values columns
                result = pd.DataFrame({
                    "timestamp": all_timestamps,
                    "values": all_values,
                })

                # Keep only monotonically increasing timestamps
                # This handles overlapping trajectory messages where a new trajectory may
                # start at a time earlier than the end of the previous trajectory
                initial_rows = len(result)
                result = keep_monotonic_timestamps(result)
                if len(result) < initial_rows:
                    logger.debug(
                        "Filtered action %s to monotonic timestamps: %d -> %d rows (%d removed)",
                        unique_key, initial_rows, len(result), initial_rows - len(result)
                    )

                # Populate resolved_features with LeRobot-facing names
                if selector_names:
                    self.resolved_features[unique_key] = (
                        len(selector_names),
                        lr_names,
                    )
                    logger.debug(
                        "Decoded action %s: %d features, %d rows",
                        unique_key, len(selector_names), len(result),
                    )
                else:
                    # Infer width from first row if selector.names is empty
                    width = len(all_values[0]) if all_values else 0
                    self.resolved_features[unique_key] = (width, [])
                    logger.debug(
                        "Decoded action %s: inferred %d features, %d rows",
                        unique_key, width, len(result),
                    )

            except Exception:
                logger.exception("Failed to load action %s", unique_key)
                raise

            self.actions[unique_key] = result

    def _load_observations(self, contract, raw_by_spec, logger):
        """Decode observation streams from the pre-fetched per-Topic DataFrames.

        The decoder approach:
        1. Load full messages from Roboto (no message_paths_include filtering)
        2. Apply type-based decoder to extract fields specified in selector.names
        3. Decoder handles field extraction (e.g., position.joint_name for JointState)

        Uses unique_key (key.topic) to store data, ensuring multiple observations
        with the same base key but different topics are kept separate.
        """
        for observation in contract.observations:
            unique_key = observation.unique_key
            logger.debug("Loading observation: %s (unique_key=%s) from topic %s",
                       observation.key, unique_key, observation.topic)

            selector_names = (observation.selector or {}).get("names", [])
            # LeRobot-facing names (may be overridden by lerobot_names)
            lr_names = observation.get_lerobot_selector_names()

            try:
                raw_df = _merge_spec_dfs(
                    raw_by_spec.get(("observation", unique_key), [])
                )

                logger.debug("Loaded %d raw messages from %s", len(raw_df), observation.topic)

                if len(raw_df) == 0:
                    logger.warning("No messages found for %s", observation.topic)
                    self.observations[unique_key] = pd.DataFrame({"timestamp": []})
                    continue

                # Apply decoder to each message
                timestamps = []
                decoded_values = []

                for _, row in raw_df.iterrows():
                    # The decoder expects the row as a message-like object
                    # Roboto returns rows with columns like "name", "position", etc.
                    decoded = decode_value(row, observation)
                    timestamps.append(row["timestamp"])
                    decoded_values.append(decoded)

                # Single ``values`` column of per-row numpy arrays — matches
                # what ``lerobot.generate_frames`` expects when it stacks /
                # merges array streams onto the reference timeline.
                result = pd.DataFrame({
                    "timestamp": timestamps,
                    "values": decoded_values,  # List of numpy arrays
                })

                # Populate resolved_features with LeRobot-facing names
                if selector_names:
                    self.resolved_features[unique_key] = (
                        len(selector_names),
                        lr_names,
                    )
                    logger.debug(
                        "Decoded observation %s: %d features, %d rows",
                        unique_key, len(selector_names), len(result),
                    )
                else:
                    # Infer width from first row if selector.names is empty
                    width = len(decoded_values[0]) if decoded_values else 0
                    self.resolved_features[unique_key] = (width, [])
                    logger.debug(
                        "Decoded observation %s: inferred %d features, %d rows",
                        unique_key, width, len(result),
                    )

            except Exception:
                logger.exception("Failed to load observation %s", unique_key)
                raise

            self.observations[unique_key] = result

    def _load_videos(self, contract, raw_by_spec, logger):
        """Wire each video spec's pre-fetched data into ``self.videos[video.key]``.

        Routes on :func:`video_spec_kind`:
          - ``video_file`` → rows flattened from ``topic.get_data()`` tuples in
            the fetch step (``video`` / ``avi_video`` / ``mp4_video``).
          - ``video_stream`` → already-decoded ``{timestamp, frame}`` rows from
            the compressed-video GOP decode in the fetch step.
          - ``video_msgs`` → the structured message-path DataFrame
            (``CompressedImage``, ``Image``).
        """
        for video in contract.videos:
            logger.debug("Loading video: %s from topic %s", video.key, video.topic)
            spec_kind = video_spec_kind(video, self.topics.get(video.topic, []))

            try:
                if spec_kind == "video_file":
                    data = _merge_video_rows(
                        raw_by_spec.get(("video_file", video.key), []),
                        _empty_video_df(),
                    )
                    if len(data) == 0:
                        logger.warning(
                            "No video frames in range for topic: %s", video.topic,
                        )
                    logger.debug(
                        "Loaded %d frames from %s (file-backed video)",
                        len(data), video.topic,
                    )
                elif spec_kind == "video_stream":
                    data = _merge_video_rows(
                        raw_by_spec.get(("video_stream", video.key), []),
                        _empty_video_stream_df(),
                    )
                    if len(data) == 0:
                        logger.warning(
                            "No compressed-video frames in range for topic: %s", video.topic,
                        )
                    logger.debug(
                        "Decoded %d frames from %s (compressed video)",
                        len(data), video.topic,
                    )
                else:
                    data = _merge_spec_dfs(
                        raw_by_spec.get(("video_msgs", video.key), [])
                    )
                    logger.debug("Loaded %d images from %s", len(data), video.topic)
                self.videos[video.key] = data
            except Exception:
                logger.exception("Failed to load video %s", video.key)
                raise

    def _load_tasks(self, contract, raw_by_spec, logger):
        if not contract.tasks:
            return
        for task in contract.tasks:
            logger.debug("Loading task: %s from topic %s", task.key, task.topic)
            try:
                data = _merge_spec_dfs(raw_by_spec.get(("task", task.key), []))
                logger.debug("Loaded %d messages from %s", len(data), task.topic)
                self.tasks[task.key] = data
            except Exception:
                logger.exception("Failed to load task %s", task.key)
                raise


    def get_timestamps_from_topic(self, topic_key: str) -> pd.Series:
        """Get the timestamps from a specific topic."""
        if topic_key in self.observations:
            return self.observations[topic_key]["timestamp"].sort_values().reset_index(drop=True)
        elif topic_key in self.actions:
            return self.actions[topic_key]["timestamp"].sort_values().reset_index(drop=True)
        elif topic_key in self.videos:
            return self.videos[topic_key]["timestamp"].sort_values().reset_index(drop=True)
        else:
            raise ValueError(f"Topic {topic_key} not found")
    
    def get_min_max_timestamps(self):
        """Get the minimum and maximum timestamps from all topics."""
        min_timestamp = None
        max_timestamp = None
        for data in [*self.observations.values(), *self.actions.values(), *self.videos.values()]:
            if min_timestamp is None:
                min_timestamp = data["timestamp"].min()
            else:
                min_timestamp = min(min_timestamp, data["timestamp"].min())
                
            if max_timestamp is None:
                max_timestamp = data["timestamp"].max()
            else:
                max_timestamp = max(max_timestamp, data["timestamp"].max())
        
        return min_timestamp, max_timestamp


def resolve_task_label(
    contract: Contract,
    episode_data: DataCollection,
    start_time_ns: int,
    end_time_ns: int,
    metadata_task: str | None,
) -> str:
    """Resolve the per-episode LeRobot ``task`` string.

    Precedence (highest first):

    1. ``metadata_task`` — the Roboto event's own ``task`` metadata field.
       When present and non-empty this always wins; it is an explicit
       per-episode annotation and the contract's ``tasks:`` stream is not
       consulted at all.
    2. The contract's ``tasks:`` stream, when ``metadata_task`` is absent
       (``None``/empty): the payload of the *first* ``std_msgs/msg/String``
       message from the *first* declared task spec whose timestamp falls
       within ``[start_time_ns, end_time_ns]`` (the episode's own window,
       not the buffered fetch window). If the contract declares more than
       one task spec, only the first is consulted and a warning names the
       ignored ones.
    3. ``"default"``, when neither of the above yields a value (no
       ``metadata_task``, no ``tasks:`` block, or no in-window message).

    Args:
        contract: The (already role-resolved) contract for this episode's
            dataset.
        episode_data: The ``DataCollection`` already fetched for this
            episode — ``episode_data.tasks[key]`` holds the raw, unsorted-
            by-window message rows for task spec ``key``.
        start_time_ns: Episode window start (unbuffered).
        end_time_ns: Episode window end (unbuffered).
        metadata_task: The Roboto event's ``task`` metadata value, if any.

    Raises:
        ValueError: The first in-window task message decodes to a
            non-string value.
    """
    from .logger import logger

    if metadata_task:
        return str(metadata_task)

    if contract.tasks:
        if len(contract.tasks) > 1:
            logger.warning(
                "Contract declares %d task specs (%s); only the first "
                "('%s') is used for the LeRobot task label.",
                len(contract.tasks),
                [t.key for t in contract.tasks],
                contract.tasks[0].key,
            )
        first_spec = contract.tasks[0]
        df = episode_data.tasks.get(first_spec.key)
        if df is not None and len(df):
            window = df[
                (df["timestamp"] >= start_time_ns)
                & (df["timestamp"] <= end_time_ns)
            ]
            if len(window):
                row = window.sort_values(by="timestamp").iloc[0]
                decoded = decode_value(row, first_spec)
                if not isinstance(decoded, str):
                    raise ValueError(
                        f"Task spec '{first_spec.key}' (topic "
                        f"'{first_spec.topic}') decoded to a non-string "
                        f"value ({type(decoded).__name__}); a "
                        "std_msgs/String payload is expected."
                    )
                return decoded

    return "default"


def _auto_bounded_tolerance_ms(fps: float) -> float:
    """Bounded default tolerance used when ``tolerance_ms``/``align`` is omitted.

    Mirrors ``runtime.stream_buffer.auto_bound_tolerance_ns`` exactly (same
    formula, milliseconds instead of nanoseconds): ``max(2/fps, 50ms)``. The
    live runtime independently bounds an unlimited tolerance to this same
    value to keep a stalled stream from feeding stale observations forever;
    resolving it here, at contract-load time, means the offline converter
    and the live runtime agree on the default without either side having to
    special-case "the other side's sentinel" at merge/sample time.
    """
    return max(2000.0 / fps, 50.0)


def _as_align(
    it: dict[str, Any],
    *,
    fps: float,
    is_image: bool = False,
) -> AlignSpec:
    """Parse an ``align:`` block from a contract stream entry.

    Tolerance semantics:

    - ``align:`` omitted entirely, or present with ``tolerance_ms`` omitted
      -> auto-bounded default, ``max(2/fps, 50ms)`` (see
      :func:`_auto_bounded_tolerance_ms`).
    - ``tolerance_ms: null`` -> unlimited (always carry forward / always
      pick nearest, however far).
    - ``tolerance_ms: 0`` -> rejected. ``0`` used to mean "unlimited" in
      older contracts; that spelling collided with "zero tolerance" and is
      no longer accepted.
    - ``tolerance_ms: <positive number>`` -> that bound, unchanged.

    The legacy ``strategy``/``tol_ms`` key spellings are no longer accepted
    — use ``method``/``tolerance_ms``.
    """
    blk = it.get("align")
    if blk is None:
        return AlignSpec(method="hold", tolerance_ms=_auto_bounded_tolerance_ms(fps))

    if "strategy" in blk or "tol_ms" in blk:
        raise ValueError(
            f"Stream '{it.get('key', '?')}': align uses 'method' and "
            "'tolerance_ms' (the legacy 'strategy'/'tol_ms' spellings are "
            "no longer supported)."
        )

    method = str(blk.get("method", "hold")).lower()

    if "tolerance_ms" not in blk:
        tolerance_ms: float | None = _auto_bounded_tolerance_ms(fps)
    else:
        raw = blk["tolerance_ms"]
        if raw is None:
            tolerance_ms = None
        elif raw == 0:
            raise ValueError(
                f"Stream '{it.get('key', '?')}': align.tolerance_ms: 0 is no "
                "longer supported — 0 previously meant 'unlimited'. Use "
                "'tolerance_ms: null' for unbounded, or omit 'tolerance_ms' "
                "for the auto-bounded default (max(2/fps, 50ms))."
            )
        else:
            tolerance_ms = float(raw)

    # Guard: images cannot use linear alignment
    if is_image and method == "linear":
        raise ValueError(
            f"Stream '{it.get('key', '?')}' is an image stream and cannot "
            f"use alignment method 'linear'. Use 'hold' or 'nearest'."
        )

    return AlignSpec(method=method, tolerance_ms=tolerance_ms)


def _parse_lerobot_names(
    selector: dict[str, Any] | None, key: str
) -> list[str] | None:
    """Extract and validate ``lerobot_names`` from a selector block.

    Returns ``None`` when no ``lerobot_names`` are specified.
    Raises ``ValueError`` if the length doesn't match ``selector.names``.
    """
    if not selector:
        return None
    lr_names = selector.get("lerobot_names")
    if lr_names is None:
        return None
    sel_names = selector.get("names", [])
    if len(lr_names) != len(sel_names):
        raise ValueError(
            f"lerobot_names length ({len(lr_names)}) != "
            f"selector.names length ({len(sel_names)}) for key '{key}': "
            f"lerobot_names={lr_names!r}, selector.names={sel_names!r}"
        )
    return lr_names


# Placeholder topic name used when a spec is role-bound and the real topic
# has not yet been resolved from dataset metadata. ``unique_key`` then falls
# back to a sanitized form of this placeholder, so feature names never
# collide across roles before resolution.
_UNRESOLVED_TOPIC_PREFIX = "__role__:"


def _topic_or_role(
    it: dict[str, Any], *, is_image: bool
) -> tuple[str, str | None]:
    """Validate exactly-one-of(topic, role) on a spec block.

    Returns ``(topic_placeholder_or_literal, role)``. When ``role`` is set,
    the returned topic is an internal placeholder that
    :func:`resolve_role_bindings` later replaces with the real topic name.
    Image streams cannot be role-bound: image topics live on a single file
    pinned by URL, which the role-discovery layer doesn't model.
    """
    topic = it.get("topic")
    role = it.get("role")
    if is_image and role is not None:
        raise ValueError(
            f"Spec '{it.get('key', '?')}' is an image stream; image streams "
            f"cannot be role-bound"
        )
    if bool(topic) == bool(role):
        raise ValueError(
            f"Spec '{it.get('key', '?')}' must set exactly one of 'topic' "
            f"or 'role' (got topic={topic!r}, role={role!r})"
        )
    if role is not None:
        return f"{_UNRESOLVED_TOPIC_PREFIX}{role}", str(role)
    return str(topic), None


def _as_qos(it: dict[str, Any]) -> QosSpec | None:
    """Parse a ``qos:`` block from a contract stream entry.

    Returns ``None`` when no block is present so codegen can dispatch on
    type and pick a sensible default (sensor topics ⇒ BEST_EFFORT;
    state/action ⇒ RELIABLE). When the block is present, every field is
    optional and falls back to the most permissive (``BEST_EFFORT /
    KEEP_LAST / 1``) so an author writing only ``qos: {depth: 5}`` gets
    the other fields filled.

    Enum values are uppercased before validation — YAML authors
    frequently write ``best_effort`` and ``keep_last`` in lowercase; the
    generated node ultimately substitutes the ROS-2 enum spelling, so
    normalising at parse time keeps the schema permissive.
    """
    blk = it.get("qos")
    if blk is None:
        return None
    return QosSpec(
        reliability=str(blk.get("reliability", "BEST_EFFORT")).upper(),
        history=str(blk.get("history", "KEEP_LAST")).upper(),
        depth=int(blk.get("depth", 1)),
    )


def _parse_transforms(it: dict[str, Any]) -> list[TransformSpec] | None:
    """Extract and parse ``transforms`` from a spec block."""
    raw = it.get("transforms")
    if not raw:
        return None
    return [
        TransformSpec(
            type=t["type"],
            params={k: v for k, v in t.items() if k not in ("type", "stage")},
            stage=t.get("stage", "post"),
        )
        for t in raw
    ]


def _spec_origin(spec: Any) -> str:
    """Human-readable "where did this come from" for a parsed spec.

    Used in error messages that must name a conflicting spec: shows the
    role for role-bound specs (whose ``topic`` is still the unresolved
    ``__role__:<role>`` placeholder at load time) and the literal topic
    otherwise.
    """
    if spec.role is not None:
        return f"role={spec.role!r}"
    return f"topic={spec.topic!r}"


def load_contract(path: Path | str) -> Contract:
    """Load + normalize contract YAML into dataclasses."""
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    fps = float(d.get("fps", 20.0))

    def _obs(it: dict[str, Any]) -> ObservationSpec:
        image = it.get("image")
        if image is not None and "depth" in image:
            raise ValueError(
                "depth images are not supported yet; remove the image.depth "
                "block to load this contract. If you need depth support, we "
                "would like to hear about it — open an issue or a PR at "
                "https://github.com/roboto-ai/roboto-physical-ai-actions"
            )
        is_image = image is not None
        selector = it.get("selector")
        topic, role = _topic_or_role(it, is_image=is_image)
        return ObservationSpec(
            key=it["key"],
            topic=topic,
            type=it["type"],
            role=role,
            selector=selector,
            image=image,
            align=_as_align(it, fps=fps, is_image=is_image),
            lerobot_names=_parse_lerobot_names(selector, it["key"]),
            transforms=_parse_transforms(it),
            qos=_as_qos(it),
        )

    def _act(it: dict[str, Any]) -> ActionSpec:
        if it.get("publish") is not None:
            raise ValueError(
                "'publish:' is no longer supported; set 'topic:' and "
                "'type:' directly on the action spec"
            )
        topic, role = _topic_or_role(it, is_image=False)
        msg_type = it["type"]
        selector = it.get("selector")
        safety_behavior = str(it.get("safety_behavior", "publish_nothing"))
        if safety_behavior not in VALID_SAFETY_BEHAVIORS:
            raise ValueError(
                f"Action '{it['key']}' has unknown safety_behavior "
                f"'{safety_behavior}'. Valid options: {VALID_SAFETY_BEHAVIORS}. "
                "(codegen only honours 'publish_nothing'; the others "
                "parse but gen-node refuses them.)"
            )
        return ActionSpec(
            key=it["key"],
            topic=topic,
            type=msg_type,
            role=role,
            selector=selector,
            align=_as_align(it, fps=fps),
            lerobot_names=_parse_lerobot_names(selector, it["key"]),
            transforms=_parse_transforms(it),
            qos=_as_qos(it),
            safety_behavior=safety_behavior,
        )

    def _task(it: dict[str, Any]) -> TaskSpec:
        return TaskSpec(
            key=it.get("key", it["topic"]),
            topic=it["topic"],
            type=it["type"],
        )

    def _spec_key_for_error(it: Any) -> str:
        return it.get("key", "?") if isinstance(it, dict) else "?"

    obs = []
    videos = []
    for idx, it in enumerate(d.get("observations") or []):
        try:
            new_obs = _obs(it)
        except (KeyError, TypeError) as e:
            raise ValueError(
                f"observations[{idx}] (key {_spec_key_for_error(it)!r}): {e}"
            ) from e
        if new_obs.image is not None:
            videos.append(new_obs)
        else:
            obs.append(new_obs)

    acts = []
    for idx, it in enumerate(d.get("actions") or []):
        try:
            acts.append(_act(it))
        except (KeyError, TypeError) as e:
            raise ValueError(
                f"actions[{idx}] (key {_spec_key_for_error(it)!r}): {e}"
            ) from e

    tks = []
    for idx, it in enumerate(d.get("tasks") or []):
        try:
            tks.append(_task(it))
        except (KeyError, TypeError) as e:
            key = it.get("key", it.get("topic", "?")) if isinstance(it, dict) else "?"
            raise ValueError(f"tasks[{idx}] (key {key!r}): {e}") from e

    # Videos are 1:1 with their key — multiple specs sharing a key would
    # silently overwrite each other in the output dataset.
    seen_video_keys: set[str] = set()
    for v in videos:
        if v.key in seen_video_keys:
            raise ValueError(
                f"Duplicate video key '{v.key}': each video spec must declare a unique key."
            )
        seen_video_keys.add(v.key)

    # Validate uniqueness of lerobot feature names across the contract.
    # Tracks which spec produced each name so a collision names both
    # conflicting specs (key + topic/role), not just the duplicated name.
    name_origin: dict[str, Any] = {}
    for spec in obs + videos + acts:
        for name in spec.get_lerobot_selector_names():
            if name in name_origin:
                prev = name_origin[name]
                raise ValueError(
                    f"Duplicate lerobot feature name '{name}': used by "
                    f"spec key={prev.key!r} ({_spec_origin(prev)}) and "
                    f"spec key={spec.key!r} ({_spec_origin(spec)})"
                )
            name_origin[name] = spec

    return Contract(
        name=d.get("name", "contract"),
        version=int(d.get("version", 1)),
        fps=fps,
        observations=obs,
        videos=videos,
        actions=acts,
        tasks=tks,
        robot_type=d.get("robot_type"),
        action_lead_steps=int(d.get("action_lead_steps", 0)),
    )

