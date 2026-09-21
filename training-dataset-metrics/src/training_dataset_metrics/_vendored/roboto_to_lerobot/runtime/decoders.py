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
ROS message decoders for converting ROS messages to numpy arrays.

Derived from Rosetta's ``rosetta/common/decoders.py`` at commit 413fc9b:
https://github.com/iblnkn/rosetta/blob/413fc9b96418da3d900d0e736c8c1facd93cfbbc/rosetta/common/decoders.py
License text: ``LICENSE-rosetta`` in this directory.

Each decoder is self-contained and registered with @register_decoder.
If you need to decode a message type that isn't here, add a new decoder.

Decoders declare their output dtype at registration time. The declaration
is informational only — it is not stored on the registry and no caller
reads it back today (see ``register_decoder``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..extract import (
    DEPTH_ENCODINGS,
    decompress_image,
    reshape_raw_image,
)
from ..video import COMPRESSED_VIDEO_SCHEMAS
from .converters import register_decoder
from .image import depth_to_uint8_rgb

if TYPE_CHECKING:
    from ..contract_utils import ActionSpec, ObservationSpec


# =============================================================================
# Image Decoders (video dtype)
#
# Signature: (msg, spec) -> np.ndarray, matching the scalar decoder contract.
# For video decoders, ``msg`` is a pandas Series whose columns are whatever
# ``DataCollection._load_videos`` produced for that message type — see
# ``lerobot.generate_frames`` for how payload columns are plumbed through.
# Every video decoder MUST return an HWC uint8 RGB array.
# =============================================================================


@register_decoder("sensor_msgs/msg/CompressedImage", dtype="video")
def _dec_compressed_image(msg: Any, spec: ObservationSpec) -> np.ndarray:
    """Decode a sensor_msgs/CompressedImage (or file-backed video) row to HWC uint8 RGB.

    Expects ``msg["data"]`` (raw encoded bytes) and ``msg["format"]``
    (format hint such as ``"jpeg"`` or ``"png"`` — informational only,
    cv2 auto-detects from magic bytes).
    """
    return decompress_image(msg["data"], msg["format"])


# File-backed video frames (``video`` / ``avi_video`` / ``mp4_video`` contract
# types — see ``contract_utils._IMAGE_VIDEO_TYPES``) all surface as
# ``(timestamp, format, data)`` rows whose ``data`` field is a single encoded
# frame. They reuse the CompressedImage path verbatim.
for _alias in ("video", "avi_video", "mp4_video"):
    register_decoder(_alias, dtype="video")(_dec_compressed_image)
del _alias


def _dec_compressed_video(msg: Any, spec: ObservationSpec) -> np.ndarray:
    """Return the already-decoded frame of a compressed-video row.

    Compressed video (``foxglove_msgs/msg/CompressedVideo``) stores one encoded
    access unit per message, and a delta frame needs its whole GOP prefix to
    decode — there is no per-message decode for this registry to perform. Those
    frames are decoded a range at a time during the fetch step (see
    ``roboto_to_lerobot.video.decode_video_stream_rows``) and arrive here
    already as HWC uint8 RGB, under a ``frame`` column.

    The registration exists so compressed video still resolves through the one
    registry every video path goes through, including the deferred-decode round
    trip in ``lerobot.materialize_deferred``.
    """
    frame = msg["frame"]
    if not isinstance(frame, np.ndarray):
        raise ValueError(
            f"Topic '{spec.topic}' compressed-video row carries "
            f"{type(frame).__name__} instead of a decoded frame array."
        )
    return frame


# Every schema spelling the compressed-video loader routes on must resolve to a
# decoder; ``contract_utils.video_spec_kind`` refuses a spec whose declared type
# is missing from this registry, so the two lists cannot drift apart silently.
for _video_schema in COMPRESSED_VIDEO_SCHEMAS:
    register_decoder(_video_schema, dtype="video")(_dec_compressed_video)
del _video_schema


# -----------------------------------------------------------------------------
# TEMP: Roboto-SDK-workaround magic bytes.
#
# roboto.Topic.get_data* currently returns a downsampled 8-bit JPEG preview
# for sensor_msgs/msg/Image topics (visualization asset substitution). When
# those bytes arrive at _dec_raw_image the advertised encoding (e.g.
# "16UC1") does not match the payload: it's really an encoded JPEG/PNG
# container.
#
# Until the SDK exposes a raw-bytes representation, treat a depth-encoded row
# whose data bytes are a compressed container as if it were a CompressedImage:
# decode with cv2 → HWC uint8 RGB → hand back to the resize step in
# lerobot.generate_frames. This skips depth_to_uint8_rgb entirely, so the
# contract's units/range/colormap/invalid keys are NOT honoured for depth
# today. The depth_to_uint8_rgb pipeline is kept intact so that the day the
# SDK serves real 16-bit depth, only this TEMP block needs to be removed.
# -----------------------------------------------------------------------------
_TEMP_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_TEMP_JPEG_MAGIC = b"\xff\xd8\xff"


def _TEMP_looks_like_compressed_image(data: bytes) -> bool:
    return (
        (len(data) >= 8 and data[:8] == _TEMP_PNG_MAGIC)
        or (len(data) >= 3 and data[:3] == _TEMP_JPEG_MAGIC)
    )


@register_decoder("sensor_msgs/msg/Image", dtype="video")
def _dec_raw_image(msg: Any, spec: ObservationSpec) -> np.ndarray:
    """Decode a raw sensor_msgs/Image row to HWC uint8 RGB.

    Dispatches on ``msg["encoding"]``. Color encodings (rgb8/bgr8/mono8) are
    reshaped in place; depth encodings (mono16/16UC1/32FC1) are normalised to
    uint8 via the ``image.depth`` block on the contract spec.
    """
    encoding = str(msg["encoding"])
    height = int(msg["height"])
    width = int(msg["width"])
    data = msg["data"]

    # ------------------- TEMP SDK workaround (depth only) --------------------
    # Remove this block when roboto Topic.get_data* can serve original bytes.
    if encoding in DEPTH_ENCODINGS and _TEMP_looks_like_compressed_image(data):
        # SDK handed us a JPEG/PNG preview instead of raw depth; fall back to
        # the same path as CompressedImage — no depth math, just a color frame
        # that the downstream resize step will handle.
        return decompress_image(data, "jpeg")
    # ------------------------- end TEMP workaround --------------------------

    arr = reshape_raw_image(data, height, width, encoding)

    if encoding in DEPTH_ENCODINGS:
        depth_cfg = (spec.image or {}).get("depth")
        if not depth_cfg:
            raise ValueError(
                f"Topic '{spec.topic}' has depth encoding '{encoding}' but the "
                f"contract spec for '{spec.key}' is missing an 'image.depth' block."
            )
        return depth_to_uint8_rgb(arr, encoding, depth_cfg)

    if encoding in ("mono8", "8UC1"):
        return np.stack([arr, arr, arr], axis=-1)
    if encoding == "rgb8":
        return arr
    if encoding == "bgr8":
        return arr[..., ::-1].copy()
    if encoding in ("rgba8", "bgra8"):
        rgb = arr[..., :3]
        if encoding == "bgra8":
            rgb = rgb[..., ::-1]
        return np.ascontiguousarray(rgb)

    raise ValueError(f"Unhandled Image encoding: {encoding!r}")


# =============================================================================
# JointState Decoder
# =============================================================================


@register_decoder("sensor_msgs/msg/JointState", dtype="float64")
def _dec_joint_state(msg: Any, spec: ObservationSpec | ActionSpec) -> np.ndarray:
    """Decode sensor_msgs/JointState.

    With selector names like ["position.joint1", "velocity.joint2"]:
      - Extracts specified fields by joint name lookup
    With bare names like ["joint1", "joint2"]:
      - Defaults to position field
    Without names:
      - Returns all positions

    Args:
        msg: Pandas Series from Roboto with columns like 'name', 'position', 'velocity', 'effort'
        spec: ObservationSpec or ActionSpec with selector.names for joint filtering
    """
    selector_names = (spec.selector or {}).get("names", [])

    # Handle both pandas Series (from Roboto) and ROS message objects
    if hasattr(msg, '__getitem__'):  # pandas Series
        name_list = msg['name']
        position_list = msg['position']
    else:  # ROS message object
        name_list = msg.name
        position_list = msg.position

    if not selector_names:
        if position_list:
            return np.asarray(position_list, dtype=np.float64)
        return np.array([], dtype=np.float64)

    name_to_idx = {name: i for i, name in enumerate(name_list)}
    out = []

    for selector in selector_names:
        # Support both "field.joint_name" and bare "joint_name" (defaults to position)
        if "." in selector:
            field, joint_name = selector.split(".", 1)
        else:
            field, joint_name = "position", selector

        if joint_name not in name_to_idx:
            raise ValueError(
                f"Joint '{joint_name}' not in message. Available: {list(name_list)}"
            )
        idx = name_to_idx[joint_name]

        # Get the field array
        if hasattr(msg, '__getitem__'):  # pandas Series
            arr = msg[field]
        else:  # ROS message object
            arr = getattr(msg, field)

        if idx >= len(arr):
            raise ValueError(f"Index {idx} out of range for {field} (len={len(arr)})")
        out.append(float(arr[idx]))

    return np.asarray(out, dtype=np.float64)


# =============================================================================
# JointTrajectory Decoder (Custom Extension - Not in Rosetta)
# =============================================================================


@register_decoder("trajectory_msgs/msg/JointTrajectory", dtype="float64")
def _dec_joint_trajectory(msg: Any, spec: ActionSpec) -> dict:
    """Decode trajectory_msgs/JointTrajectory.

    NOTE: This is a custom extension not present in Rosetta. Rosetta uses
    JointState for actions instead of JointTrajectory.

    This decoder handles the nested structure of JointTrajectory messages.
    It returns a dict with 'timestamps' and 'values' arrays that can be
    used to create multiple rows (one per trajectory point).

    Args:
        msg: Pandas Series from Roboto with flattened columns like:
             - 'header.stamp.sec', 'header.stamp.nanosec'
             - 'joint_names' (list)
             - 'points' (defaultlist of point objects)
        spec: ActionSpec with selector.names for joint filtering

    Returns:
        dict with:
            - 'timestamps': list of absolute timestamps (int64 nanoseconds)
            - 'values': list of numpy arrays (one per point)
            - 'is_flattened': True (signals this needs special handling)
    """
    selector_names = (spec.selector or {}).get("names", [])

    # Roboto flattens nested messages into dot-notation columns
    # Access header timestamp from flattened columns
    if 'header.stamp.sec' in msg.index:
        # Roboto's flattened format
        base_ns = int(msg['header.stamp.sec'] * 1_000_000_000 + msg['header.stamp.nanosec'])
        joint_names = msg['joint_names']
        points = msg['points']
    elif hasattr(msg, 'header'):
        # ROS message object format (for future compatibility)
        base_ns = int(msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec)
        joint_names = msg.joint_names
        points = msg.points
    else:
        raise ValueError(f"Unexpected message format. Available keys: {list(msg.index)}")

    if not points:
        return {
            'timestamps': [],
            'values': [],
            'is_flattened': True
        }

    # Map joint names to indices if selector specified
    if selector_names:
        name_to_idx = {name: i for i, name in enumerate(joint_names)}
        indices = []
        for joint_name in selector_names:
            if joint_name not in name_to_idx:
                raise ValueError(
                    f"Joint '{joint_name}' not in trajectory. Available: {list(joint_names)}"
                )
            indices.append(name_to_idx[joint_name])
    else:
        indices = None

    # Extract each point as a separate row
    timestamps = []
    values = []

    for point in points:
        # Points are objects with attributes (from Roboto's defaultlist)
        # Access time_from_start and positions
        if hasattr(point, 'time_from_start'):
            # Object with attributes
            offset_ns = int(point.time_from_start.sec * 1_000_000_000 + point.time_from_start.nanosec)
            positions = point.positions
        elif isinstance(point, dict):
            # Dict format (fallback)
            time_from_start = point['time_from_start']
            offset_ns = int(time_from_start['sec'] * 1_000_000_000 + time_from_start['nanosec'])
            positions = point['positions']
        else:
            raise ValueError(f"Unexpected point format: {type(point)}")

        abs_timestamp = base_ns + offset_ns
        timestamps.append(abs_timestamp)

        # Extract positions (filtered by selector if specified)
        if indices is not None:
            point_values = np.array([positions[i] for i in indices], dtype=np.float32)
        else:
            point_values = np.array(positions, dtype=np.float32)

        values.append(point_values)

    return {
        'timestamps': timestamps,
        'values': values,
        'is_flattened': True
    }


# =============================================================================
# MultiDOFCommand Decoder (From Rosetta)
# =============================================================================


@register_decoder("control_msgs/msg/MultiDOFCommand", dtype="float64")
def _dec_multidof_command(msg: Any, spec: ObservationSpec) -> np.ndarray:
    """Decode control_msgs/MultiDOFCommand.

    With selector names like ["values.joint1", "values_dot.joint1"]:
      - Extracts specified DOF values by name
    Without names:
      - Returns [values, values_dot] concatenated
    """
    selector_names = (spec.selector or {}).get("names", [])

    if not selector_names:
        values = np.asarray(msg.values, dtype=np.float64) if msg.values else np.array([], dtype=np.float64)
        values_dot = np.asarray(msg.values_dot, dtype=np.float64) if msg.values_dot else np.array([], dtype=np.float64)
        return np.concatenate([values, values_dot])

    dof_index = {name: i for i, name in enumerate(msg.dof_names)}
    out = []

    for selector in selector_names:
        if selector.startswith("values_dot."):
            dof_name = selector[11:]
            arr = msg.values_dot
        elif selector.startswith("values."):
            dof_name = selector[7:]
            arr = msg.values
        else:
            dof_name = selector
            arr = msg.values

        if dof_name not in dof_index:
            raise ValueError(
                f"DOF '{dof_name}' not in message. Available: {list(msg.dof_names)}"
            )
        idx = dof_index[dof_name]
        if idx >= len(arr):
            raise ValueError(f"Index {idx} out of range (len={len(arr)})")
        out.append(float(arr[idx]))

    return np.asarray(out, dtype=np.float64)


# =============================================================================
# Helper for dot-notation field access
# =============================================================================


def _dot_get(obj: Any, path: str) -> Any:
    """Resolve a dotted field path against a decoded message.

    Two message shapes reach the decoders. Offline (batch conversion), ``obj``
    is a pandas Series whose columns are Roboto's *already-flattened*
    dot-notation labels — ``'angular_velocity.x'`` is a single column, so the
    whole path is indexed in one shot. Live (rclpy replay), ``obj`` is a ROS
    message object whose nesting is real, so it is walked attribute by
    attribute. Bracket-indexing a ROS object (or a Series missing the column)
    raises, which is the signal to fall back to attribute walking.
    """
    try:
        return obj[path]
    except (KeyError, TypeError, IndexError):
        for part in path.split("."):
            obj = getattr(obj, part)
        return obj


# =============================================================================
# IMU Decoder
# =============================================================================


@register_decoder("sensor_msgs/msg/Imu", dtype="float64")
def _dec_imu(msg: Any, spec: ObservationSpec) -> np.ndarray:
    """Decode sensor_msgs/Imu.

    With selector names: extracts specified dotted paths
    Without names: returns [quat(4), angular_vel(3), linear_accel(3)]
    """
    selector_names = (spec.selector or {}).get("names", [])

    if not selector_names:
        return np.concatenate([
            np.array([
                msg.orientation.x, msg.orientation.y,
                msg.orientation.z, msg.orientation.w
            ], dtype=np.float64),
            np.array([
                msg.angular_velocity.x, msg.angular_velocity.y,
                msg.angular_velocity.z
            ], dtype=np.float64),
            np.array([
                msg.linear_acceleration.x, msg.linear_acceleration.y,
                msg.linear_acceleration.z
            ], dtype=np.float64),
        ])

    return np.asarray([float(_dot_get(msg, name)) for name in selector_names], dtype=np.float64)


# =============================================================================
# Odometry Decoder
# =============================================================================


@register_decoder("nav_msgs/msg/Odometry", dtype="float64")
def _dec_odometry(msg: Any, spec: ObservationSpec) -> np.ndarray:
    """Decode nav_msgs/Odometry.

    With selector names: extracts specified dotted paths
    Without names: returns [position(3), orientation_quat(4)]
    """
    selector_names = (spec.selector or {}).get("names", [])

    if not selector_names:
        return np.concatenate([
            np.array([
                msg.pose.pose.position.x, msg.pose.pose.position.y,
                msg.pose.pose.position.z
            ], dtype=np.float64),
            np.array([
                msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z, msg.pose.pose.orientation.w
            ], dtype=np.float64),
        ])

    return np.asarray([float(_dot_get(msg, name)) for name in selector_names], dtype=np.float64)


# =============================================================================
# Twist Decoder
# =============================================================================


@register_decoder("geometry_msgs/msg/Twist", dtype="float64")
def _dec_twist(msg: Any, spec: ObservationSpec) -> np.ndarray:
    """Decode geometry_msgs/Twist.

    With selector names: extracts specified dotted paths
    Without names: returns [linear(3), angular(3)]
    """
    selector_names = (spec.selector or {}).get("names", [])

    if not selector_names:
        return np.concatenate([
            np.array([msg.linear.x, msg.linear.y, msg.linear.z], dtype=np.float64),
            np.array([msg.angular.x, msg.angular.y, msg.angular.z], dtype=np.float64),
        ])

    return np.asarray([float(_dot_get(msg, name)) for name in selector_names], dtype=np.float64)



# =============================================================================
# Array Decoders
# =============================================================================


@register_decoder("std_msgs/msg/Float32MultiArray", dtype="float32")
def _dec_float32_array(msg: Any, spec: ObservationSpec) -> np.ndarray:
    """Decode std_msgs/Float32MultiArray to float32 array."""
    return np.asarray(msg.data, dtype=np.float32)


@register_decoder("std_msgs/msg/Float64MultiArray", dtype="float64")
def _dec_float64_array(msg: Any, spec: ObservationSpec) -> np.ndarray:
    """Decode std_msgs/Float64MultiArray to float64 array."""
    return np.asarray(msg.data, dtype=np.float64)


@register_decoder("std_msgs/msg/Int32MultiArray", dtype="int32")
def _dec_int32_array(msg: Any, spec: ObservationSpec) -> np.ndarray:
    """Decode std_msgs/Int32MultiArray to int32 array."""
    return np.asarray(msg.data, dtype=np.int32)


# =============================================================================
# Scalar Decoders
# =============================================================================


@register_decoder("std_msgs/msg/Float32", dtype="float32")
def _dec_float32(msg: Any, spec: ObservationSpec) -> np.ndarray:
    """Decode std_msgs/Float32 to float32 scalar."""
    return np.array([msg.data], dtype=np.float32)


@register_decoder("std_msgs/msg/Float64", dtype="float64")
def _dec_float64(msg: Any, spec: ObservationSpec) -> np.ndarray:
    """Decode std_msgs/Float64 to float64 scalar."""
    return np.array([msg.data], dtype=np.float64)


@register_decoder("std_msgs/msg/Int32", dtype="int32")
def _dec_int32(msg: Any, spec: ObservationSpec) -> np.ndarray:
    """Decode std_msgs/Int32 to int32 scalar."""
    return np.array([msg.data], dtype=np.int32)


@register_decoder("std_msgs/msg/Int64", dtype="int64")
def _dec_int64(msg: Any, spec: ObservationSpec) -> np.ndarray:
    """Decode std_msgs/Int64 to int64 scalar."""
    return np.array([msg.data], dtype=np.int64)


@register_decoder("std_msgs/msg/String", dtype="string")
def _dec_string(msg: Any, spec: ObservationSpec) -> str:
    """Decode std_msgs/String to Python string."""
    return str(msg.data)


# =============================================================================
# String-Typed Message Decoders
# =============================================================================


@register_decoder("string_typed_msg", dtype="float64")
def _dec_string_typed_msg(msg: Any, spec: ObservationSpec | ActionSpec) -> np.ndarray:
    """Decode string-typed messages where field values are stored as strings/objects.

    Message structure (pandas Series from Roboto):
    - Index contains field names (e.g., "effector_pose_0_0", "motor_currents_4")
    - Values contain the data as strings/objects that need float conversion

    With selector names like ["effector_pose_0_0", "effector_pose_0_1"]:
      - Extracts values at those index keys
      - Converts the string values to floats
    Without selector names:
      - Raises an error (selector.names is required for this decoder)

    Args:
        msg: Pandas Series from Roboto with field names as index
        spec: ObservationSpec or ActionSpec with selector.names for field selection

    Returns:
        numpy array of float64 values in the order specified by selector.names
    """
    selector_names = (spec.selector or {}).get("names", [])

    if not selector_names:
        raise ValueError(
            "string_typed_msg decoder requires selector.names to specify which fields to extract"
        )

    # Extract values by index key and convert to float
    out = []
    for field_name in selector_names:
        if field_name not in msg.index:
            raise ValueError(
                f"Field '{field_name}' not in message. Available: {list(msg.index)}"
            )
        out.append(float(msg[field_name]))

    return np.asarray(out, dtype=np.float64)


# =============================================================================
# ANYmal quadruped decoders (custom ANYmal ROS1 message types)
#
# The ANYmal legged robot logs proprioception and per-actuator commands in
# custom message types that are absent from the standard ROS catalog, so the
# generic decoders above cannot reach their fields. Both decoders read the
# offline shape — a pandas Series whose nested fields Roboto has flattened into
# dot-notation columns (parallel-array joints) or a list of per-actuator dicts —
# and fall back to attribute access so the live rclpy path keeps working.
# =============================================================================

# ANYmal series-elastic actuator readings carry no per-element joint name on the
# wire (the ``name`` field is empty), so the twelve readings are addressed
# positionally. This is the leg-joint order the driver publishes, and it matches
# the order the state estimator reports in ``AnymalState.joints`` — verified on
# real data by ``actuator_readings[i].state.joint_position ==
# AnymalState.joints.position[i]`` element for element.
_ANYMAL_JOINT_ORDER = (
    "LF_HAA", "LF_HFE", "LF_KFE",
    "RF_HAA", "RF_HFE", "RF_KFE",
    "LH_HAA", "LH_HFE", "LH_KFE",
    "RH_HAA", "RH_HFE", "RH_KFE",
)

# Fields addressable inside an AnymalState ``joints`` sub-message. A bare
# selector name (no dotted prefix) defaults to ``position``.
_ANYMAL_STATE_JOINT_FIELDS = ("position", "velocity", "acceleration", "effort")


@register_decoder("anymal_msgs/AnymalState", dtype="float64")
def _dec_anymal_state(msg: Any, spec: ObservationSpec | ActionSpec) -> np.ndarray:
    """Decode anymal_msgs/AnymalState joint fields.

    ``AnymalState`` bundles the estimator output; the proprioceptive joint
    arrays live under ``joints`` as parallel ``name`` / ``position`` /
    ``velocity`` / ``acceleration`` / ``effort`` lists (Roboto flattens these
    to ``joints.name`` etc. columns offline).

    Selector names pick joints by name, optionally prefixed with the field:
      - ``"LF_HAA"``            -> ``joints.position`` for that joint
      - ``"velocity.LF_HAA"``   -> ``joints.velocity`` for that joint
    Without selector names, returns every joint's position in message order.

    Args:
        msg: Pandas Series with ``joints.name`` / ``joints.position`` / ...
            columns (offline), or an AnymalState-like object (live).
        spec: ObservationSpec or ActionSpec whose ``selector.names`` choose
            joints and, optionally, the field per joint.
    """
    selector_names = (spec.selector or {}).get("names", [])

    is_series = hasattr(msg, "__getitem__")
    names = msg["joints.name"] if is_series else msg.joints.name

    def _field(field: str):
        return msg[f"joints.{field}"] if is_series else getattr(msg.joints, field)

    if not selector_names:
        positions = _field("position")
        return np.asarray(positions, dtype=np.float64)

    name_to_idx = {name: i for i, name in enumerate(names)}
    out = []
    for selector in selector_names:
        if "." in selector:
            field, joint_name = selector.split(".", 1)
        else:
            field, joint_name = "position", selector

        if field not in _ANYMAL_STATE_JOINT_FIELDS:
            raise ValueError(
                f"Unknown AnymalState joint field '{field}'. "
                f"Expected one of {list(_ANYMAL_STATE_JOINT_FIELDS)}."
            )
        if joint_name not in name_to_idx:
            raise ValueError(
                f"Joint '{joint_name}' not in message. Available: {list(names)}"
            )
        arr = _field(field)
        idx = name_to_idx[joint_name]
        if idx >= len(arr):
            raise ValueError(
                f"Index {idx} out of range for joints.{field} (len={len(arr)})"
            )
        out.append(float(arr[idx]))

    return np.asarray(out, dtype=np.float64)


def _read_nested(element: Any, path: str) -> Any:
    """Read a dot-notation ``path`` from one actuator reading.

    Offline the reading is a nested ``dict``; live it is a ROS message object.
    Each path segment is indexed as a mapping key when possible and otherwise
    read as an attribute, so both shapes resolve with the same call.
    """
    for part in path.split("."):
        element = element[part] if isinstance(element, dict) else getattr(element, part)
    return element


@register_decoder("series_elastic_actuator_msgs/SeActuatorReadings", dtype="float64")
def _dec_se_actuator_readings(msg: Any, spec: ObservationSpec | ActionSpec) -> np.ndarray:
    """Decode series_elastic_actuator_msgs/SeActuatorReadings.

    One message carries a ``readings`` list with one entry per leg actuator.
    Each entry has a ``commanded`` sub-message (the controller/policy setpoint:
    ``position``, ``velocity``, ``joint_torque``, ...) and a ``state``
    sub-message (the measured actuator: ``joint_position``, ``joint_velocity``,
    ``joint_torque``, ...). The wire message has no per-actuator name, so the
    entries are addressed positionally via ``_ANYMAL_JOINT_ORDER``.

    Selector names are ``<field-path>.<joint>``: the joint is the final token
    and everything before it is the dotted field path into the reading. A bare
    joint name (no field path) defaults to ``commanded.position`` — the policy's
    joint-position action. Examples:
      - ``"LF_HAA"``                       -> ``commanded.position``
      - ``"commanded.velocity.LF_HAA"``    -> commanded velocity
      - ``"state.joint_position.LF_HAA"``  -> measured position
    Without selector names, returns ``commanded.position`` for every joint in
    ``_ANYMAL_JOINT_ORDER``.

    Args:
        msg: Pandas Series with a ``readings`` column (offline), or an
            SeActuatorReadings-like object (live).
        spec: ObservationSpec or ActionSpec whose ``selector.names`` choose the
            joint and, optionally, the field path per joint.
    """
    readings = msg["readings"] if hasattr(msg, "__getitem__") else msg.readings

    n = len(_ANYMAL_JOINT_ORDER)
    if len(readings) != n:
        raise ValueError(
            f"Expected {n} actuator readings ({list(_ANYMAL_JOINT_ORDER)}), "
            f"got {len(readings)}; the positional joint mapping cannot be applied."
        )
    by_joint = {joint: readings[i] for i, joint in enumerate(_ANYMAL_JOINT_ORDER)}

    selector_names = (spec.selector or {}).get("names", [])
    if not selector_names:
        return np.asarray(
            [float(_read_nested(by_joint[j], "commanded.position"))
             for j in _ANYMAL_JOINT_ORDER],
            dtype=np.float64,
        )

    out = []
    for selector in selector_names:
        *field_parts, joint_name = selector.split(".")
        field_path = ".".join(field_parts) if field_parts else "commanded.position"
        if joint_name not in by_joint:
            raise ValueError(
                f"Joint '{joint_name}' not in ANYmal actuator order. "
                f"Available: {list(_ANYMAL_JOINT_ORDER)}"
            )
        out.append(float(_read_nested(by_joint[joint_name], field_path)))

    return np.asarray(out, dtype=np.float64)

