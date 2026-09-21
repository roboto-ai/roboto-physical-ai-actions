from typing import Any

import cv2
import numpy as np
import pandas as pd

from .logger import logger

NANO_SEC_PER_SEC = 1_000_000_000


def ros_time_to_nanoseconds(sec: int, nsec: int) -> int:
    """
    Convert ROS timestamp (sec + nsec) to nanoseconds since Unix epoch.

    Args:
        sec: Seconds component of ROS timestamp
        nsec: Nanoseconds component of ROS timestamp

    Returns:
        int64 nanoseconds since Unix epoch
    """
    return int(sec * NANO_SEC_PER_SEC + nsec)



def keep_monotonic_timestamps(df: pd.DataFrame):
    """
    Keep only rows where timestamps are monotonically increasing.
    When a timestamp decreases, pop previous rows until finding one less than the current value.

    Args:
        df: DataFrame with a 'timestamp' column

    Returns:
        DataFrame with only monotonically increasing timestamps
    """
    keep = []
    current_max = float("-inf")

    for i, t in enumerate(df["timestamp"]):
        if t >= current_max:
            # still (or newly) monotonic
            keep.append(i)
            current_max = t
        else:
            # rewind: pop indices until t > previous element
            while keep and df["timestamp"].iloc[keep[-1]] > t:
                keep.pop()
            keep.append(i)
            current_max = t

    return df.iloc[keep].reset_index(drop=True)

def _add_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the DataFrame has an int64 nanosecond ``"timestamp"`` column.

    Downstream ``merge_asof`` calls compare timestamps against a base
    timeline built from ``event.start_time`` (always ns since epoch), so
    every source DataFrame must expose nanoseconds.

    Three cases:
      1. ``"timestamp"`` is a datetime column (common when the message
         payload carries its own ``timestamp`` field — Roboto returns it
         as ``datetime64[us]`` / ``[ms]`` / ``[ns]``). Normalise to
         ``datetime64[ns]`` *before* casting to int64, otherwise the cast
         preserves the original unit and silently produces microseconds or
         milliseconds instead of nanoseconds.
      2. ``"timestamp"`` is already numeric. Assume int64 nanoseconds (we
         can't reliably guess units from the magnitude alone); returned
         unchanged.
      3. No ``"timestamp"`` column. Derive one from the DatetimeIndex that
         ``get_data_as_df`` returns by default.
    """
    if "timestamp" in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            out = df.copy()
            out["timestamp"] = (
                df["timestamp"].astype("datetime64[ns]").astype("int64")
            )
            return out
        return df
    out = df.reset_index()
    idx_col = out.columns[0]
    out["timestamp"] = pd.to_datetime(out[idx_col]).astype("int64")
    out = out.drop(columns=[idx_col])
    return out

def _sanitize_topic_for_key(topic: str) -> str:
    """Sanitize a topic name to be used as part of a key.

    Removes leading slashes and replaces remaining slashes with underscores.
    """
    # Remove leading slashes
    sanitized = topic.lstrip("/")
    # Replace remaining slashes with underscores
    sanitized = sanitized.replace("/", "_")
    return sanitized

def flatten_joint_state(
    df: pd.DataFrame,
    *,
    joint_names: list[str] | None = None,
) -> pd.DataFrame:
    """Flatten JointState messages into scalar columns.

    JointState has parallel arrays `name` and `position`. This function
    creates one scalar column per joint: `position.<joint_name>`.

    Args:
        df: DataFrame with `header.stamp.sec`, `header.stamp.nanosec`,
            `name` (list[str]), and `position` (list[float]).
        joint_names: Optional list of joint names to extract (in order).
            If None, uses all joints from the first message.

    Returns:
        DataFrame with `timestamp` column and one `position.<joint_name>`
        column per joint (float32).
    """
    timestamps: list[int] = []
    joint_data: dict[str, list[float]] = {}

    for _, row in df.iterrows():
        timestamp_ns = ros_time_to_nanoseconds(
            row["header.stamp.sec"], row["header.stamp.nanosec"]
        )
        timestamps.append(timestamp_ns)

        names = row["name"]
        positions = row["position"]

        # Build name->position mapping for this row
        # strict=False: a JointState may carry fewer `position` entries than
        # `name` (the field is optional in the message spec). Truncating to the
        # shorter of the two keeps one malformed message from failing the run.
        name_to_pos = dict(zip(names, positions, strict=False))

        # If joint_names not specified, use all from first message
        if joint_names is None:
            joint_names = list(names)
            # Initialize columns
            for jn in joint_names:
                joint_data[f"position.{jn}"] = []

        # Extract positions in the specified order
        for jn in joint_names:
            if jn not in name_to_pos:
                raise ValueError(
                    f"Joint '{jn}' not found in message. Available: {list(names)}"
                )
            joint_data[f"position.{jn}"].append(np.float32(name_to_pos[jn]))

    result = pd.DataFrame({"timestamp": timestamps, **joint_data})
    logger.info(
        "Flattened JointState: %d joints, %d rows",
        len(joint_names) if joint_names else 0,
        len(result),
    )
    return result


def flatten_nested_array(
    df: pd.DataFrame,
    *,
    array_field: str,
    values_field: str,
    timestamp_field: str,
) -> pd.DataFrame:
    """Flatten a nested repeated field into individual rows with absolute timestamps.

    This is a generic flattener: it does not know about any specific message
    type.  The caller tells it *which* column contains the repeated array,
    *which* sub-field inside each element holds the numeric values, and
    *which* sub-field holds the time offset from the header stamp.

    Args:
        df: DataFrame with ``header.stamp.sec``, ``header.stamp.nanosec``,
            and a column named *array_field* whose cells are ``list[dict]``.
        array_field: Column name of the repeated field (e.g. ``"points"``).
        values_field: Key inside each element dict that contains the numeric
            values (e.g. ``"positions"``).
        timestamp_field: Key inside each element dict that contains a ROS
            duration ``{sec, nanosec}`` offset from the header stamp
            (e.g. ``"time_from_start"``).

    Returns:
        DataFrame with two columns:
        - ``timestamp``: absolute ``int64`` nanosecond timestamp
        - ``values``: ``np.ndarray(N,)`` of ``float32``
    """
    timestamps: list[int] = []
    values_list: list[np.ndarray] = []

    for _, row in df.iterrows():
        base_ns = ros_time_to_nanoseconds(
            row["header.stamp.sec"], row["header.stamp.nanosec"]
        )

        elements = row[array_field]
        if not elements:
            logger.warning(
                "Message at timestamp %d has empty '%s' array.",
                base_ns,
                array_field,
            )
            continue

        for element in elements:
            values = np.array(element[values_field], dtype=np.float32)
            offset = element[timestamp_field]
            offset_ns = ros_time_to_nanoseconds(offset["sec"], offset["nanosec"])

            timestamps.append(base_ns + offset_ns)
            values_list.append(values)

    result = pd.DataFrame({"timestamp": timestamps, "values": values_list})
    result = keep_monotonic_timestamps(result)
    logger.info(
        "Flattened '%s.%s': %d rows from %d messages",
        array_field,
        values_field,
        len(result),
        len(df),
    )
    return result



def get_image_dimensions(
    compressed_data: bytes, format_str: str
) -> tuple[int, int, int]:
    """
    Get image dimensions without full decompression.

    Args:
        compressed_data: Compressed image bytes from ROS CompressedImage message
        format_str: Format string from CompressedImage message (e.g., "jpeg", "png")

    Returns:
        Tuple of (height, width, channels) in pixels
    """
    img_array = np.frombuffer(compressed_data, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"Failed to decode image with format: {format_str}")

    return img.shape


def decompress_image(compressed_data: bytes, format_str: str) -> np.ndarray:
    """
    Decompress CompressedImage data using cv2.imdecode and convert BGR to RGB.

    Args:
        compressed_data: Compressed image bytes from ROS CompressedImage message
            or raw image bytes from AVI video extraction
        format_str: Format string from CompressedImage message (e.g., "jpeg", "png", "avi")

    Returns:
        numpy array (H, W, 3) uint8 in RGB format
    """
    img_array = np.frombuffer(compressed_data, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"Failed to decode image with format: {format_str}")

    # Convert BGR to RGB (cv2 loads as BGR, but AVI frames may already be in correct format)
    # Check if format is avi - the image data from Roboto's get_data() for AVI files
    # is typically already processed and may be in different formats
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img_rgb


# =============================================================================
# Raw sensor_msgs/msg/Image decoding helpers
# =============================================================================

# ROS encoding → (numpy dtype, channel count) for reshape of raw Image bytes.
_ROS_IMAGE_DTYPE: dict[str, tuple[type, int]] = {
    "mono8":  (np.uint8,   1),
    "mono16": (np.uint16,  1),
    "16UC1":  (np.uint16,  1),
    "8UC1":   (np.uint8,   1),
    "32FC1":  (np.float32, 1),
    "rgb8":   (np.uint8,   3),
    "bgr8":   (np.uint8,   3),
    "rgba8":  (np.uint8,   4),
    "bgra8":  (np.uint8,   4),
}

# Encodings that carry depth (scalar range) rather than color.
DEPTH_ENCODINGS = frozenset({"mono16", "16UC1", "32FC1"})

_COLORMAP_LUT: dict[str, int] = {
    "viridis": cv2.COLORMAP_VIRIDIS,
    "turbo":   cv2.COLORMAP_TURBO,
    "jet":     cv2.COLORMAP_JET,
    "inferno": cv2.COLORMAP_INFERNO,
    "plasma":  cv2.COLORMAP_PLASMA,
    "magma":   cv2.COLORMAP_MAGMA,
}


_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC = b"\xff\xd8\xff"


def _maybe_decode_compressed_image_bytes(
    data: bytes, encoding: str
) -> np.ndarray | None:
    """Return a decoded ndarray if ``data`` is PNG/JPEG-compressed, else ``None``.

    Roboto ingests some sensor_msgs/msg/Image topics by PNG-encoding the pixel
    buffer for storage (lossless for uint16 depth). When that happens the raw
    byte count no longer matches H*W*channels, so we detect the container via
    magic bytes and decode with cv2 preserving the native bit depth.
    """
    if len(data) >= 8 and data[:8] == _PNG_MAGIC:
        pass
    elif len(data) >= 3 and data[:3] == _JPEG_MAGIC:
        pass
    else:
        return None
    img_array = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(
            f"Detected compressed image container for encoding '{encoding}' "
            "but cv2.imdecode failed."
        )
    return img


def reshape_raw_image(
    data: bytes, height: int, width: int, encoding: str
) -> np.ndarray:
    """Interpret raw ROS sensor_msgs/Image bytes as a native-dtype ndarray.

    Returns a 2D array (H, W) for single-channel encodings and a 3D array
    (H, W, C) for multi-channel ones. Native dtype (uint8/uint16/float32) is
    preserved — callers are responsible for any range/channel conversion.

    Raises ValueError for unsupported encodings.
    """
    if encoding not in _ROS_IMAGE_DTYPE:
        raise ValueError(
            f"Unsupported raw Image encoding: '{encoding}'. "
            f"Supported: {sorted(_ROS_IMAGE_DTYPE)}"
        )
    dtype, channels = _ROS_IMAGE_DTYPE[encoding]

    decoded = _maybe_decode_compressed_image_bytes(data, encoding)
    if decoded is not None:
        if decoded.dtype != np.dtype(dtype):
            decoded = decoded.astype(dtype, copy=False)
        return decoded

    arr = np.frombuffer(data, dtype=dtype)
    expected = height * width * channels
    if arr.size != expected:
        raise ValueError(
            f"Raw Image byte count mismatch for encoding '{encoding}': "
            f"expected {expected} elements ({height}x{width}x{channels}), got {arr.size}"
        )
    return arr.reshape((height, width, channels)) if channels > 1 else arr.reshape((height, width))


def _resolve_depth_scale(encoding: str, units: Any) -> float:
    """Resolve the scale factor that converts raw depth values into meters.

    ``units`` may be:
      - ``"m"`` → 1.0
      - ``"mm"`` → 0.001
      - a numeric (int/float) value → used as-is (raw * scale = meters)
      - ``None`` → auto-pick per encoding (uint16 depth → mm; float32 → m)
    """
    if units is None:
        return 0.001 if encoding in ("mono16", "16UC1") else 1.0
    if isinstance(units, (int, float)) and not isinstance(units, bool):
        return float(units)
    if units == "m":
        return 1.0
    if units == "mm":
        return 0.001
    raise ValueError(
        f"Unrecognised depth.units value: {units!r}. Expected 'm', 'mm', or a numeric scale factor."
    )


def _apply_invalid_policy(
    depth_m: np.ndarray, invalid_mask: np.ndarray, policy: Any, near_m: float, far_m: float
) -> np.ndarray:
    """Replace pixels flagged by ``invalid_mask`` per ``policy``."""
    if not invalid_mask.any():
        return depth_m
    if policy == "near" or policy is None:
        fill = near_m
    elif policy == "far":
        fill = far_m
    elif policy == "zero":
        fill = 0.0
    elif policy == "nan":
        fill = np.nan
    elif isinstance(policy, (int, float)) and not isinstance(policy, bool):
        fill = float(policy)
    else:
        raise ValueError(
            f"Unrecognised depth.invalid policy: {policy!r}. "
            "Expected 'near', 'far', 'zero', 'nan', or a numeric value."
        )
    depth_m = depth_m.copy()
    depth_m[invalid_mask] = fill
    return depth_m


def depth_to_uint8_rgb(
    depth_raw: np.ndarray, encoding: str, depth_cfg: dict
) -> np.ndarray:
    """Convert a 2D depth ndarray (uint16 or float32) into an HWC uint8 RGB
    image suitable for h264 encoding.

    Pipeline:
      1. Scale raw values to meters (``depth_cfg["units"]``, or auto per encoding).
      2. Replace invalid pixels (0 for uint16; NaN/inf for float32) per
         ``depth_cfg["invalid"]`` (default: ``"far"``).
      3. Clip to ``depth_cfg["range"] = [near_m, far_m]`` if
         ``depth_cfg["clip"]`` (default ``True``).
      4. Normalise to uint8 over the range.
      5. Apply ``depth_cfg["colormap"]`` (default ``"grayscale"``). Non-grayscale
         choices use OpenCV colormaps.

    ``depth_cfg["range"]`` is REQUIRED — this function cannot pick a sensible
    default without knowing the task scene geometry.
    """
    if "range" not in depth_cfg:
        raise ValueError(
            "Depth encoding requires 'image.depth.range: [near_m, far_m]' in the contract."
        )
    near_m, far_m = float(depth_cfg["range"][0]), float(depth_cfg["range"][1])
    if not (far_m > near_m):
        raise ValueError(f"depth.range must satisfy far > near, got [{near_m}, {far_m}]")

    scale = _resolve_depth_scale(encoding, depth_cfg.get("units"))
    depth_m = depth_raw.astype(np.float32) * scale

    if np.issubdtype(depth_raw.dtype, np.integer):
        invalid_mask = depth_raw == 0
    else:
        invalid_mask = ~np.isfinite(depth_m)

    depth_m = _apply_invalid_policy(
        depth_m, invalid_mask, depth_cfg.get("invalid", "far"), near_m, far_m
    )

    if depth_cfg.get("clip", True):
        depth_m = np.clip(depth_m, near_m, far_m)

    norm = ((depth_m - near_m) / (far_m - near_m) * 255.0)
    mono8 = np.clip(norm, 0, 255).astype(np.uint8)

    colormap = depth_cfg.get("colormap", "grayscale")
    if colormap == "grayscale":
        return np.stack([mono8, mono8, mono8], axis=-1)
    if colormap not in _COLORMAP_LUT:
        raise ValueError(
            f"Unrecognised colormap '{colormap}'. "
            f"Supported: grayscale, {', '.join(sorted(_COLORMAP_LUT))}."
        )
    bgr = cv2.applyColorMap(mono8, _COLORMAP_LUT[colormap])
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
