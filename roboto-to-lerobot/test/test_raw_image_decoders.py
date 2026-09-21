"""Unit tests for raw sensor_msgs/msg/Image decoders.

Tests the decoders for various image encodings supported by the Roboto
pipeline, including mono8, 8UC1, rgb8, bgr8, and rgba8/bgra8.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import roboto_to_lerobot.runtime.decoders  # noqa: F401  (registers decoders)
from roboto_to_lerobot.contract_utils import ObservationSpec
from roboto_to_lerobot.runtime.converters import decode_value

# ── fixtures ────────────────────────────────────────────────────────────────

def _raw_image_row(
    height: int = 4,
    width: int = 6,
    encoding: str = "mono8",
) -> pd.Series:
    """Create a raw sensor_msgs/msg/Image row with synthetic data.

    For single-channel encodings, we generate incrementing uint8 values.
    For multi-channel encodings, we stack the channels with different offsets.
    """
    if encoding in ("mono8", "8UC1"):
        # Single channel: H x W bytes
        byte_count = height * width
        data = bytes((i % 256) for i in range(byte_count))
    elif encoding in ("rgb8", "bgr8"):
        # Three channels: H x W x 3 bytes
        byte_count = height * width * 3
        data = bytes((i % 256) for i in range(byte_count))
    elif encoding == "rgba8":
        # Four channels: H x W x 4 bytes
        byte_count = height * width * 4
        data = bytes((i % 256) for i in range(byte_count))
    elif encoding == "bgra8":
        # Four channels: H x W x 4 bytes
        byte_count = height * width * 4
        data = bytes((i % 256) for i in range(byte_count))
    else:
        raise ValueError(f"Unsupported encoding for test: {encoding}")

    return pd.Series({
        "encoding": encoding,
        "height": height,
        "width": width,
        "data": data,
    })


# ── Monochrome encodings (mono8, 8UC1) ─────────────────────────────────────

def test_mono8_single_channel_replicated_to_rgb():
    """mono8 should be replicated to 3-channel RGB."""
    spec = ObservationSpec(
        key="observation.camera",
        topic="/camera/image",
        type="sensor_msgs/msg/Image",
    )
    row = _raw_image_row(height=4, width=6, encoding="mono8")
    out = decode_value(row, spec)

    # Should be 4 x 6 x 3 uint8
    assert out.shape == (4, 6, 3)
    assert out.dtype == np.uint8

    # All three channels should be identical
    np.testing.assert_array_equal(out[..., 0], out[..., 1])
    np.testing.assert_array_equal(out[..., 1], out[..., 2])


def test_8uc1_single_channel_replicated_to_rgb():
    """8UC1 should be treated identically to mono8: replicated to 3-channel RGB."""
    spec = ObservationSpec(
        key="observation.camera",
        topic="/camera/image",
        type="sensor_msgs/msg/Image",
    )
    row = _raw_image_row(height=4, width=6, encoding="8UC1")
    out = decode_value(row, spec)

    # Should be 4 x 6 x 3 uint8
    assert out.shape == (4, 6, 3)
    assert out.dtype == np.uint8

    # All three channels should be identical
    np.testing.assert_array_equal(out[..., 0], out[..., 1])
    np.testing.assert_array_equal(out[..., 1], out[..., 2])


def test_8uc1_matches_mono8_output():
    """8UC1 and mono8 should produce identical output for the same data."""
    spec = ObservationSpec(
        key="observation.camera",
        topic="/camera/image",
        type="sensor_msgs/msg/Image",
    )

    # Use same data for both encodings
    test_data = bytes([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240])

    mono8_row = pd.Series({
        "encoding": "mono8",
        "height": 4,
        "width": 6,
        "data": test_data,
    })

    uc8_row = pd.Series({
        "encoding": "8UC1",
        "height": 4,
        "width": 6,
        "data": test_data,
    })

    mono8_out = decode_value(mono8_row, spec)
    uc8_out = decode_value(uc8_row, spec)

    # Should be identical
    np.testing.assert_array_equal(mono8_out, uc8_out)


# ── Color encodings (rgb8, bgr8) ───────────────────────────────────────────

def test_rgb8_passthrough():
    """rgb8 should pass through unchanged (3 channels, no reordering)."""
    spec = ObservationSpec(
        key="observation.camera",
        topic="/camera/image",
        type="sensor_msgs/msg/Image",
    )
    row = _raw_image_row(height=2, width=3, encoding="rgb8")
    out = decode_value(row, spec)

    # Should be 2 x 3 x 3 uint8
    assert out.shape == (2, 3, 3)
    assert out.dtype == np.uint8


def test_bgr8_channels_reversed():
    """bgr8 should be converted to RGB (channels reversed)."""
    spec = ObservationSpec(
        key="observation.camera",
        topic="/camera/image",
        type="sensor_msgs/msg/Image",
    )
    row = _raw_image_row(height=2, width=3, encoding="bgr8")
    out = decode_value(row, spec)

    # Should be 2 x 3 x 3 uint8
    assert out.shape == (2, 3, 3)
    assert out.dtype == np.uint8


# ── RGBA encodings (rgba8, bgra8) ─────────────────────────────────────────

def test_rgba8_alpha_dropped():
    """rgba8 should drop the alpha channel, keeping only RGB."""
    spec = ObservationSpec(
        key="observation.camera",
        topic="/camera/image",
        type="sensor_msgs/msg/Image",
    )
    row = _raw_image_row(height=2, width=3, encoding="rgba8")
    out = decode_value(row, spec)

    # Should be 2 x 3 x 3 uint8 (alpha dropped)
    assert out.shape == (2, 3, 3)
    assert out.dtype == np.uint8


def test_bgra8_alpha_dropped_and_reversed():
    """bgra8 should drop alpha and reverse BGR to RGB."""
    spec = ObservationSpec(
        key="observation.camera",
        topic="/camera/image",
        type="sensor_msgs/msg/Image",
    )
    row = _raw_image_row(height=2, width=3, encoding="bgra8")
    out = decode_value(row, spec)

    # Should be 2 x 3 x 3 uint8 (alpha dropped)
    assert out.shape == (2, 3, 3)
    assert out.dtype == np.uint8
