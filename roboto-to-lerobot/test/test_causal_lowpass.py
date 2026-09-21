"""Unit tests for :class:`roboto_to_lerobot.runtime.causal_lowpass.CausalLowpass`.

The one property that matters here is *parity*: feeding samples through
:meth:`CausalLowpass.push` one at a time, carrying ``zi`` across calls,
must reproduce exactly what the converter's batch
``transforms._butterworth_lowpass_causal`` computes for the same
``(cutoff_hz, order, fs)`` over the same sequence. That equivalence is the
whole reason a causal filter designed offline can be reproduced live —
see the module docstring in ``causal_lowpass.py`` for the argument.
"""

from __future__ import annotations

import numpy as np
import pytest
from roboto_to_lerobot.runtime.causal_lowpass import CausalLowpass
from roboto_to_lerobot.transforms import TRANSFORMS


def _sine(freq_hz: float, fps: float, n: int, *, amplitude: float = 1.0) -> np.ndarray:
    t = np.arange(n) / fps
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float64)


def test_push_matches_converter_batch_output_single_channel():
    fps = 100.0
    n = 300
    cutoff_hz = 5.0
    order = 4

    data = _sine(30.0, fps, n)[:, None]  # (T, 1)

    batch_out = TRANSFORMS["butterworth_lowpass_causal"](
        data, {"cutoff_hz": cutoff_hz, "order": order}, fps,
    )

    online = CausalLowpass(cutoff_hz=cutoff_hz, order=order, fs=fps)
    online_out = np.stack([online.push(data[i]) for i in range(n)])

    np.testing.assert_allclose(online_out, batch_out, atol=1e-10)


def test_push_matches_converter_batch_output_multi_channel():
    """Multiple channels filtered independently must match too — each
    channel's zi is seeded from its own first sample."""
    fps = 50.0
    n = 400
    cutoff_hz = 3.0
    order = 2

    ch_a = _sine(2.0, fps, n, amplitude=1.0)
    ch_b = _sine(8.0, fps, n, amplitude=0.5) + 2.0  # non-zero-mean channel
    data = np.stack([ch_a, ch_b], axis=1)  # (T, 2)

    batch_out = TRANSFORMS["butterworth_lowpass_causal"](
        data, {"cutoff_hz": cutoff_hz, "order": order}, fps,
    )

    online = CausalLowpass(cutoff_hz=cutoff_hz, order=order, fs=fps)
    online_out = np.stack([online.push(data[i]) for i in range(n)])

    np.testing.assert_allclose(online_out, batch_out, atol=1e-10)


def test_push_matches_converter_batch_output_with_designed_fs_hz():
    """The parity invariant must hold even when the design rate (fs) is not
    the sequence's own nominal sampling rate — mirrors a pre-stage transform
    where fs_hz is declared explicitly rather than inferred."""
    design_fs = 90.0
    n = 250
    cutoff_hz = 4.0
    order = 3

    data = _sine(15.0, design_fs, n)[:, None]

    batch_out = TRANSFORMS["butterworth_lowpass_causal"](
        data, {"cutoff_hz": cutoff_hz, "order": order, "fs_hz": design_fs}, fps=30.0,
    )

    online = CausalLowpass(cutoff_hz=cutoff_hz, order=order, fs=design_fs)
    online_out = np.stack([online.push(data[i]) for i in range(n)])

    np.testing.assert_allclose(online_out, batch_out, atol=1e-10)


def test_push_returns_shape_matching_input_channel_count():
    online = CausalLowpass(cutoff_hz=5.0, order=2, fs=100.0)
    out = online.push(np.array([1.0, 2.0, 3.0]))
    assert out.shape == (3,)


def test_constructor_validates_cutoff_at_init_not_first_push():
    """Nyquist violations must fail at construction (adapter boot time),
    not on the first push — matches CausalLowpass's boot-time-failure
    contract, mirroring the converter's transform validation."""
    with pytest.raises(ValueError, match="must be strictly below the Nyquist"):
        CausalLowpass(cutoff_hz=20.0, order=2, fs=30.0)
