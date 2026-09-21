"""Unit tests for ``roboto_to_lerobot.transforms``.

Focus is the Butterworth low-pass pair: the existing zero-phase
``butterworth_lowpass`` (``sosfiltfilt``, non-causal) and the new
``butterworth_lowpass_causal`` (``sosfilt``, forward-only). The causal
variant exists so a filter applied during training can be reproduced
sample-by-sample at online inference, where only past samples are
available — that's the whole point of the test coverage below: attenuation
still works, but the output lags the zero-phase version, a constant signal
doesn't get a startup transient, and an out-of-range cutoff_hz fails with a
message that names the offending fs instead of scipy's opaque one.
"""

from __future__ import annotations

import numpy as np
import pytest
from roboto_to_lerobot.transforms import TRANSFORMS


def _sine(freq_hz: float, fps: float, n: int, *, amplitude: float = 1.0) -> np.ndarray:
    t = np.arange(n) / fps
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float64)


def test_causal_lowpass_attenuates_high_frequency_and_preserves_shape():
    fps = 100.0
    n = 500
    # Well above cutoff (5 Hz): a 4th-order Butterworth should crush this to
    # near-zero amplitude once the filter settles.
    high = _sine(30.0, fps, n, amplitude=1.0)
    data = np.stack([high, high], axis=1)  # (T, 2)

    fn = TRANSFORMS["butterworth_lowpass_causal"]
    out = fn(data, {"cutoff_hz": 5.0, "order": 4}, fps)

    assert out.shape == data.shape

    # Settled tail (skip the filter's transient): amplitude should be
    # substantially attenuated relative to the raw unit-amplitude input.
    tail_rms = float(np.sqrt(np.mean(out[200:] ** 2)))
    assert tail_rms < 0.3


def test_causal_lowpass_lags_zero_phase_variant_on_shifted_step():
    fps = 100.0
    n = 400
    step = np.zeros(n, dtype=np.float64)
    step[100:] = 1.0
    data = step[:, None]  # (T, 1)

    params = {"cutoff_hz": 5.0, "order": 2}
    causal = TRANSFORMS["butterworth_lowpass_causal"](data, params, fps)
    zero_phase = TRANSFORMS["butterworth_lowpass"](data, params, fps)

    # Zero-phase filtering is symmetric around the step and crosses the
    # midpoint (0.5) essentially at the step location. The causal filter,
    # having only seen past samples, must cross later — that lag is the
    # signature of non-zero group delay.
    def crossing_index(series: np.ndarray) -> int:
        above = np.where(series[:, 0] >= 0.5)[0]
        assert len(above) > 0
        return int(above[0])

    causal_crossing = crossing_index(causal)
    zero_phase_crossing = crossing_index(zero_phase)

    assert causal_crossing > zero_phase_crossing
    assert causal_crossing - zero_phase_crossing >= 2


def test_causal_lowpass_constant_input_passes_through_unchanged():
    fps = 50.0
    n = 200
    data = np.full((n, 3), 7.5, dtype=np.float64)

    fn = TRANSFORMS["butterworth_lowpass_causal"]
    out = fn(data, {"cutoff_hz": 5.0, "order": 2}, fps)

    assert out.shape == data.shape
    # zi initialization should suppress the startup step transient, so even
    # the very first output samples stay close to the constant input.
    np.testing.assert_allclose(out, 7.5, atol=1e-9)


@pytest.mark.parametrize(
    "transform_name",
    ["butterworth_lowpass", "butterworth_lowpass_causal"],
)
def test_lowpass_rejects_cutoff_at_or_above_nyquist(transform_name):
    fps = 29.817
    data = np.zeros((50, 1), dtype=np.float64)
    fn = TRANSFORMS[transform_name]

    with pytest.raises(ValueError, match="must be strictly below the Nyquist"):
        fn(data, {"cutoff_hz": 15.0}, fps)


@pytest.mark.parametrize(
    "transform_name",
    ["butterworth_lowpass", "butterworth_lowpass_causal"],
)
def test_lowpass_rejects_non_positive_cutoff(transform_name):
    fps = 30.0
    data = np.zeros((50, 1), dtype=np.float64)
    fn = TRANSFORMS[transform_name]

    with pytest.raises(ValueError, match="must be strictly below the Nyquist"):
        fn(data, {"cutoff_hz": 0.0}, fps)


# ----------------------------------------------------------------------------
# fs_hz — the live-runtime coefficient-parity knob (causal variant only)
# ----------------------------------------------------------------------------


def test_fs_hz_overrides_design_rate():
    """Declaring fs_hz designs the filter at that rate instead of the
    passed fps — the whole point being that a live adapter re-designing
    the same filter with a declared fs_hz (rather than whatever fps it
    happens to observe) reproduces this exact output."""
    n = 500
    high = _sine(30.0, 100.0, n, amplitude=1.0)  # sampled "at" 100 Hz nominally
    data = np.stack([high, high], axis=1)

    fn = TRANSFORMS["butterworth_lowpass_causal"]
    params = {"cutoff_hz": 5.0, "order": 4}

    # Called with fps=100 (no fs_hz) vs fps=40 but fs_hz=100 pinned in
    # params: the two should be identical because fs_hz wins.
    out_no_fs_hz = fn(data, params, fps=100.0)
    out_with_fs_hz = fn(data, {**params, "fs_hz": 100.0}, fps=40.0)
    np.testing.assert_allclose(out_no_fs_hz, out_with_fs_hz, atol=1e-12)

    # And it must differ from a filter genuinely designed at the deviating
    # fps=40 rate (no fs_hz override) — proving fs_hz actually changed the
    # design rate rather than being silently ignored.
    out_design_at_40 = fn(data, params, fps=40.0)
    assert not np.allclose(out_with_fs_hz, out_design_at_40)


def test_fs_hz_is_validated_against_nyquist_not_fps():
    """A cutoff_hz that's safe against fps but violates Nyquist for the
    declared fs_hz must still raise — fs_hz governs validation once set."""
    fn = TRANSFORMS["butterworth_lowpass_causal"]
    data = np.zeros((50, 1), dtype=np.float64)

    # cutoff_hz=8 is safe at fps=30 (Nyquist=15) but violates Nyquist at
    # fs_hz=12 (Nyquist=6).
    with pytest.raises(ValueError, match="must be strictly below the Nyquist"):
        fn(data, {"cutoff_hz": 8.0, "fs_hz": 12.0}, fps=30.0)


def test_fs_hz_large_deviation_from_measured_fps_logs_warning(caplog):
    fn = TRANSFORMS["butterworth_lowpass_causal"]
    n = 200
    data = np.zeros((n, 1), dtype=np.float64)

    with caplog.at_level("WARNING"):
        fn(data, {"cutoff_hz": 5.0, "fs_hz": 100.0}, fps=50.0)  # 100% deviation

    assert any(
        "fs_hz" in record.message and "50" in record.message and "100" in record.message
        for record in caplog.records
    ), f"expected a deviation warning naming both rates; got {[r.message for r in caplog.records]}"


def test_fs_hz_small_deviation_from_measured_fps_does_not_warn(caplog):
    fn = TRANSFORMS["butterworth_lowpass_causal"]
    n = 200
    data = np.zeros((n, 1), dtype=np.float64)

    with caplog.at_level("WARNING"):
        # ~2% deviation, well under the 10% threshold.
        fn(data, {"cutoff_hz": 5.0, "fs_hz": 100.0}, fps=98.0)

    assert not any("deviates" in record.message for record in caplog.records)
