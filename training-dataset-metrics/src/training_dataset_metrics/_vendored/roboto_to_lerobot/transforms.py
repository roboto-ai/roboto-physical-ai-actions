# roboto_to_lerobot/transforms.py
# -----------------------------------------------------------------------------
# Registry of transforms applied to observation/action timeseries.
# Each transform operates on the full episode stacked as an (T, N)
# ndarray.  Transforms may optionally receive and return timestamps.
# -----------------------------------------------------------------------------

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

from .contract_utils import TransformSpec

logger = logging.getLogger(__name__)

# Transform function signature:
#   (data, params, fps, timestamps?) -> data | (data, timestamps)
#
#   data:       np.ndarray of shape (T, N)
#   params:     free-form kwargs from the contract YAML
#   fps:        sampling frequency of the signal at the time this transform is
#               called (Hz).  For pre-alignment transforms this is the raw
#               capture rate; for post-alignment transforms it is contract.fps.
#   timestamps: optional np.ndarray of shape (T,) — provided for pre-alignment
#
# A transform may return just the data array, or a tuple (data, timestamps)
# when it changes the time axis (e.g. resampling).
TransformFn = Callable[..., np.ndarray | tuple[np.ndarray, np.ndarray]]

TRANSFORMS: dict[str, TransformFn] = {}


def register_transform(name: str):
    """Decorator to register a transform function by name."""

    def _wrap(fn: TransformFn) -> TransformFn:
        TRANSFORMS[name] = fn
        return fn

    return _wrap


def apply_transforms(
    data: np.ndarray,
    specs: list[TransformSpec],
    fps: float,
    timestamps: np.ndarray | None = None,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Apply a chain of transforms sequentially.

    When *timestamps* is provided, each transform receives it as a fourth
    argument.  If a transform returns a ``(data, timestamps)`` tuple the
    updated timestamps are threaded through to the next transform, and *fps*
    is recomputed from the new timestamps so that subsequent transforms in the
    chain always receive the correct sampling frequency.

    The fps recomputation is unit-agnostic: a scale factor
    ``ts_scale = fps * median(diff(timestamps))`` is derived from the initial
    fps + timestamps pair, then used as ``fps = ts_scale / median(diff(new_ts))``
    after any transform that modifies the time axis.

    Returns:
        If *timestamps* was ``None``: the transformed data array.
        If *timestamps* was provided: ``(data, timestamps)`` tuple (timestamps
        may have been modified by one of the transforms).
    """
    # Derive scale factor so fps can be recomputed after timestamp-modifying
    # transforms, regardless of the timestamp unit (ns, s, …).
    ts_scale: float | None = None
    if timestamps is not None and len(timestamps) > 1:
        ts_scale = fps * float(np.median(np.diff(timestamps)))

    for spec in specs:
        fn = TRANSFORMS.get(spec.type)
        if fn is None:
            raise ValueError(
                f"Unknown transform: '{spec.type}'. "
                f"Available: {sorted(TRANSFORMS)}"
            )
        if timestamps is not None:
            result = fn(data, spec.params, fps, timestamps=timestamps)
        else:
            result = fn(data, spec.params, fps)

        if isinstance(result, tuple):
            data, timestamps = result
            # Recompute fps from the updated timestamps so the next transform
            # in the chain receives the correct sampling frequency.
            if ts_scale is not None and len(timestamps) > 1:
                fps = ts_scale / float(np.median(np.diff(timestamps)))
        else:
            data = result

    if timestamps is not None:
        return data, timestamps
    return data


# ---------------------------------------------------------------------------
# Built-in transforms
# ---------------------------------------------------------------------------


@register_transform("finite_difference")
def _finite_difference(
    data: np.ndarray, params: dict, fps: float, **kwargs: object
) -> np.ndarray:
    """Frame-to-frame finite difference.
    out[i] = in[i+1] - in[i]

    Last row is set to repeat the second-to-last value so that the output
    shape matches the input.
    """
    out = np.empty_like(data, dtype=np.float64)
    out[:-1] = np.diff(data, axis=0)
    out[-1] = out[-2]
    return out


def _validate_cutoff_hz(transform_name: str, cutoff_hz: float, fps: float) -> None:
    """Raise a clear ``ValueError`` if *cutoff_hz* is not a valid low-pass
    cutoff for a signal sampled at *fps*.

    scipy's own check (``0 < Wn < fs/2``, enforced inside ``butter``) raises
    an opaque ``ValueError: Digital filter critical frequencies must be
    0 < Wn < fs/2 (fs=...)`` that doesn't say which transform or param is at
    fault, nor which *fps* it means. This matters because ``fps`` is *not*
    always the contract's declared output rate: for ``stage: pre`` transforms
    it's the measured raw topic rate (see module docstring above), so a
    cutoff_hz that looks safe against the contract fps can still violate
    Nyquist against the raw capture rate.
    """
    nyquist = fps / 2.0
    if not (0 < cutoff_hz < nyquist):
        raise ValueError(
            f"{transform_name} cutoff_hz={cutoff_hz} must be strictly below "
            f"the Nyquist frequency fs/2={nyquist:.3f} (fs={fps:.3f} Hz). "
            f"For stage:pre transforms fs is the measured raw topic rate, "
            f"not the contract's output fps."
        )


@register_transform("butterworth_lowpass")
def _butterworth_lowpass(
    data: np.ndarray, params: dict, fps: float, **kwargs: object
) -> np.ndarray:
    """Zero-phase Butterworth low-pass filter.

    Uses ``scipy.signal.sosfiltfilt`` (forward-backward filtering), which
    has zero phase distortion but is *non-causal*: each output sample
    depends on both past and future samples in the episode. That makes it
    unreproducible online — a live policy only ever has past samples
    available, so a filter applied this way during training cannot be
    replicated bit-for-bit at deployment time. Prefer this for stream
    preprocessing where only offline reproducibility matters (e.g.
    smoothing an ``observation.state`` stream that isn't also filtered by
    the deployed policy's own inputs). For any stream where training and
    inference must apply the identical filter, use
    ``butterworth_lowpass_causal`` instead.

    Required params:
        cutoff_hz:  filter cutoff frequency (Hz)
    Optional params:
        order:      filter order (default 2)
    """
    from scipy.signal import butter, sosfiltfilt

    cutoff_hz = params["cutoff_hz"]
    order = params.get("order", 2)
    _validate_cutoff_hz("butterworth_lowpass", cutoff_hz, fps)
    sos = butter(order, cutoff_hz, btype="low", output="sos", fs=fps)
    return sosfiltfilt(sos, data, axis=0)


@register_transform("butterworth_lowpass_causal")
def _butterworth_lowpass_causal(
    data: np.ndarray, params: dict, fps: float, **kwargs: object
) -> np.ndarray:
    """Causal (forward-only) Butterworth low-pass filter.

    Uses ``scipy.signal.sosfilt`` — a strictly forward filter, so each
    output sample depends only on current and *past* input samples. That
    makes it causal and deployable online: the exact same filter (same
    coefficients, same recursive form) can run sample-by-sample at
    inference time and reproduce what training saw, eliminating the
    train/deploy skew that ``butterworth_lowpass``'s zero-phase
    ``sosfiltfilt`` cannot avoid. The tradeoff is phase lag / group delay —
    unlike the zero-phase variant, filtered features are shifted in time
    relative to the input, roughly proportional to filter order and
    inversely proportional to cutoff_hz. Prefer this variant for any
    action/state stream where the same filter must be reproduced online at
    deployment; prefer ``butterworth_lowpass`` when only offline
    reproducibility matters and zero phase distortion is desired.

    The filter's initial state is set via ``sosfilt_zi`` scaled by the
    first sample of each channel, so a constant (or slowly varying) signal
    passes through without the startup step transient a zero-initialized
    filter would otherwise introduce — while remaining strictly causal.

    Required params:
        cutoff_hz:  filter cutoff frequency (Hz)
    Optional params:
        order:      filter order (default 2)
        fs_hz:      design sample rate override (Hz). This is the crux of
                    *live-runtime coefficient parity*: for ``stage: pre``
                    transforms, ``fps`` here is the raw topic rate this
                    particular training run happened to measure (see
                    ``lerobot.py``'s ``raw_fs``) — a number the live
                    runtime, which never sees the offline episode, cannot
                    recompute on its own. Declaring ``fs_hz`` freezes the
                    design rate into the contract so training and
                    deployment build the *identical* ``sos`` coefficients
                    (``runtime.causal_lowpass.CausalLowpass`` reads the
                    same ``fs_hz`` param). When set, ``fs_hz`` replaces
                    ``fps`` both for the filter design and for the Nyquist
                    validation below — a cutoff safe against the contract
                    fps can still violate Nyquist against a declared
                    fs_hz, and vice versa. When ``fs_hz`` deviates from the
                    measured ``fps`` passed in by more than 10%, a
                    stream's *actual* rate has drifted from what the
                    contract author declared — filter coefficients
                    designed for one rate but fed samples at a materially
                    different rate no longer approximate the intended
                    cutoff. That's a real parity risk, but a hard error
                    would be too aggressive for streams with legitimately
                    irregular timing, so this only logs a warning naming
                    both numbers.
    """
    from scipy.signal import butter, sosfilt, sosfilt_zi

    cutoff_hz = params["cutoff_hz"]
    order = params.get("order", 2)
    fs_hz = params.get("fs_hz")
    design_fps = fps
    if fs_hz is not None:
        design_fps = float(fs_hz)
        if fps > 0:
            deviation = abs(design_fps - fps) / fps
            if deviation > 0.10:
                logger.warning(
                    "butterworth_lowpass_causal: declared fs_hz=%.4f deviates "
                    "from the measured fps=%.4f by %.1f%% (>10%%). The live "
                    "runtime designs this filter with fs_hz, so a wrongly "
                    "declared rate is a silent train/deploy parity break — "
                    "verify fs_hz matches this stream's true sample rate.",
                    design_fps, fps, deviation * 100.0,
                )
    _validate_cutoff_hz("butterworth_lowpass_causal", cutoff_hz, design_fps)
    sos = butter(order, cutoff_hz, btype="low", output="sos", fs=design_fps)
    # sosfilt_zi(sos) has shape (n_sections, 2); broadcast against each
    # channel's first sample so every channel's filter state starts at
    # steady-state for that channel's own initial value, not zero.
    zi = sosfilt_zi(sos)[:, :, None] * data[0]
    out, _ = sosfilt(sos, data, axis=0, zi=zi)
    return out


@register_transform("resample_uniform")
def _resample_uniform(
    data: np.ndarray, params: dict, fps: float, timestamps: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Resample an irregularly sampled signal onto a uniform grid using
    nearest-neighbour selection.

    The uniform grid spans ``[timestamps[0], timestamps[-1]]`` with a step
    size equal to the **median** inter-sample interval of the original
    signal.  Each output sample is taken from the nearest original sample.

    This is a *pre-alignment* transform (``stage: pre``).  It returns
    ``(resampled_data, uniform_timestamps)``.
    """
    if len(timestamps) < 2:
        return data, timestamps

    dt = np.median(np.diff(timestamps))
    n_samples = int(np.round((timestamps[-1] - timestamps[0]) / dt)) + 1
    uniform_ts = np.linspace(
        timestamps[0], timestamps[-1], n_samples,
    ).astype(timestamps.dtype)

    # Nearest-neighbour: find the index in the original timestamps closest
    # to each uniform timestamp.
    indices = np.searchsorted(timestamps, uniform_ts, side="left")
    # searchsorted gives the insertion point; compare with neighbours to
    # pick the truly nearest sample.
    indices = np.clip(indices, 0, len(timestamps) - 1)
    left = np.clip(indices - 1, 0, len(timestamps) - 1)
    use_left = np.abs(timestamps[left] - uniform_ts) < np.abs(timestamps[indices] - uniform_ts)
    indices[use_left] = left[use_left]

    return data[indices], uniform_ts