# roboto_to_lerobot/transforms.py
# -----------------------------------------------------------------------------
# Registry of transforms applied to observation/action timeseries.
# Each transform operates on a stream stacked as a (T, N) ndarray: the whole
# episode at stage pre, or one unbroken run of the frames the stream has a
# sample for at stage post. N is every column of the stream, or the
# ``columns`` its spec lists.
# Transforms may optionally receive and return timestamps.
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

    A spec that lists ``columns`` gives its transform only those columns and
    writes the result back in their place; the other columns pass through
    unchanged. Such a transform must return the shape it was given, so one
    that changes the time axis or the column count raises ``ValueError``.

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
        target = data if spec.columns is None else data[:, spec.columns]
        if timestamps is not None:
            result = fn(target, spec.params, fps, timestamps=timestamps)
        else:
            result = fn(target, spec.params, fps)
        if spec.columns is not None:
            if isinstance(result, tuple) or result.shape != target.shape:
                raise ValueError(
                    f"Transform '{spec.type}' cannot take 'columns': it changes "
                    "the number of rows or columns it is given."
                )
            # A float64 copy: an integer stream would truncate the result,
            # and the caller's array stays unchanged.
            data = np.array(data, dtype=np.float64)
            data[:, spec.columns] = result
            result = data

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
    shape matches the input. A single frame has no neighbour to differ from,
    so its difference is zero.
    """
    if len(data) < 2:
        return np.zeros_like(data, dtype=np.float64)
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
    depends on both past and future samples. That makes it
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
    # By default sosfiltfilt extends each end by 3 * ntaps samples, with ntaps
    # computed as below, and raises on input no longer than that, which a
    # short run of frames between two gaps can be. Cap the padding at one
    # sample less than the input; longer input keeps the default padding.
    ntaps = 2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum())
    return sosfiltfilt(sos, data, axis=0, padlen=min(3 * ntaps, len(data) - 1))


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


# ---------------------------------------------------------------------------
# Rotations, angles and polar vectors
#
# Linear interpolation is wrong for these as recorded: a quaternion and its
# negation are the same rotation, an angle jumps by a full turn where it
# wraps, and a polar vector's angle wraps too. Each pre-stage transform below
# puts a stream into a form that ``align: linear`` interpolates correctly;
# the matching post-stage transform, where there is one, restores the
# recorded form. List ``columns`` to apply one to part of a stream.
# ---------------------------------------------------------------------------


def _angle_period(params: dict) -> float:
    return 360.0 if params.get("degrees") else 2 * np.pi


def _radians(values: np.ndarray, params: dict) -> np.ndarray:
    return np.deg2rad(values) if params.get("degrees") else values


def _require_width(name: str, data: np.ndarray, width: int) -> None:
    if data.shape[1] != width:
        raise ValueError(f"{name} takes {width} columns, got {data.shape[1]}; select them with 'columns'.")


@register_transform("quaternion_continuity")
def _quaternion_continuity(data: np.ndarray, params: dict, fps: float, **kwargs: object) -> np.ndarray:
    """Negate each quaternion whose dot product with the one before it is negative.

    q and -q are the same rotation, so no rotation changes, and the stream
    never jumps from q to -q. Use at ``stage: pre`` before ``align: linear``,
    followed by a post-stage ``quaternion_normalize``. Takes the 4 quaternion
    components in any order.
    """
    _require_width("quaternion_continuity", data, 4)
    dots = np.einsum("ij,ij->i", data[1:], data[:-1])
    sign = np.concatenate([[1.0], np.cumprod(np.where(dots < 0, -1.0, 1.0))])
    return data * sign[:, None]


@register_transform("quaternion_normalize")
def _quaternion_normalize(data: np.ndarray, params: dict, fps: float, **kwargs: object) -> np.ndarray:
    """Scale each quaternion to unit length.

    After ``quaternion_continuity`` and ``align: linear`` this gives normalized
    linear interpolation, which follows the same path as SLERP. Two unit
    quaternions whose dot product is not negative are at most 90° apart as
    4-vectors, so a linear blend of the two has norm at least cos 45° ≈ 0.707.
    A norm below 0.7 means two opposite samples were blended, so the transform
    raises rather than return an arbitrary rotation.
    """
    _require_width("quaternion_normalize", data, 4)
    norm = np.linalg.norm(data, axis=1, keepdims=True)
    if (norm < 0.7).any():
        raise ValueError(
            f"quaternion_normalize found a quaternion of norm {float(norm.min()):.3f}: two samples blended "
            "into it point in opposite directions. Add a 'quaternion_continuity' transform with "
            "'stage: pre' on the same columns."
        )
    return data / norm


@register_transform("unwrap_angle")
def _unwrap_angle(data: np.ndarray, params: dict, fps: float, **kwargs: object) -> np.ndarray:
    """Remove the full-turn jumps where each angle column wraps, so it changes continuously.

    Use at ``stage: pre``, and set ``degrees: true`` for angles in degrees.
    Assumes an angle turns less than half a turn between samples.
    """
    return np.unwrap(data, axis=0, period=_angle_period(params))


@register_transform("wrap_angle")
def _wrap_angle(data: np.ndarray, params: dict, fps: float, **kwargs: object) -> np.ndarray:
    """Bring each angle column back into [-180, 180) degrees or [-pi, pi) radians; undoes ``unwrap_angle``."""
    period = _angle_period(params)
    return (data + period / 2) % period - period / 2


@register_transform("polar_to_cartesian")
def _polar_to_cartesian(data: np.ndarray, params: dict, fps: float, **kwargs: object) -> np.ndarray:
    """Turn ``[r, theta]``, theta measured from +x towards +y, into ``[x, y]``.

    Set ``degrees: true`` for theta in degrees.
    """
    _require_width("polar_to_cartesian", data, 2)
    r, theta = data[:, 0], _radians(data[:, 1], params)
    return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)


@register_transform("spherical_to_cartesian")
def _spherical_to_cartesian(data: np.ndarray, params: dict, fps: float, **kwargs: object) -> np.ndarray:
    """Turn ``[r, theta, phi]`` into ``[x, y, z]``.

    theta is the inclination, measured from +z; phi is the azimuth, measured
    from +x towards +y. Set ``degrees: true`` for angles in degrees.
    """
    _require_width("spherical_to_cartesian", data, 3)
    r, theta, phi = data[:, 0], _radians(data[:, 1], params), _radians(data[:, 2], params)
    return np.stack(
        [r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)],
        axis=1,
    )
