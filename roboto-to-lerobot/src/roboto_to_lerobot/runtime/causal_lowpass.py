"""Incremental (per-sample) causal Butterworth low-pass filter.

This is the live-runtime counterpart to
``transforms._butterworth_lowpass_causal``: that converter function filters
an entire ``(T, N)`` episode in one ``scipy.signal.sosfilt`` call.
``sosfilt`` implements a strictly causal IIR filter as a per-section
recursive difference equation — each output sample is a function of the
current input and a small, fixed amount of carried state (``zi``), never of
future samples. That means feeding the same samples through one at a time,
carrying ``zi`` across calls, is mathematically identical to a single batch
call over the whole sequence: there is no information the batch call has
that the running one lacks. ``test_causal_lowpass.py`` proves this
numerically against the converter's batch implementation for the same
``(cutoff_hz, order, fs)``.

That equivalence is what makes a causal Butterworth filter reproducible
live at all: training designs and applies the filter offline in one batch
call; the live runtime can only ever see one sample at a time as messages
arrive. :class:`CausalLowpass` is the online form of the same filter, built
from the same ``(cutoff_hz, order, fs)`` the converter used — see
``runtime.live_adapter.LiveAdapter`` for how ``fs`` is pinned to the
contract's declared ``fs_hz`` (pre-stage) or ``contract.fps`` (post-stage)
so the coefficients match training bit-for-bit.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from ..transforms import _validate_cutoff_hz

__all__ = ("CausalLowpass",)


class CausalLowpass:
    """Stateful, sample-by-sample causal Butterworth low-pass filter.

    One instance filters one (spec, transform) pair's channel vector across
    the lifetime of a live run. ``zi`` (the per-section filter state) is
    lazily seeded on the first :meth:`push` call as
    ``sosfilt_zi(sos) * first_value``, mirroring the converter's batch
    seeding (``sosfilt_zi(sos)[:, :, None] * data[0]``) exactly — so the
    online filter starts at the same per-channel steady-state the offline
    batch call does, rather than a zero-initialized state that would
    introduce a spurious startup transient absent from training. Online and
    offline must agree from sample 0, not just after the filter settles.

    Args:
        cutoff_hz: filter cutoff frequency (Hz).
        order: filter order.
        fs: design sample rate (Hz). Must be the *same* value used to
            design the training-time filter — the contract's declared
            ``fs_hz`` for a ``stage: pre`` transform, or ``contract.fps``
            for ``stage: post`` — not necessarily whatever rate the live
            topic happens to publish at. See ``LiveAdapter`` for where
            that value comes from and why it matters.

    Raises:
        ValueError: if ``cutoff_hz`` is not strictly between ``0`` and
            ``fs / 2`` (Nyquist) — see ``transforms._validate_cutoff_hz``.
            Raised from ``__init__`` (construction time), not the first
            :meth:`push`, so a misconfigured contract fails at adapter
            boot rather than mid-episode.
    """

    __slots__ = ("_sos", "_zi")

    def __init__(self, cutoff_hz: float, order: int, fs: float) -> None:
        _validate_cutoff_hz("butterworth_lowpass_causal", cutoff_hz, fs)
        self._sos = butter(order, cutoff_hz, btype="low", output="sos", fs=fs)
        self._zi: np.ndarray | None = None  # lazily seeded on first push()

    def push(self, value: np.ndarray) -> np.ndarray:
        """Filter one sample, carrying filter state across calls.

        Args:
            value: shape-``(n,)`` array — one entry per channel. A scalar
                stream still arrives as shape-``(1,)`` (matching what the
                runtime's decoders emit for scalar message fields), not a
                bare Python float.

        Returns:
            The filtered shape-``(n,)`` sample, same dtype policy as
            ``sosfilt`` (float64 internally; callers cast as needed).
        """
        value = np.asarray(value, dtype=np.float64)
        if self._zi is None:
            # sosfilt_zi(sos) has shape (n_sections, 2); broadcast against
            # this first sample's per-channel value exactly as the
            # converter's batch call seeds zi from data[0].
            self._zi = sosfilt_zi(self._sos)[:, :, None] * value
        out, self._zi = sosfilt(self._sos, value[None, :], axis=0, zi=self._zi)
        return out[0]
