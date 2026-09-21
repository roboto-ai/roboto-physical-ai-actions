"""Pure per-frame image ops shared by the converter and the live runtime node.

``resize_image`` is the single home for the shape-guarded ``cv2.resize``
idiom that ``lerobot.py``'s two frame paths use, so the live-inference path
resizes identically to the converter.

``depth_to_uint8_rgb`` is re-exported from ``extract.py`` (its helpers
``_resolve_depth_scale`` / ``_apply_invalid_policy`` / ``_COLORMAP_LUT``
stay there for now) so that the invocation path decoders import is
``runtime.image``. The implementation can move here once the runtime needs
to ship without ``extract.py``'s deps.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..extract import depth_to_uint8_rgb as depth_to_uint8_rgb  # re-export

__all__ = ("depth_to_uint8_rgb", "resize_image")


def resize_image(
    image: np.ndarray, expected_h: int, expected_w: int
) -> np.ndarray:
    """Resize ``image`` to ``(expected_h, expected_w)`` using INTER_LINEAR.

    No-op when the image already has the expected shape — preserves the
    by-design fast path the converter relies on for native-resolution
    streams. cv2.resize takes ``(width, height)`` despite ``shape`` being
    ``(height, width)``; this wrapper swaps so callers can pass them in
    the same order.

    Load-bearing: the no-op branch returns the *same* array object. Callers
    (lerobot.py) rely on this to call ``resize_image`` unconditionally while
    staying byte-equivalent to the old ``if shape != …: cv2.resize`` guard;
    dropping it would run INTER_LINEAR on already-correct frames.
    """
    if image.shape[:2] == (expected_h, expected_w):
        return image
    return cv2.resize(
        image,
        (expected_w, expected_h),
        interpolation=cv2.INTER_LINEAR,
    )
