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
ROS message decoder registry.

Derived from Rosetta's ``rosetta/common/converters.py`` at commit aa04ebe:
https://github.com/iblnkn/rosetta/blob/aa04ebea2519f5c3adab95f6b7f28c2ec24274bc/rosetta/common/converters.py
License text: ``LICENSE-rosetta`` in this directory.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..contract_utils import ActionSpec, ObservationSpec


__all__ = ("DECODERS", "DecoderFn", "decode_value", "register_decoder")


DecoderFn = Callable[[Any, "ObservationSpec | ActionSpec"], np.ndarray | str]


# Registry populated by the ``@register_decoder`` decorators in the sibling
# ``runtime/decoders.py``. Keys are ROS message type strings; values are the
# decoder callables.
DECODERS: dict[str, DecoderFn] = {}


def register_decoder(type_str: str, dtype: str):
    """Register a decoder for a ROS message type.

    ``dtype`` is declarative only: each decoder states the LeRobot dtype it
    produces, but the value is not stored on the registry and no caller reads
    it back today (kept as a documentation hook for a future decoder/encoder
    dtype-agreement check).

    Example:
        @register_decoder("sensor_msgs/msg/JointState", dtype="float64")
        def decode_joint_state(msg, spec):
            return np.array(msg.position, dtype=np.float64)
    """

    def _wrap(fn: DecoderFn):
        DECODERS[type_str] = fn
        return fn

    return _wrap


def decode_value(msg, spec: ObservationSpec | ActionSpec) -> Any:
    """Decode a ROS message using the registered decoder for ``spec.type``.

    Raises ``ValueError`` if no decoder is registered for the message type.
    """
    fn = DECODERS.get(spec.type)
    if not fn:
        raise ValueError(f"No decoder registered for message type: {spec.type}")
    return fn(msg, spec)
