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


__all__ = (
    "DECODERS",
    "POSITIONAL_TYPES",
    "DecoderFn",
    "check_positional_width",
    "decode_value",
    "register_decoder",
)


DecoderFn = Callable[[Any, "ObservationSpec | ActionSpec"], np.ndarray | str]


# Registry populated by the ``@register_decoder`` decorators in the sibling
# ``runtime/decoders.py``. Keys are ROS message type strings; values are the
# decoder callables.
DECODERS: dict[str, DecoderFn] = {}

# Message types whose decoder, without ``selector.names``, returns a vector of
# numbers in an order the message fixes (one value per joint, array element or
# field), so ``selector.lerobot_names`` can label that vector by position.
POSITIONAL_TYPES: set[str] = set()


def register_decoder(type_str: str, dtype: str, *, positional: bool = False):
    """Register a decoder for a ROS message type.

    ``dtype`` is declarative only: each decoder states the LeRobot dtype it
    produces, but the value is not stored on the registry and no caller reads
    it back today (kept as a documentation hook for a future decoder/encoder
    dtype-agreement check).

    ``positional`` adds the type to :data:`POSITIONAL_TYPES`: set it when the
    decoder, given no ``selector.names``, returns a vector of numbers whose
    order the message fixes.

    Example:
        @register_decoder("sensor_msgs/msg/JointState", dtype="float64")
        def decode_joint_state(msg, spec):
            return np.array(msg.position, dtype=np.float64)
    """

    def _wrap(fn: DecoderFn):
        DECODERS[type_str] = fn
        if positional:
            POSITIONAL_TYPES.add(type_str)
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


def check_positional_width(value: Any, spec: ObservationSpec | ActionSpec) -> None:
    """Raise ``ValueError`` if ``value`` and the names labelling it by position differ in length.

    A spec labels its decoded vector by position when it sets
    ``selector.lerobot_names`` without ``selector.names``: the first name
    labels the first value, and so on. A message with more or fewer values
    than names would shift or drop labels without any other error, so the
    message is refused. Other specs pass unchecked.
    """
    labels = spec.lerobot_names
    if not labels or (spec.selector or {}).get("names"):
        return
    if len(value) != len(labels):
        raise ValueError(
            f"Spec '{spec.key}' on topic '{spec.topic}': a message decoded to "
            f"{len(value)} values, but selector.lerobot_names labels "
            f"{len(labels)} values by position."
        )
