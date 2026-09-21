"""Bag-replay driver for the live runtime — smoke-test only.

Not a public API. The generated node owns rclpy and a real
clock; this adapter substitutes a recorded MCAP bag and a fixed tick
cadence so the kernel (decoders, buffers, encoders) can be exercised
end-to-end under stock Python in CI and locally.

Uses ``mcap_ros2.reader.read_ros2_messages`` to deserialize messages
into ROS message objects whose attribute access (``msg.position``,
``msg.data``, ``msg.name``, ...) matches what the runtime decoders
already understand. Messages are emitted in ``log_time`` order to mimic
arrival order at a live subscriber; the per-tick sample uses the same
``log_time`` so the simulator's clock and the buffer ``ts_ns`` come from
the same source.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from .live_adapter import LiveAdapter

SampleResult = tuple[int, dict[str, np.ndarray] | None]


class ReplayAdapter:
    """Drive a :class:`LiveAdapter` from an MCAP bag.

    On :meth:`run`, the reader fans messages into ``adapter.on_message``
    in log-time order; between messages, ticks fire at fixed
    ``step_ns`` intervals starting at ``bag_start``. Each tick yields
    ``(tick_ts_ns, sample_or_none)`` so callers can collect the policy
    inputs offline.

    Parity caveat: this is a smoke driver, NOT a byte-parity oracle. The
    tick grid is anchored to ``bag_start`` (the first message's log time),
    whereas the converter anchors its grid to the event's ``start_time``.
    Unless the first message lands exactly on the event start, every tick
    is phase-shifted versus the converter grid, so the as-of joins can land
    on different samples. Matching the converter grid exactly would require
    plumbing the event-window origin in and is deferred.

    Args:
        adapter: the :class:`LiveAdapter` to drive — already constructed
            against the same contract whose topics appear in the bag.
        bag_path: path to an MCAP file with the ros2 profile.
        step_ns: tick interval in nanoseconds. Typically
            ``int(1e9 / contract.fps)``.
    """

    def __init__(
        self, adapter: LiveAdapter, bag_path: Path, step_ns: int
    ) -> None:
        if step_ns <= 0:
            raise ValueError(f"step_ns must be > 0, got {step_ns}")
        self._adapter = adapter
        self._bag_path = Path(bag_path)
        self._step_ns = int(step_ns)

    def run(self) -> Iterator[SampleResult]:
        """Replay the bag and yield one ``(tick_ts_ns, sample_or_none)`` per tick.

        ``read_ros2_messages`` is imported here rather than at module top so
        this module declares no direct ``mcap_ros2`` dependency. Importing the
        module still drags ``mcap_ros2`` in transitively via the converter
        stack (see ``test_runtime_replay_adapter_importable_directly``); the
        lazy import keeps the seam ready for the heavy-dep split and defers the
        deserialization cost to actual ReplayAdapter consumers.

        Raises ``ValueError`` if the adapter subscribes to no topics, or if the
        bag yields no messages — for a driver whose whole job is to produce
        samples, an empty result is a setup error, not a valid empty run. A bag
        that carries only topics the contract doesn't consume is not empty: it
        still ticks, yielding ``None`` samples the caller can detect.

        Ticks are emitted only up to the last message's timestamp; there is no
        trailing tick beyond the data window (the converter likewise stops at
        its last in-window grid point).
        """
        from mcap_ros2.reader import read_ros2_messages

        adapter_topics = set(self._adapter.topics)
        if not adapter_topics:
            raise ValueError(
                "ReplayAdapter's LiveAdapter subscribes to no topics; the "
                "contract declares no observations or videos, so replay can "
                "produce nothing. Check the contract."
            )

        bag_start: int | None = None
        next_tick: int | None = None

        for record in read_ros2_messages(
            str(self._bag_path), log_time_order=True
        ):
            ts_ns = record.log_time_ns
            if bag_start is None:
                bag_start = ts_ns
                next_tick = bag_start

            while next_tick is not None and next_tick <= ts_ns:
                yield next_tick, self._adapter.sample(next_tick)
                next_tick += self._step_ns

            topic = record.channel.topic
            if topic not in adapter_topics:
                # Bags often carry topics the contract doesn't consume —
                # ignore them rather than forcing the LiveAdapter to raise.
                continue
            self._adapter.on_message(topic, record.ros_msg, ts_ns)

        if bag_start is None:
            raise ValueError(
                f"Replay produced no ticks: bag {self._bag_path} yielded no "
                "messages. Check the path and that the bag is non-empty."
            )
