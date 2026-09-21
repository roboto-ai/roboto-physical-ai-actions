"""Tests for :func:`resolve_task_label` — wiring the contract's ``tasks:``
stream to the per-episode LeRobot ``task`` string.

Resolution precedence (highest first):

1. The Roboto event's own ``task`` metadata, when present and non-empty.
   This always wins — the contract's ``tasks:`` stream is not even
   consulted.
2. The first ``std_msgs/msg/String`` message from the contract's *first*
   declared task spec whose timestamp falls within the episode window.
3. ``"default"``.

These tests exercise ``resolve_task_label`` directly (contract_utils.py),
building ``DataCollection.tasks`` through the real fetch path with a fake
``Topic`` (mirrors ``test_data_collection_parallel.py``'s ``FakeTopic``) so
the decode step (``std_msgs/msg/String`` -> ``str``) runs for real.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import pytest
from roboto_to_lerobot.contract_utils import (
    Contract,
    DataCollection,
    TaskSpec,
    resolve_task_label,
)
from roboto_to_lerobot.lerobot import generate_frames


@dataclass
class FakeTopic:
    """Minimal ``roboto.Topic``-shaped stub — attribute access + fetch only."""

    topic_id: str
    name: str = ""
    start_time: int | None = None
    end_time: int | None = None
    df: pd.DataFrame = field(default_factory=pd.DataFrame)

    def get_data_as_df(
        self,
        *,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        message_paths_include: list[str] | None = None,
    ) -> pd.DataFrame:
        return self.df.copy(deep=True)


def _string_msg_df(timestamps: list[int], payloads: list[Any]) -> pd.DataFrame:
    """Shaped like what Roboto returns for ``std_msgs/msg/String`` rows —
    one row per message, ``data`` holding the field the decoder reads."""
    return pd.DataFrame({
        "timestamp": pd.Series(timestamps, dtype="int64"),
        "data": payloads,
    })


def _contract(*, tasks: list[TaskSpec], fps: float = 30.0) -> Contract:
    return Contract(
        name="t", version=1, fps=fps, action_lead_steps=0,
        observations=[], videos=[], actions=[], tasks=tasks, robot_type=None,
    )


def _dc_with_tasks(contract: Contract, topics: dict[str, list[FakeTopic]],
                    *, start_ns: int = 0, end_ns: int = 10**9) -> DataCollection:
    return DataCollection(contract, topics, start_ns, end_ns)


# ---------------------------------------------------------------------------
# No tasks: declared, no metadata -> "default".
# ---------------------------------------------------------------------------


def test_no_task_spec_and_no_metadata_falls_back_to_default():
    contract = _contract(tasks=[])
    dc = _dc_with_tasks(contract, {})

    result = resolve_task_label(
        contract=contract, episode_data=dc,
        start_time_ns=0, end_time_ns=1000, metadata_task=None,
    )
    assert result == "default"


# ---------------------------------------------------------------------------
# Metadata present -> always wins, contract task stream not consulted
# (even when it has an in-window message that would otherwise match).
# ---------------------------------------------------------------------------


def test_event_metadata_task_wins_over_task_stream():
    task_spec = TaskSpec(key="task", topic="/task", type="std_msgs/msg/String")
    contract = _contract(tasks=[task_spec])
    topic = FakeTopic(topic_id="tp", df=_string_msg_df([500], ["pick up the block"]))
    dc = _dc_with_tasks(contract, {"/task": [topic]})

    result = resolve_task_label(
        contract=contract, episode_data=dc,
        start_time_ns=0, end_time_ns=1000, metadata_task="stack the blocks",
    )
    assert result == "stack the blocks"


def test_falsy_metadata_task_defers_to_task_stream():
    """An empty-string metadata task is treated as absent (matches the
    pre-existing ``... or "default"`` truthiness convention)."""
    task_spec = TaskSpec(key="task", topic="/task", type="std_msgs/msg/String")
    contract = _contract(tasks=[task_spec])
    topic = FakeTopic(topic_id="tp", df=_string_msg_df([500], ["pick up the block"]))
    dc = _dc_with_tasks(contract, {"/task": [topic]})

    result = resolve_task_label(
        contract=contract, episode_data=dc,
        start_time_ns=0, end_time_ns=1000, metadata_task="",
    )
    assert result == "pick up the block"


# ---------------------------------------------------------------------------
# No metadata -> task stream's first in-window message wins.
# ---------------------------------------------------------------------------


def test_task_stream_message_in_window_is_used_when_metadata_absent():
    task_spec = TaskSpec(key="task", topic="/task", type="std_msgs/msg/String")
    contract = _contract(tasks=[task_spec])
    # Two messages; only the second is inside [1000, 2000].
    topic = FakeTopic(topic_id="tp", df=_string_msg_df(
        [500, 1500], ["ignored (before window)", "pick up the block"],
    ))
    dc = _dc_with_tasks(contract, {"/task": [topic]}, start_ns=0, end_ns=3000)

    result = resolve_task_label(
        contract=contract, episode_data=dc,
        start_time_ns=1000, end_time_ns=2000, metadata_task=None,
    )
    assert result == "pick up the block"


def test_first_in_window_message_wins_when_several_match():
    task_spec = TaskSpec(key="task", topic="/task", type="std_msgs/msg/String")
    contract = _contract(tasks=[task_spec])
    topic = FakeTopic(topic_id="tp", df=_string_msg_df(
        [1100, 1200, 1300], ["first", "second", "third"],
    ))
    dc = _dc_with_tasks(contract, {"/task": [topic]}, start_ns=0, end_ns=3000)

    result = resolve_task_label(
        contract=contract, episode_data=dc,
        start_time_ns=1000, end_time_ns=2000, metadata_task=None,
    )
    assert result == "first"


# ---------------------------------------------------------------------------
# Task stream declared but no message in the episode window -> "default".
# ---------------------------------------------------------------------------


def test_task_stream_with_no_in_window_message_falls_back_to_default():
    task_spec = TaskSpec(key="task", topic="/task", type="std_msgs/msg/String")
    contract = _contract(tasks=[task_spec])
    topic = FakeTopic(topic_id="tp", df=_string_msg_df([50_000], ["far away"]))
    dc = _dc_with_tasks(contract, {"/task": [topic]}, start_ns=0, end_ns=100_000)

    result = resolve_task_label(
        contract=contract, episode_data=dc,
        start_time_ns=1000, end_time_ns=2000, metadata_task=None,
    )
    assert result == "default"


def test_empty_task_stream_falls_back_to_default():
    """The task topic exists in the contract but produced zero rows (e.g.
    the topic was never published in this recording)."""
    task_spec = TaskSpec(key="task", topic="/task", type="std_msgs/msg/String")
    contract = _contract(tasks=[task_spec])
    dc = _dc_with_tasks(contract, {})  # no matching topic fetched at all

    result = resolve_task_label(
        contract=contract, episode_data=dc,
        start_time_ns=1000, end_time_ns=2000, metadata_task=None,
    )
    assert result == "default"
    assert dc.tasks["task"].empty


# ---------------------------------------------------------------------------
# Multiple task specs: first wins, warning names the ignored ones.
# ---------------------------------------------------------------------------


def test_multiple_task_specs_first_wins_and_warns(caplog):
    first = TaskSpec(key="task_a", topic="/task_a", type="std_msgs/msg/String")
    second = TaskSpec(key="task_b", topic="/task_b", type="std_msgs/msg/String")
    contract = _contract(tasks=[first, second])
    topic_a = FakeTopic(topic_id="ta", df=_string_msg_df([1500], ["from A"]))
    topic_b = FakeTopic(topic_id="tb", df=_string_msg_df([1500], ["from B"]))
    dc = _dc_with_tasks(
        contract, {"/task_a": [topic_a], "/task_b": [topic_b]},
        start_ns=0, end_ns=3000,
    )

    with caplog.at_level("WARNING", logger="roboto_to_lerobot"):
        result = resolve_task_label(
            contract=contract, episode_data=dc,
            start_time_ns=1000, end_time_ns=2000, metadata_task=None,
        )

    assert result == "from A"
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("task_a" in m and "only the first" in m for m in warnings)


# ---------------------------------------------------------------------------
# Non-string decoded payload -> clear ValueError naming the spec.
# ---------------------------------------------------------------------------


def test_non_string_payload_raises_value_error_naming_spec():
    # std_msgs/msg/Float32 decodes to a numpy array, not a string — a task
    # spec pointed at the wrong message type must fail loudly, not silently
    # hand a numpy array to the writer as a "task" label.
    task_spec = TaskSpec(key="task", topic="/task", type="std_msgs/msg/Float32")
    contract = _contract(tasks=[task_spec])
    topic = FakeTopic(topic_id="tp", df=pd.DataFrame({
        "timestamp": pd.Series([1500], dtype="int64"),
        "data": [3.14],
    }))
    dc = _dc_with_tasks(contract, {"/task": [topic]}, start_ns=0, end_ns=3000)

    with pytest.raises(ValueError, match="task"):
        resolve_task_label(
            contract=contract, episode_data=dc,
            start_time_ns=1000, end_time_ns=2000, metadata_task=None,
        )


# ---------------------------------------------------------------------------
# Integration: the resolved label actually reaches the frame dicts that
# ``generate_frames`` yields — the seam this feature wires into.
# ---------------------------------------------------------------------------


def test_resolved_task_label_reaches_generated_frames():
    task_spec = TaskSpec(key="task", topic="/task", type="std_msgs/msg/String")
    contract = _contract(tasks=[task_spec])
    topic = FakeTopic(topic_id="tp", df=_string_msg_df([1500], ["pick up the block"]))
    dc = _dc_with_tasks(contract, {"/task": [topic]}, start_ns=0, end_ns=3000)

    resolved = resolve_task_label(
        contract=contract, episode_data=dc,
        start_time_ns=1000, end_time_ns=2000, metadata_task=None,
    )
    assert resolved == "pick up the block"

    # A plain DataFrame (rather than a Series) so ``generate_frames`` has a
    # real ``.iterrows()`` to call even though this contract has no
    # observations/actions/videos to merge onto the timeline — the task
    # stream isn't part of that merge, it's resolved up front instead.
    ref_ts = pd.DataFrame({"timestamp": pd.Series([1000, 2000], dtype="int64")})
    frames = list(generate_frames(contract, dc, ref_ts, task=resolved))
    assert len(frames) == 2
    for f in frames:
        assert f["task"] == "pick up the block"
