"""Tests for the parallel topic-fetch path in ``DataCollection.__init__``.

Each test pins one load-bearing axis of the parallel fetch:

* parallelism — fetches run concurrently rather than serially;
* equivalence — the parallel path produces the same per-spec DataFrames
  the old serial path produced;
* chunked recordings — multiple Topics with the same name and disjoint
  windows merge in submission order;
* empty-intersection — Topics outside the event window do not get fetched;
* de-dup — work items sharing
  ``(topic_id, start_ns, end_ns, message_paths)`` collapse to one fetch.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from roboto_to_lerobot.contract_utils import (
    _TOPIC_FETCH_THREADS,
    ActionSpec,
    Contract,
    DataCollection,
    ObservationSpec,
    TaskSpec,
    _run_topic_fetches,
    _TopicWorkItem,
)


# A ``Topic``-shaped stub. We deliberately do not subclass ``roboto.Topic``
# — the fetch helpers only need attribute access (``topic_id``, ``start_time``,
# ``end_time``) and the two fetch methods. Counting calls lets the dedup
# test assert that identical work items collapse to a single fetch.
@dataclass
class FakeTopic:
    topic_id: str
    name: str = ""
    start_time: int | None = None
    end_time: int | None = None
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    sleep_s: float = 0.0
    call_log: list[tuple] = field(default_factory=list)
    # Per-Topic counter mutated atomically (GIL covers the int +=).
    call_count: list[int] = field(default_factory=lambda: [0])

    def get_data_as_df(
        self,
        *,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        message_paths_include: list[str] | None = None,
    ) -> pd.DataFrame:
        self.call_count[0] += 1
        self.call_log.append((
            int(start_time.value),
            int(end_time.value),
            tuple(message_paths_include or ()),
        ))
        if self.sleep_s:
            time.sleep(self.sleep_s)
        return self.df.copy(deep=True)


def _make_task_df(timestamps: list[int], values: list[Any]) -> pd.DataFrame:
    """Construct a DataFrame whose ``timestamp`` column is int64 ns.

    Mirrors what ``topic.get_data_as_df`` returns *after* ``_add_timestamp_column``
    has normalised the timestamps — using int64 here means the helper's
    fast-path returns the DF unchanged.
    """
    return pd.DataFrame({"timestamp": pd.Series(timestamps, dtype="int64"),
                         "value": values})


def _task_contract(task_keys_and_topics: list[tuple[str, str]]) -> Contract:
    return Contract(
        name="t", version=1, fps=30, action_lead_steps=0,
        observations=[], videos=[], actions=[],
        tasks=[TaskSpec(key=k, topic=t, type="x") for k, t in task_keys_and_topics],
        robot_type=None,
    )


# ---------------------------------------------------------------------------
# Parallelism: 6 fetches × 0.5 s should complete in ~0.5 s, not ~3 s.
# ---------------------------------------------------------------------------

def test_run_topic_fetches_runs_in_parallel():
    """With ``_TOPIC_FETCH_THREADS >> n_items`` the wall-clock is the *max*
    of per-item sleeps, not the sum."""
    n_items = 6
    sleep_per = 0.4
    items = [
        _TopicWorkItem(
            spec_kind="task", spec_key=f"k{i}", topic_name=f"/t{i}",
            topic=FakeTopic(
                topic_id=f"tp_{i}",
                df=_make_task_df([i * 1000], [i]),
                sleep_s=sleep_per,
            ),
            start_ns=0, end_ns=10**9,
            message_paths_include=None,
            submission_idx=i,
        )
        for i in range(n_items)
    ]
    assert n_items <= _TOPIC_FETCH_THREADS

    t0 = time.monotonic()
    result = _run_topic_fetches(items)
    elapsed = time.monotonic() - t0

    assert len(result) == n_items
    # Slack accounts for thread spin-up + scheduler jitter; serial would be
    # ~n_items × sleep_per (2.4s); parallel should be a small multiple of
    # sleep_per. Anything below ~1.5s only fits the parallel regime.
    assert elapsed < n_items * sleep_per * 0.6, (
        f"elapsed={elapsed:.2f}s — fetches appear to have run serially"
    )


# ---------------------------------------------------------------------------
# Equivalence: parallel path produces the same per-spec results an in-test
# serial reference would.
# ---------------------------------------------------------------------------

def test_parallel_load_matches_serial_reference():
    """Build a DataCollection through the parallel path and compare each
    ``self.tasks[key]`` against an in-test serial concat+sort."""
    df_a1 = _make_task_df([100, 300, 200], ["a", "c", "b"])
    df_a2 = _make_task_df([500, 400], ["e", "d"])
    df_b = _make_task_df([10, 20, 30], ["x", "y", "z"])

    topic_a1 = FakeTopic(topic_id="a1", df=df_a1)
    topic_a2 = FakeTopic(topic_id="a2", df=df_a2)
    topic_b = FakeTopic(topic_id="b", df=df_b)

    topics = {"/A": [topic_a1, topic_a2], "/B": [topic_b]}
    contract = _task_contract([("task_a", "/A"), ("task_b", "/B")])

    dc = DataCollection(contract, topics, 0, 10**9)

    # Serial reference: per spec, concat per-Topic DataFrames in source
    # order and sort by timestamp.
    def _serial(topic_list):
        merged = pd.concat([t.df.copy(deep=True) for t in topic_list],
                            ignore_index=True)
        return merged.sort_values(by="timestamp").reset_index(drop=True)

    pd.testing.assert_frame_equal(dc.tasks["task_a"], _serial(topics["/A"]))
    pd.testing.assert_frame_equal(dc.tasks["task_b"], _serial(topics["/B"]))


# ---------------------------------------------------------------------------
# Chunked recordings: same topic name across 3 Topics with disjoint windows
# concatenate in submission order then sort by timestamp.
# ---------------------------------------------------------------------------

def test_chunked_recording_merges_in_submission_order():
    """Three chunks of the same topic, disjoint windows, intentionally
    presented out of chronological order in the ``topics`` dict — the
    submission-order list is preserved by the fan-out, and the post-concat
    sort produces the chronological result."""
    chunk_a = FakeTopic(topic_id="ca", start_time=0,    end_time=1000,
                        df=_make_task_df([100, 500],   ["a", "a2"]))
    chunk_b = FakeTopic(topic_id="cb", start_time=1000, end_time=2000,
                        df=_make_task_df([1500],        ["b"]))
    chunk_c = FakeTopic(topic_id="cc", start_time=2000, end_time=3000,
                        df=_make_task_df([2100, 2900], ["c", "c2"]))

    # Deliberately not in time order to prove the sort runs.
    topics = {"/A": [chunk_b, chunk_a, chunk_c]}
    contract = _task_contract([("task_a", "/A")])

    dc = DataCollection(contract, topics, 0, 3000)
    out = dc.tasks["task_a"]

    assert list(out["timestamp"]) == [100, 500, 1500, 2100, 2900]
    assert list(out["value"]) == ["a", "a2", "b", "c", "c2"]
    # Every chunk was fetched exactly once.
    assert chunk_a.call_count[0] == 1
    assert chunk_b.call_count[0] == 1
    assert chunk_c.call_count[0] == 1


# ---------------------------------------------------------------------------
# Window-intersection skip: Topics whose time range falls entirely outside
# the requested window must not be fetched.
# ---------------------------------------------------------------------------

def test_topics_outside_window_are_not_fetched():
    """If a Topic's ``end_time`` is before the window's start (or its
    ``start_time`` after the window's end) it gets dropped at work-item
    submission, so ``get_data_as_df`` never runs for it."""
    in_window = FakeTopic(
        topic_id="in", start_time=1000, end_time=2000,
        df=_make_task_df([1500], ["in"]),
    )
    before_window = FakeTopic(
        topic_id="before", start_time=0, end_time=500,
        df=_make_task_df([200], ["b"]),
    )
    after_window = FakeTopic(
        topic_id="after", start_time=5000, end_time=6000,
        df=_make_task_df([5500], ["a"]),
    )

    topics = {"/A": [before_window, in_window, after_window]}
    contract = _task_contract([("task_a", "/A")])

    dc = DataCollection(contract, topics, 1000, 4000)
    out = dc.tasks["task_a"]

    assert list(out["timestamp"]) == [1500]
    assert in_window.call_count[0] == 1
    assert before_window.call_count[0] == 0
    assert after_window.call_count[0] == 0


def test_empty_intersection_yields_empty_dataframe():
    """When *every* Topic is window-excluded, the loader sees an empty
    list and produces an empty DataFrame — no exception, no fetch."""
    out_of_window = FakeTopic(
        topic_id="oow", start_time=5000, end_time=6000,
        df=_make_task_df([5500], ["x"]),
    )
    topics = {"/A": [out_of_window]}
    contract = _task_contract([("task_a", "/A")])

    dc = DataCollection(contract, topics, 0, 1000)

    assert dc.tasks["task_a"].empty
    assert out_of_window.call_count[0] == 0


# ---------------------------------------------------------------------------
# De-dup: two specs whose work items share (topic_id, window, paths)
# collapse to a single fetch. Real-world trigger: an observation and an
# action pinned to the same topic without message-path filtering.
# ---------------------------------------------------------------------------

def test_identical_work_items_dedup_to_one_fetch():
    """Two specs pointing at the same Topic over the same window with
    no message-path filter should fetch exactly once. The result fans out
    to both specs' per-spec result lists."""
    shared = FakeTopic(
        topic_id="shared",
        df=pd.DataFrame({
            "timestamp": pd.Series([10, 20], dtype="int64"),
            "value":     [1.5, 2.5],
        }),
    )
    items = [
        _TopicWorkItem(
            spec_kind="observation", spec_key="obs.k",
            topic_name="/T", topic=shared,
            start_ns=0, end_ns=1000,
            message_paths_include=None,
            submission_idx=0,
        ),
        _TopicWorkItem(
            spec_kind="action", spec_key="act.k",
            topic_name="/T", topic=shared,
            start_ns=0, end_ns=1000,
            message_paths_include=None,
            submission_idx=1,
        ),
    ]

    result = _run_topic_fetches(items)

    assert shared.call_count[0] == 1
    pd.testing.assert_frame_equal(result[("observation", "obs.k")][0],
                                   result[("action", "act.k")][0])


def test_different_message_paths_are_distinct_fetches():
    """Same Topic + window but different ``message_paths_include`` must
    not dedup — pyarrow would project different columns."""
    shared = FakeTopic(
        topic_id="shared",
        df=pd.DataFrame({
            "timestamp": pd.Series([10], dtype="int64"),
            "value":     [9.9],
        }),
    )
    items = [
        _TopicWorkItem(
            spec_kind="video_msgs", spec_key="v1",
            topic_name="/V", topic=shared,
            start_ns=0, end_ns=1000,
            message_paths_include=["data"],
            submission_idx=0,
        ),
        _TopicWorkItem(
            spec_kind="video_msgs", spec_key="v2",
            topic_name="/V", topic=shared,
            start_ns=0, end_ns=1000,
            message_paths_include=["header", "data"],
            submission_idx=1,
        ),
    ]

    _run_topic_fetches(items)
    assert shared.call_count[0] == 2


# ---------------------------------------------------------------------------
# Empty input contract: no specs to walk, no fetches, no exception.
# ---------------------------------------------------------------------------

def test_empty_work_list_returns_empty_dict():
    assert _run_topic_fetches([]) == {}


def test_data_collection_with_empty_contract():
    contract = Contract(
        name="empty", version=1, fps=30, action_lead_steps=0,
        observations=[], videos=[], actions=[], tasks=[], robot_type=None,
    )
    dc = DataCollection(contract, topics={}, start_time_ns=0, end_time_ns=10**9)
    assert dc.tasks == {}
    assert dc.observations == {}
    assert dc.actions == {}
    assert dc.videos == {}


# ---------------------------------------------------------------------------
# Spec without matching topic in ``topics``: work list stays empty; the
# loader produces an empty result without raising.
# ---------------------------------------------------------------------------

def test_spec_with_missing_topic_yields_empty_result():
    contract = _task_contract([("task_a", "/missing")])
    dc = DataCollection(contract, topics={}, start_time_ns=0, end_time_ns=10**9)
    assert dc.tasks["task_a"].empty


# ---------------------------------------------------------------------------
# Stress: many work items + a single Topic that records the threads it
# was called from. The fetch step touches >1 distinct thread, confirming
# the executor parallelised the workload (not just the sleep-based wall-
# clock measurement above, which can flake on a loaded host).
# ---------------------------------------------------------------------------

class _ThreadProbingTopic:
    """Topic stub that records the thread id of each fetch."""
    def __init__(self, topic_id: str):
        self.topic_id = topic_id
        self.start_time = None
        self.end_time = None
        self.thread_ids: list[int] = []
        self._lock = threading.Lock()

    def get_data_as_df(self, *, start_time, end_time, message_paths_include=None):
        # Sleep long enough that the thread pool keeps multiple workers
        # spun up concurrently; without sleep the executor can serialise
        # by reusing the same hot worker.
        time.sleep(0.1)
        with self._lock:
            self.thread_ids.append(threading.get_ident())
        return pd.DataFrame({"timestamp": pd.Series([0], dtype="int64")})


def test_multiple_threads_observed_during_fetch():
    topics = [_ThreadProbingTopic(f"tp_{i}") for i in range(_TOPIC_FETCH_THREADS)]
    items = [
        _TopicWorkItem(
            spec_kind="task", spec_key=f"k{i}", topic_name=f"/t{i}",
            topic=topics[i],
            start_ns=0, end_ns=1000,
            message_paths_include=None,
            submission_idx=i,
        )
        for i in range(_TOPIC_FETCH_THREADS)
    ]
    _run_topic_fetches(items)
    distinct = {tid for t in topics for tid in t.thread_ids}
    assert len(distinct) >= 2, "fetches ran on a single thread"


# ---------------------------------------------------------------------------
# Observations + actions: the integration path exercising decoder-bearing
# specs (``Float32MultiArray`` is one of the simplest registered decoders).
# Proves that ``_load_observations`` / ``_load_actions`` correctly read from
# ``raw_by_spec`` and produce the expected ``values`` column.
# ---------------------------------------------------------------------------

def _float_array_df(timestamps: list[int], arrays: list[list[float]]) -> pd.DataFrame:
    """DataFrame shaped like what Roboto returns for ``Float32MultiArray``
    after ``message_paths_include=None`` — one row per message, ``data``
    holding the array payload the decoder reads from."""
    return pd.DataFrame({
        "timestamp": pd.Series(timestamps, dtype="int64"),
        "data": arrays,
    })


def test_observation_decoder_runs_against_parallel_results():
    contract = Contract(
        name="t", version=1, fps=30, action_lead_steps=0,
        observations=[ObservationSpec(
            key="observation.state", topic="/state",
            type="std_msgs/msg/Float32MultiArray",
            selector={"names": ["a", "b"]},
        )],
        actions=[ActionSpec(
            key="action", topic="/cmd",
            type="std_msgs/msg/Float32MultiArray",
            selector={"names": ["c", "d"]},
        )],
        videos=[], tasks=[], robot_type=None,
    )
    obs_topic = FakeTopic(
        topic_id="obs", df=_float_array_df([10, 20], [[1.0, 2.0], [3.0, 4.0]]),
    )
    act_topic = FakeTopic(
        topic_id="act", df=_float_array_df([15, 25], [[0.1, 0.2], [0.3, 0.4]]),
    )
    topics = {"/state": [obs_topic], "/cmd": [act_topic]}

    dc = DataCollection(contract, topics, 0, 100)

    # ``unique_key`` strips the leading slash from the topic name and
    # replaces inner slashes with underscores.
    obs_df = dc.observations["observation.state.state"]
    act_df = dc.actions["action.cmd"]
    assert list(obs_df["timestamp"]) == [10, 20]
    assert list(act_df["timestamp"]) == [15, 25]
    assert len(obs_df["values"][0]) == 2
    assert len(act_df["values"][0]) == 2
