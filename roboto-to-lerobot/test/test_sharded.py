"""Tests for the sharded-writer path.

The integration that drives real ``multiprocessing.Process`` shards + a real
``aggregate_datasets`` merge is exercised end to end by
``scripts/verify_byte_equivalence.py``. This file covers the pieces that
compose into that integration:

* ``_resolve_shard_count`` — parameter > env > default precedence, mirrors
  ``_resolve_pool_size``.
* ``_partition_shards`` — chronological contiguous splits, empty-chunk
  pruning, no events lost or duplicated.
* ``_renumber_episode_to_event`` — per-shard local episode_index becomes
  the global aggregated index.
* ``_run_pool_drain`` — drain order honoured even when futures complete
  out of order, accumulator counters match the per-frame data.
* ``_warn_oversubscription`` — logs a warning when ``shard_count ×
  encoder_threads`` exceeds the host vCPU count.
* End-to-end shape parity — two shards run synchronously (pool +
  ``_worker_build_episode`` monkey-patched) on the same events produce,
  after renumbering, the same per-episode manifest a single-writer drain
  would have produced.
"""

from __future__ import annotations

from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest
from roboto_to_lerobot.event_worker import EventTask, WorkerEpisode
from roboto_to_lerobot.main import (
    _build_skipped_summary,
    _DrainAccum,
    _drop_empty_shards,
    _partition_shards,
    _renumber_episode_to_event,
    _resolve_shard_count,
    _run_pool_drain,
    _shard_path_supported,
    _warn_oversubscription,
)

# ---------------------------------------------------------------------------
# Test doubles for the synchronous pool + writer.
# ---------------------------------------------------------------------------


class _SyncPool:
    """Drop-in replacement for ``ProcessPoolExecutor`` that runs submitted
    callables inline. Mirrors the surface ``_run_pool_drain`` touches
    (``submit`` returning a pre-resolved Future; ``shutdown`` is a no-op).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._initializer = kwargs.get("initializer")
        self._initargs = kwargs.get("initargs", ())
        if self._initializer is not None:
            # The real pool invokes ``initializer`` once per worker; for
            # the synchronous double we run it exactly once so that any
            # state ``_worker_init`` would set up is still set up before
            # any task runs.
            self._initializer(*self._initargs)

    def submit(self, fn, task):
        f: Future = Future()
        try:
            f.set_result(fn(task))
        except BaseException as exc:
            f.set_exception(exc)
        return f

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        pass


class _FakeWriter:
    """In-memory stand-in for ``LeRobotWriter``. Records every
    ``add_frame`` + ``save_episode`` call so the test can assert frame
    order matches event order and counters match per-episode frame
    counts."""

    def __init__(self) -> None:
        self.episodes: list[list[dict[str, Any]]] = []
        self._cur: list[dict[str, Any]] = []
        self.finalized = False

    def add_frame(self, frame: dict[str, Any], task: str) -> None:
        # ``_drain_one_episode`` passes ``task`` as kwarg; mirror the
        # real adapter and store it on the recorded frame.
        self._cur.append({**frame, "task": task})

    def save_episode(self) -> None:
        self.episodes.append(self._cur)
        self._cur = []

    def finalize(self) -> None:
        self.finalized = True


def _fake_build_episode(task: EventTask) -> WorkerEpisode:
    """Build a deterministic 3-frame episode keyed by ``event_idx``.

    Frame ``x`` carries ``(event_idx, frame_idx)`` so a downstream
    assertion can verify frames were drained in the right order without
    needing to round-trip through ``materialize_deferred``.
    """
    return WorkerEpisode(
        event_idx=task.event_idx,
        task_label=task.task_label,
        frames=[
            {
                "x": np.array([task.event_idx, i], dtype=np.float32),
                "task": task.task_label,
            }
            for i in range(3)
        ],
    )


def _stub_worker_init(*args: Any, **kwargs: Any) -> None:
    """Stub: the real ``_worker_init`` would do HTTP fetches to Roboto."""
    return None


def _patch_drain_internals(monkeypatch) -> None:
    """Swap the drain loop's process-pool internals for synchronous doubles.

    ``__init__.py`` re-exports ``main`` (the function) at the package
    level, which shadows the submodule for string-based lookups —
    ``monkeypatch.setattr("roboto_to_lerobot.main.X", ...)`` resolves
    ``roboto_to_lerobot.main`` to the function and bombs. Go through
    ``sys.modules`` to grab the actual module, then patch attributes
    on it directly.
    """
    import sys
    main_mod = sys.modules["roboto_to_lerobot.main"]
    monkeypatch.setattr(main_mod, "ProcessPoolExecutor", _SyncPool)
    monkeypatch.setattr(main_mod, "_worker_build_episode", _fake_build_episode)
    monkeypatch.setattr(main_mod, "_worker_init", _stub_worker_init)


def _event(i: int, *, ds: str = "ds_a", task: str = "default") -> SimpleNamespace:
    """SimpleNamespace shaped like the bits of ``roboto.Event`` the
    partition function reads — keeps tests free of SDK fixtures."""
    return SimpleNamespace(
        start_time=i * 1_000_000,
        end_time=i * 1_000_000 + 500_000,
        event_id=f"ev_{i}",
    )


def _drain_inputs(n: int) -> tuple[list, list]:
    """Build (matched, matched_events) pairs for ``n`` synthetic events."""
    matched = [(i, "ds_a", "default", 3) for i in range(n)]
    events = [_event(i) for i in range(n)]
    return matched, events


# ---------------------------------------------------------------------------
# _resolve_shard_count: parameter > env > default precedence.
# ---------------------------------------------------------------------------


def test_resolve_shard_count_defaults_to_one(monkeypatch):
    monkeypatch.delenv("ROBOTO_TO_LEROBOT_SHARD_COUNT", raising=False)
    assert _resolve_shard_count() == 1


def test_resolve_shard_count_reads_env(monkeypatch):
    monkeypatch.setenv("ROBOTO_TO_LEROBOT_SHARD_COUNT", "4")
    assert _resolve_shard_count() == 4


def test_resolve_shard_count_param_overrides_env(monkeypatch):
    monkeypatch.setenv("ROBOTO_TO_LEROBOT_SHARD_COUNT", "4")
    assert _resolve_shard_count(2) == 2
    assert _resolve_shard_count("2") == 2


def test_resolve_shard_count_treats_empty_string_as_unset(monkeypatch):
    """An empty ``ROBOTO_TO_LEROBOT_SHARD_COUNT=`` must fall through to the
    default just like ``None`` does — defensive handling of the env-var
    override path so a caller that forwards the var unconditionally is safe.
    (No wrapper currently forwards this var; the live flow passes shard_count
    as an action parameter. The empty-as-unset contract is retained for the
    env-var override itself.)"""
    monkeypatch.setenv("ROBOTO_TO_LEROBOT_SHARD_COUNT", "")
    assert _resolve_shard_count() == 1
    assert _resolve_shard_count(param_value="") == 1


def test_resolve_shard_count_rejects_bad_values(monkeypatch):
    monkeypatch.setenv("ROBOTO_TO_LEROBOT_SHARD_COUNT", "many")
    with pytest.raises(ValueError, match="not an integer"):
        _resolve_shard_count()
    monkeypatch.delenv("ROBOTO_TO_LEROBOT_SHARD_COUNT", raising=False)
    with pytest.raises(ValueError, match=">= 1"):
        _resolve_shard_count(0)


# ---------------------------------------------------------------------------
# _shard_path_supported: the lerobot-0.5-only merge API probe that decides
# whether a shard_count > 1 request is honoured or ignored.
# ---------------------------------------------------------------------------


def test_shard_path_supported_true_when_aggregate_importable():
    """With the merge API present (a lerobot 0.5.x venv, i.e. the v3_0
    variant) a shard_count > 1 request is honoured rather than clamped.

    Skipped rather than asserted unconditionally because
    ``./scripts/setup.sh --lerobot-version 0.3.3`` is a supported dev setup
    for reproducing the v2_1 variant, and 0.3.x has no merge API.
    """
    pytest.importorskip(
        "lerobot.datasets.aggregate",
        reason="venv pinned to lerobot 0.3.x (v2_1 variant); no merge API",
    )
    assert _shard_path_supported() is True


def test_shard_path_supported_false_when_aggregate_missing(monkeypatch):
    """A lerobot 0.3.x image (the v2_1 variant) has no
    ``lerobot.datasets.aggregate``. The probe must report that as an absent
    capability rather than raising, because the caller clamps shard_count to
    1 and carries on: action.json ships shard_count=3 for v3_0's benefit and
    both variants register from it, so a default v2_1 invocation must run,
    not fail.

    Forcing ``None`` into ``sys.modules`` makes the import machinery raise
    ImportError for that name only, so this holds on a 0.5.x venv too.
    """
    import sys

    monkeypatch.setitem(sys.modules, "lerobot.datasets.aggregate", None)
    assert _shard_path_supported() is False


# ---------------------------------------------------------------------------
# _partition_shards: LPT bin-pack + staggered-heavy rotation.
# A frame-balanced bin-pack replaced the earlier contiguous-chunk
# partitioner; it rotates each shard's heavy-first drain so shards don't
# decode their heaviest events at the same wall-clock instant. The set-equivalence tests below cover the
# "every event makes it through, with the right counts" invariant; the
# integration tests at the bottom of the file confirm the partition
# produces a valid merged dataset.
# ---------------------------------------------------------------------------


def _varied_inputs(frame_counts: list[int]) -> tuple[list, list]:
    """Build (matched, matched_events) with per-event frame counts.

    Mirrors :func:`_drain_inputs` but lets each test specify a custom
    frame-count distribution — the staggered partitioner's behaviour
    only differs from naive round-robin when frames vary across events.
    """
    matched = [(i, "ds_a", "default", nf) for i, nf in enumerate(frame_counts)]
    events = [_event(i) for i in range(len(frame_counts))]
    return matched, events


def test_partition_shards_bin_packs_frames_evenly():
    """LPT bin-pack: shard frame totals stay within one event's worth
    of each other for varied frame counts."""
    # Hand-picked to expose imbalance under naive round-robin:
    # [10, 9, 8, 7, 6, 5, 4, 3] / 2 shards → contiguous would give
    # [10+9+8+7=34, 6+5+4+3=18], skew 1.89×. LPT yields a near-perfect
    # split.
    matched, events = _varied_inputs([10, 9, 8, 7, 6, 5, 4, 3])
    chunks_d, _ = _partition_shards(
        matched, events, shard_count=2, buffer_ns=0, action_lead_ns=0,
    )
    totals = []
    for c in chunks_d:
        totals.append(sum(matched[d["event_idx"]][3] for d in c))
    assert max(totals) - min(totals) <= max(matched, key=lambda m: m[3])[3]


def test_partition_shards_staggers_heaviest_event_position():
    """Within-shard rotation puts shard k's heaviest event at offset
    ``k * L // K`` so the K bandwidth-heavy decode windows don't
    synchronise at t=0 across shards."""
    # Four shards × four events each, with one clearly heaviest event
    # per shard after LPT (frames 16,15,14,13 are the heaviest).
    frames = [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    matched, events = _varied_inputs(frames)
    chunks_d, _ = _partition_shards(
        matched, events, shard_count=4, buffer_ns=0, action_lead_ns=0,
    )
    # In each shard, find the position of the shard's heaviest event.
    positions = []
    for c in chunks_d:
        nframes = [matched[d["event_idx"]][3] for d in c]
        positions.append(nframes.index(max(nframes)))
    # Shard 0 leads with its heaviest (position 0); subsequent shards
    # offset by ``k * 4 // 4 = k`` so the heaviest lands at positions
    # 0, 1, 2, 3 respectively.
    assert positions == [0, 1, 2, 3]


def test_partition_shards_preserves_every_event_exactly_once():
    """No event lost or duplicated regardless of frame distribution."""
    frames = [100, 99, 1, 2, 50, 50, 3, 80]
    matched, events = _varied_inputs(frames)
    chunks_d, chunks_t = _partition_shards(
        matched, events, shard_count=3, buffer_ns=0, action_lead_ns=0,
    )
    flat_drain = [d["event_idx"] for c in chunks_d for d in c]
    flat_tasks = [t.event_idx for c in chunks_t for t in c]
    assert sorted(flat_drain) == list(range(len(frames)))
    assert sorted(flat_tasks) == list(range(len(frames)))


def test_partition_shards_more_shards_than_events_drops_empty_chunks():
    """We never spawn an idle shard; empty chunks are pruned so the
    spawned count equals ``len(chunks_d)`` rather than the requested
    ``shard_count``."""
    matched, events = _drain_inputs(2)
    chunks_d, chunks_t = _partition_shards(
        matched, events, shard_count=5, buffer_ns=0, action_lead_ns=0,
    )
    assert len(chunks_d) == 2
    assert len(chunks_t) == 2


def test_partition_shards_propagates_buffer_and_action_lead():
    matched, events = _drain_inputs(2)
    _, chunks_t = _partition_shards(
        matched, events, shard_count=1, buffer_ns=42, action_lead_ns=7,
    )
    assert chunks_t[0][0].buffer_ns == 42
    assert chunks_t[0][0].action_lead_ns == 7


def test_partition_shards_handles_empty_input():
    """Zero events → zero shards, no crash."""
    chunks_d, chunks_t = _partition_shards(
        [], [], shard_count=4, buffer_ns=0, action_lead_ns=0,
    )
    assert chunks_d == []
    assert chunks_t == []


def test_partition_shards_deterministic_across_runs():
    """Identical inputs produce byte-identical partitions — the LPT
    tiebreak and within-shard sort tiebreak both fall back to the
    original slot index for this guarantee."""
    matched, events = _varied_inputs([5, 5, 5, 5, 5, 5])
    runs = [
        _partition_shards(
            matched, events, shard_count=3, buffer_ns=0, action_lead_ns=0,
        )
        for _ in range(3)
    ]
    for chunks_d, _ in runs[1:]:
        assert [[d["event_idx"] for d in c] for c in chunks_d] == \
            [[d["event_idx"] for d in c] for c in runs[0][0]]


# ---------------------------------------------------------------------------
# _renumber_episode_to_event: per-shard 0-based -> global 0-based.
# ---------------------------------------------------------------------------


def test_renumber_episode_to_event_globalises_indices():
    shard_results = [
        {
            "shard_idx": 0,
            "total_episodes": 3,
            "episode_to_event": [
                {"episode_index": 0, "event_id": "a"},
                {"episode_index": 1, "event_id": "b"},
                {"episode_index": 2, "event_id": "c"},
            ],
        },
        {
            "shard_idx": 1,
            "total_episodes": 2,
            "episode_to_event": [
                {"episode_index": 0, "event_id": "d"},
                {"episode_index": 1, "event_id": "e"},
            ],
        },
    ]
    flat = _renumber_episode_to_event(shard_results)
    assert [e["episode_index"] for e in flat] == [0, 1, 2, 3, 4]
    assert [e["event_id"] for e in flat] == ["a", "b", "c", "d", "e"]


def test_renumber_episode_to_event_preserves_other_fields():
    shard_results = [
        {
            "shard_idx": 0,
            "total_episodes": 1,
            "episode_to_event": [
                {"episode_index": 0, "n_frames": 7, "task": "grab"},
            ],
        },
        {
            "shard_idx": 1,
            "total_episodes": 1,
            "episode_to_event": [
                {"episode_index": 0, "n_frames": 9, "task": "stack"},
            ],
        },
    ]
    flat = _renumber_episode_to_event(shard_results)
    assert flat[0]["n_frames"] == 7
    assert flat[0]["task"] == "grab"
    assert flat[1]["n_frames"] == 9
    assert flat[1]["task"] == "stack"


# ---------------------------------------------------------------------------
# _warn_oversubscription: warns when shard_count × encoder_threads > vCPUs.
# ---------------------------------------------------------------------------


def test_warn_oversubscription_fires_on_violation(monkeypatch, caplog):
    import os as _os  # patched globally; monkeypatch undoes after the test
    monkeypatch.setattr(_os, "cpu_count", lambda: 4)
    with caplog.at_level("WARNING", logger="roboto_to_lerobot"):
        _warn_oversubscription(shard_count=4, encoder_threads=2, pool_size=1)
    msgs = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("Oversubscription guard" in m for m in msgs)


def test_warn_oversubscription_quiet_when_within_budget(monkeypatch, caplog):
    import os as _os
    monkeypatch.setattr(_os, "cpu_count", lambda: 16)
    with caplog.at_level("WARNING", logger="roboto_to_lerobot"):
        _warn_oversubscription(shard_count=4, encoder_threads=2, pool_size=2)
    msgs = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert not any("Oversubscription guard" in m for m in msgs)


# ---------------------------------------------------------------------------
# _run_pool_drain: episodes drain in event_idx order, counters match the
# data produced by the (mocked) worker.
# ---------------------------------------------------------------------------


def _make_drain_inputs(matched, events):
    drain: list[dict] = []
    tasks: list[EventTask] = []
    for slot, (event_idx, src_ds_id, task_label, num_frames) in enumerate(matched):
        ev = events[slot]
        drain.append({
            "event_idx": event_idx,
            "src_ds_id": src_ds_id,
            "task_label": task_label,
            "event_id": ev.event_id,
            "start_time_ns": ev.start_time,
            "end_time_ns": ev.end_time,
        })
        tasks.append(EventTask(
            event_idx=event_idx,
            event_id=ev.event_id,
            src_ds_id=src_ds_id,
            start_time_ns=ev.start_time,
            end_time_ns=ev.end_time,
            task_label=task_label,
            num_frames=num_frames,
            buffer_ns=0,
            action_lead_ns=0,
        ))
    return drain, tasks


def test_run_pool_drain_preserves_event_order(monkeypatch):
    """With ``ProcessPoolExecutor`` swapped for the synchronous double,
    every event drains in chronological order and the accumulator's
    ``episode_to_event`` matches the drain order 1:1."""
    _patch_drain_internals(monkeypatch)

    matched, events = _drain_inputs(5)
    drain_order, pool_tasks = _make_drain_inputs(matched, events)
    writer = _FakeWriter()
    accumulator = _DrainAccum()

    _run_pool_drain(
        writer=writer,
        video_specs_by_key={},
        pool_tasks=pool_tasks,
        drain_order=drain_order,
        descriptors_by_ds={},
        contracts_by_ds={},
        pool_size=2,
        accumulator=accumulator,
    )

    # 5 events × 3 frames each = 15 total; counters and writer agree.
    assert accumulator.total_episodes == 5
    assert accumulator.total_frames == 15
    assert len(writer.episodes) == 5
    assert [e["event_id"] for e in accumulator.episode_to_event] == [
        f"ev_{i}" for i in range(5)
    ]
    # ``x`` is (event_idx, frame_idx); first frame of each episode encodes
    # the event_idx — proves we drained in event_idx order, not pool-
    # completion order.
    first_frames = [ep[0]["x"][0] for ep in writer.episodes]
    assert first_frames == [0, 1, 2, 3, 4]


def test_run_pool_drain_with_preloaded_event(monkeypatch):
    """The preloaded event slot drains inline (no future submitted for
    it) while the rest go through the pool; episode_to_event still
    comes out in chronological order."""
    _patch_drain_internals(monkeypatch)

    matched, events = _drain_inputs(4)
    drain_order, _ = _make_drain_inputs(matched, events)
    # Preload event_idx=2 — neither the first nor the last, to make sure
    # the loop handles inline-drain mid-sequence.
    preloaded_idx = 2
    pool_tasks = [
        EventTask(
            event_idx=i, event_id=f"ev_{i}", src_ds_id="ds_a",
            start_time_ns=i * 1_000_000, end_time_ns=i * 1_000_000 + 500_000,
            task_label="default", num_frames=3,
            buffer_ns=0, action_lead_ns=0,
        )
        for i in range(4) if i != preloaded_idx
    ]
    preloaded_episode = WorkerEpisode(
        event_idx=preloaded_idx,
        task_label="default",
        frames=[
            {"x": np.array([preloaded_idx, i], dtype=np.float32), "task": "default"}
            for i in range(3)
        ],
    )

    writer = _FakeWriter()
    accumulator = _DrainAccum()
    _run_pool_drain(
        writer=writer,
        video_specs_by_key={},
        pool_tasks=pool_tasks,
        drain_order=drain_order,
        descriptors_by_ds={},
        contracts_by_ds={},
        pool_size=2,
        accumulator=accumulator,
        preloaded_event_idx=preloaded_idx,
        preloaded_episode=preloaded_episode,
    )

    assert accumulator.total_episodes == 4
    assert [e["episode_index"] for e in accumulator.episode_to_event] == [0, 1, 2, 3]
    # Preloaded episode lands in chronological position (slot 2), proven
    # by recovering the event_idx from the first frame.
    assert [ep[0]["x"][0] for ep in writer.episodes] == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# Sharded shape parity — the headline guarantee. Two shards processing
# disjoint chronological slices, then renumbered, yield the same per-episode
# manifest a single-writer drain would have produced.
# ---------------------------------------------------------------------------


def _drain_one_shard(matched_chunk, events_chunk, monkeypatch):
    """Helper: run ``_run_pool_drain`` synchronously and return the
    accumulator + writer (simulates what ``_run_shard_subprocess`` does
    inside a real shard)."""
    _patch_drain_internals(monkeypatch)

    drain_order, pool_tasks = _make_drain_inputs(matched_chunk, events_chunk)
    writer = _FakeWriter()
    accumulator = _DrainAccum()
    _run_pool_drain(
        writer=writer,
        video_specs_by_key={},
        pool_tasks=pool_tasks,
        drain_order=drain_order,
        descriptors_by_ds={},
        contracts_by_ds={},
        pool_size=2,
        accumulator=accumulator,
    )
    return accumulator, writer


def test_sharded_merge_matches_single_writer_shape(monkeypatch):
    """End-to-end shape parity: a K-shard run + renumber yields the
    same set of (event_id, n_frames, task, ...) entries the single-
    writer drain produces.

    After the bin-pack partitioner switch, the merged dataset is no
    longer globally chronological — episodes are grouped by shard, not
    by start_time.
    The invariant that still must hold is "every event makes it
    through, with the right frame count / task / timestamps" — i.e.
    set-equivalence, not order-equivalence. Order-sensitive
    consumers can resort by ``start_time_ns`` from the manifest.
    """
    n = 6
    matched, events = _drain_inputs(n)

    # Single-writer baseline (chronological order by construction).
    baseline_acc, baseline_writer = _drain_one_shard(matched, events, monkeypatch)

    # Two-shard run: partition then drain each chunk independently.
    chunks_drain, _ = _partition_shards(
        matched, events, shard_count=2, buffer_ns=0, action_lead_ns=0,
    )
    assert len(chunks_drain) == 2

    shard_results = []
    shard_writers = []
    for i, chunk in enumerate(chunks_drain):
        chunk_matched = [matched[d["event_idx"]] for d in chunk]
        chunk_events = [events[d["event_idx"]] for d in chunk]
        acc, w = _drain_one_shard(chunk_matched, chunk_events, monkeypatch)
        shard_writers.append(w)
        shard_results.append({
            "shard_idx": i,
            "total_episodes": acc.total_episodes,
            "total_frames": acc.total_frames,
            "per_dataset_counts": dict(acc.per_dataset_counts),
            "episode_to_event": acc.episode_to_event,
        })

    # Per-shard totals add up to the baseline.
    assert sum(r["total_episodes"] for r in shard_results) == baseline_acc.total_episodes
    assert sum(r["total_frames"] for r in shard_results) == baseline_acc.total_frames

    merged = _renumber_episode_to_event(shard_results)
    assert len(merged) == len(baseline_acc.episode_to_event)

    # Merged manifest carries a contiguous 0..N-1 episode_index ...
    assert [e["episode_index"] for e in merged] == list(range(n))

    # ... and the set of (event_id-keyed) episode entries equals the
    # baseline's, modulo episode_index ordering.
    def _by_event(entries):
        return {
            e["event_id"]: {k: v for k, v in e.items() if k != "episode_index"}
            for e in entries
        }
    assert _by_event(merged) == _by_event(baseline_acc.episode_to_event)

    # Frames per shard: concatenated, the multiset of episodes equals
    # the baseline. Order differs (bin-pack groups by shard, baseline
    # is chronological) — sort both by the event_idx encoded in the
    # first frame's ``x`` to compare apples to apples.
    concat_episodes = [ep for w in shard_writers for ep in w.episodes]
    assert len(concat_episodes) == len(baseline_writer.episodes)
    got_by_ev = {int(ep[0]["x"][0]): ep for ep in concat_episodes}
    want_by_ev = {int(ep[0]["x"][0]): ep for ep in baseline_writer.episodes}
    assert got_by_ev.keys() == want_by_ev.keys()
    for ev_idx in sorted(want_by_ev):
        got_ep, want_ep = got_by_ev[ev_idx], want_by_ev[ev_idx]
        assert len(got_ep) == len(want_ep)
        for got_f, want_f in zip(got_ep, want_ep, strict=True):
            np.testing.assert_array_equal(got_f["x"], want_f["x"])
            assert got_f["task"] == want_f["task"]


def test_sharded_merge_with_three_shards(monkeypatch):
    """Same shape parity claim with shard_count=3 and 7 events. Tests
    the partitioner survives an event count that doesn't divide evenly
    into the requested shard count."""
    n = 7
    matched, events = _drain_inputs(n)
    baseline_acc, _ = _drain_one_shard(matched, events, monkeypatch)

    chunks_drain, _ = _partition_shards(
        matched, events, shard_count=3, buffer_ns=0, action_lead_ns=0,
    )

    shard_results = []
    for i, chunk in enumerate(chunks_drain):
        chunk_matched = [matched[d["event_idx"]] for d in chunk]
        chunk_events = [events[d["event_idx"]] for d in chunk]
        acc, _ = _drain_one_shard(chunk_matched, chunk_events, monkeypatch)
        shard_results.append({
            "shard_idx": i,
            "total_episodes": acc.total_episodes,
            "total_frames": acc.total_frames,
            "per_dataset_counts": dict(acc.per_dataset_counts),
            "episode_to_event": acc.episode_to_event,
        })

    merged = _renumber_episode_to_event(shard_results)
    # Same event set, contiguous episode_index, no event lost.
    assert sorted(e["event_id"] for e in merged) == sorted(
        e["event_id"] for e in baseline_acc.episode_to_event
    )
    assert [e["episode_index"] for e in merged] == list(range(n))


# ---------------------------------------------------------------------------
# Per-event soft-drop: a single failing event is logged + recorded in the
# accumulator's ``skipped_events`` list, and the rest of the drain continues.
# ---------------------------------------------------------------------------


class _FailingWriter(_FakeWriter):
    """``_FakeWriter`` that fails ``save_episode`` for a configurable set of
    save-call indices and tracks ``discard_episode`` invocations.

    Indexed by call count (0 = first ``save_episode``) rather than event_idx
    because the writer doesn't see event_idx — matches what the real
    LeRobotDataset adapter sees in production.
    """

    def __init__(self, *, fail_on_save_calls: set[int] | None = None) -> None:
        super().__init__()
        self.fail_on_save_calls = fail_on_save_calls or set()
        self.save_call_count = 0
        self.discard_calls = 0

    def save_episode(self) -> None:
        idx = self.save_call_count
        self.save_call_count += 1
        if idx in self.fail_on_save_calls:
            raise RuntimeError(f"injected save failure on call {idx}")
        super().save_episode()

    def discard_episode(self) -> None:
        self.discard_calls += 1
        # Mirror what the real adapter does: drop the in-progress buffer
        # so the next add_frame starts fresh.
        self._cur = []


def _failing_build_episode_factory(fail_event_idxs: set[int]):
    """Return a ``_worker_build_episode`` replacement that raises for the
    listed ``event_idx`` values, otherwise builds a normal episode."""

    def _build(task: EventTask) -> WorkerEpisode:
        if task.event_idx in fail_event_idxs:
            raise RuntimeError(f"injected worker failure for event {task.event_idx}")
        return _fake_build_episode(task)

    return _build


def test_pool_drain_skips_worker_failure(monkeypatch):
    """A worker-side raise on event 1 of 3 drops only that episode; the
    remaining events drain normally and get contiguous episode indices."""
    import sys
    main_mod = sys.modules["roboto_to_lerobot.main"]
    monkeypatch.setattr(main_mod, "ProcessPoolExecutor", _SyncPool)
    monkeypatch.setattr(
        main_mod, "_worker_build_episode",
        _failing_build_episode_factory({1}),
    )
    monkeypatch.setattr(main_mod, "_worker_init", _stub_worker_init)

    matched, events = _drain_inputs(3)
    drain_order, pool_tasks = _make_drain_inputs(matched, events)
    writer = _FailingWriter()
    accumulator = _DrainAccum()

    _run_pool_drain(
        writer=writer,
        video_specs_by_key={},
        pool_tasks=pool_tasks,
        drain_order=drain_order,
        descriptors_by_ds={},
        contracts_by_ds={},
        pool_size=2,
        accumulator=accumulator,
    )

    # Event 1 is dropped; events 0 and 2 survive with contiguous indices.
    assert accumulator.total_episodes == 2
    assert [e["episode_index"] for e in accumulator.episode_to_event] == [0, 1]
    assert [e["event_id"] for e in accumulator.episode_to_event] == ["ev_0", "ev_2"]
    assert len(accumulator.skipped_events) == 1
    skip = accumulator.skipped_events[0]
    assert skip["stage"] == "worker"
    assert skip["event_id"] == "ev_1"
    assert skip["error_class"] == "RuntimeError"
    assert "injected worker failure" in skip["error_message"]
    assert "Traceback" in skip["traceback"]
    # Writer was never touched for the failed event → no discard call.
    assert writer.discard_calls == 0


def test_pool_drain_skips_writer_failure_and_recovers(monkeypatch):
    """A ``save_episode`` failure mid-drain triggers ``discard_episode`` on
    the writer, then the next event saves cleanly."""
    _patch_drain_internals(monkeypatch)

    matched, events = _drain_inputs(3)
    drain_order, pool_tasks = _make_drain_inputs(matched, events)
    # Fail the second save (call index 1, i.e. event_idx=1's save).
    writer = _FailingWriter(fail_on_save_calls={1})
    accumulator = _DrainAccum()

    _run_pool_drain(
        writer=writer,
        video_specs_by_key={},
        pool_tasks=pool_tasks,
        drain_order=drain_order,
        descriptors_by_ds={},
        contracts_by_ds={},
        pool_size=2,
        accumulator=accumulator,
    )

    assert accumulator.total_episodes == 2
    assert [e["event_id"] for e in accumulator.episode_to_event] == ["ev_0", "ev_2"]
    assert len(accumulator.skipped_events) == 1
    skip = accumulator.skipped_events[0]
    assert skip["stage"] == "drain"
    assert skip["event_id"] == "ev_1"
    # Writer.discard_episode was called exactly once for the failed save.
    assert writer.discard_calls == 1
    # The next episode's frames are present and uncontaminated.
    assert len(writer.episodes) == 2
    assert writer.episodes[-1][0]["x"][0] == 2  # event_idx=2 in frame 0


def test_pool_drain_only_preloaded_failure(monkeypatch):
    """The single-event preload short-circuit path also soft-drops on a
    drain-side failure — no exception escapes."""
    _patch_drain_internals(monkeypatch)

    matched, events = _drain_inputs(1)
    drain_order, _ = _make_drain_inputs(matched, events)
    preloaded_episode = WorkerEpisode(
        event_idx=0,
        task_label="default",
        frames=[
            {"x": np.array([0, i], dtype=np.float32), "task": "default"}
            for i in range(3)
        ],
    )
    # Force the very first save to fail.
    writer = _FailingWriter(fail_on_save_calls={0})
    accumulator = _DrainAccum()

    _run_pool_drain(
        writer=writer,
        video_specs_by_key={},
        pool_tasks=[],
        drain_order=drain_order,
        descriptors_by_ds={},
        contracts_by_ds={},
        pool_size=1,
        accumulator=accumulator,
        preloaded_event_idx=0,
        preloaded_episode=preloaded_episode,
    )

    assert accumulator.total_episodes == 0
    assert len(accumulator.skipped_events) == 1
    assert accumulator.skipped_events[0]["stage"] == "drain"
    assert writer.discard_calls == 1


def test_pool_drain_continues_when_all_events_fail(monkeypatch):
    """If every event fails (here: every worker raises), the drain returns
    normally with zero successful episodes and N skip entries."""
    import sys
    main_mod = sys.modules["roboto_to_lerobot.main"]
    monkeypatch.setattr(main_mod, "ProcessPoolExecutor", _SyncPool)
    monkeypatch.setattr(
        main_mod, "_worker_build_episode",
        _failing_build_episode_factory({0, 1, 2}),
    )
    monkeypatch.setattr(main_mod, "_worker_init", _stub_worker_init)

    matched, events = _drain_inputs(3)
    drain_order, pool_tasks = _make_drain_inputs(matched, events)
    writer = _FailingWriter()
    accumulator = _DrainAccum()

    # Must not raise.
    _run_pool_drain(
        writer=writer,
        video_specs_by_key={},
        pool_tasks=pool_tasks,
        drain_order=drain_order,
        descriptors_by_ds={},
        contracts_by_ds={},
        pool_size=2,
        accumulator=accumulator,
    )

    assert accumulator.total_episodes == 0
    assert accumulator.total_frames == 0
    assert len(accumulator.skipped_events) == 3
    assert {s["stage"] for s in accumulator.skipped_events} == {"worker"}
    assert sorted(s["event_id"] for s in accumulator.skipped_events) == [
        "ev_0", "ev_1", "ev_2",
    ]


def test_shard_subprocess_propagates_skipped_events(monkeypatch, tmp_path):
    """``_run_shard_subprocess`` puts ``skipped_events`` into its result JSON
    so the parent can fold per-shard skips into the manifest."""
    import json
    import sys
    main_mod = sys.modules["roboto_to_lerobot.main"]
    monkeypatch.setattr(main_mod, "ProcessPoolExecutor", _SyncPool)
    monkeypatch.setattr(
        main_mod, "_worker_build_episode",
        _failing_build_episode_factory({1}),
    )
    monkeypatch.setattr(main_mod, "_worker_init", _stub_worker_init)

    class _WriterFactory:
        """Provide a ``create`` classmethod matching the writer Protocol —
        ``_run_shard_subprocess`` calls ``LeRobotWriter.create(...)`` not
        an instance constructor."""
        instances: ClassVar[list[_FailingWriter]] = []

        @classmethod
        def create(cls, **_kwargs):
            w = _FailingWriter()
            cls.instances.append(w)
            return w

    monkeypatch.setattr(main_mod, "LeRobotWriter", _WriterFactory)

    matched, events = _drain_inputs(3)
    drain_order, pool_tasks = _make_drain_inputs(matched, events)
    result_path = tmp_path / "shard-000.result.json"

    main_mod._run_shard_subprocess(
        shard_idx=0,
        log_level=20,  # INFO
        shard_root=str(tmp_path / "shard-000"),
        repo_id="shard-000",
        fps=30,
        features={},
        robot_type="unknown",
        encoder_threads=1,
        pool_size=2,
        video_specs_by_key={},
        descriptors_by_ds={},
        contracts_by_ds={},
        pool_tasks=pool_tasks,
        drain_order=drain_order,
        result_path=str(result_path),
    )

    payload = json.loads(result_path.read_text())
    assert payload["status"] == "ok"
    assert payload["total_episodes"] == 2
    assert len(payload["skipped_events"]) == 1
    skip = payload["skipped_events"][0]
    assert skip["stage"] == "worker"
    assert skip["event_id"] == "ev_1"


def test_build_skipped_summary_counters_match_per_field():
    """``_build_skipped_summary`` produces the manifest rollup; verify it
    counts by stage, error class, and source dataset, and dedupes the
    event_ids list."""
    skipped = [
        {
            "event_id": "ev_1",
            "source_dataset_id": "ds_a",
            "stage": "worker",
            "error_class": "RuntimeError",
        },
        {
            "event_id": "ev_2",
            "source_dataset_id": "ds_a",
            "stage": "drain",
            "error_class": "RuntimeError",
        },
        {
            "event_id": "ev_3",
            "source_dataset_id": "ds_b",
            "stage": "preload",
            "error_class": "ValueError",
        },
        {
            "event_id": None,
            "source_dataset_id": "ds_b",
            "stage": "worker",
            "error_class": "ValueError",
        },
    ]
    summary = _build_skipped_summary(skipped)
    assert summary["count"] == 4
    assert summary["by_stage"] == {"worker": 2, "drain": 1, "preload": 1}
    assert summary["by_error_class"] == {"RuntimeError": 2, "ValueError": 2}
    assert summary["by_source_dataset"] == {"ds_a": 2, "ds_b": 2}
    # ``None`` event_ids drop out of the visible list — they're useless for
    # tracing back to a source event.
    assert summary["event_ids"] == ["ev_1", "ev_2", "ev_3"]


def test_build_skipped_summary_empty_input():
    """No skips → zero counters, empty event_ids list — never a None."""
    summary = _build_skipped_summary([])
    assert summary == {
        "count": 0,
        "by_stage": {},
        "by_error_class": {},
        "by_source_dataset": {},
        "event_ids": [],
    }


# ---------------------------------------------------------------------------
# _drop_empty_shards: a shard whose every event soft-dropped still finalizes
# and reports ``status: "ok"`` / ``total_episodes: 0`` (see
# ``_run_shard_subprocess``). Passing such a shard into ``aggregate_datasets``
# crashes deep inside lerobot with a misleading Hugging Face Hub 401 instead
# of surfacing the real per-event drop reasons already sitting in
# ``skipped_events``. These tests exercise the guard that catches this before
# ``_aggregate_shards`` ever runs.
# ---------------------------------------------------------------------------


def _skip_entry(event_id: str, *, stage: str = "worker", error_class: str = "RuntimeError") -> dict:
    """Minimal skip-record shape — only the fields ``_build_skipped_summary``
    (and therefore ``_summarize_skip_reasons``) reads."""
    return {
        "event_id": event_id,
        "source_dataset_id": "ds_a",
        "stage": stage,
        "error_class": error_class,
        "error_message": f"injected failure for {event_id}",
    }


def test_drop_empty_shards_excludes_empty_and_keeps_non_empty(caplog):
    """One empty shard + one non-empty shard: the empty one is excluded from
    the returned (aggregatable) list, and a WARNING names its shard index and
    drop reasons."""
    shard_results = [
        {
            "shard_idx": 0,
            "repo_id": "shard-000",
            "total_episodes": 0,
            "skipped_events": [_skip_entry("ev_0"), _skip_entry("ev_1")],
        },
        {
            "shard_idx": 1,
            "repo_id": "shard-001",
            "total_episodes": 2,
            "skipped_events": [],
        },
    ]

    with caplog.at_level("WARNING", logger="roboto_to_lerobot"):
        result = _drop_empty_shards(shard_results)

    assert result == [shard_results[1]]

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    msg = warnings[0].getMessage()
    assert "0" in msg  # names the empty shard's index
    assert "ev_0" in msg and "ev_1" in msg  # summarised drop reasons


def test_drop_empty_shards_all_empty_raises_with_drop_reasons():
    """Every shard producing 0 episodes must raise loudly with the drop
    reasons front and center, instead of silently reaching
    ``_aggregate_shards`` and masking them behind a downstream HF-Hub 401."""
    shard_results = [
        {
            "shard_idx": 0,
            "repo_id": "shard-000",
            "total_episodes": 0,
            "skipped_events": [_skip_entry("ev_0", error_class="ValueError")],
        },
        {
            "shard_idx": 1,
            "repo_id": "shard-001",
            "total_episodes": 0,
            "skipped_events": [_skip_entry("ev_1")],
        },
    ]

    with pytest.raises(RuntimeError) as exc_info:
        _drop_empty_shards(shard_results)

    msg = str(exc_info.value)
    assert "All 2 shard(s) produced 0 episodes" in msg
    assert "ev_0" in msg and "ev_1" in msg
    assert "ValueError" in msg and "RuntimeError" in msg


def test_drop_empty_shards_all_non_empty_is_a_noop(caplog):
    """No empty shards → the input passes through unchanged and no warning
    fires."""
    shard_results = [
        {"shard_idx": 0, "repo_id": "shard-000", "total_episodes": 3, "skipped_events": []},
        {"shard_idx": 1, "repo_id": "shard-001", "total_episodes": 1, "skipped_events": []},
    ]

    with caplog.at_level("WARNING", logger="roboto_to_lerobot"):
        result = _drop_empty_shards(shard_results)

    assert result == shard_results
    assert not [r for r in caplog.records if r.levelname == "WARNING"]
