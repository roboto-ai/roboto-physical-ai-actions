"""Per-event worker for the process-pool fan-out.

Each worker runs fetch + alignment + frame assembly for one event and ships
the resulting frame list back to the main process, which serialises
``writer.add_frame`` + ``writer.save_episode`` calls. The work graph is:

1. ``main.py`` builds one ``EventTask`` per event and submits them all to a
   ``ProcessPoolExecutor`` pinned to ``mp_context="spawn"``. Spawn re-imports
   the module in each worker — no inherited threads, fds, or half-initialised
   lock state from the parent's writer or its ffmpeg encoder subprocess. Auth
   still works because subprocesses inherit env vars and
   ``RobotoClient.defaulted()`` re-reads the on-disk config inside
   ``_worker_init``.
2. ``_worker_init`` reconstructs ``roboto.Topic`` objects from
   ``TopicDescriptor`` payloads via ``Topic.from_id`` — one HTTP GET per
   topic per worker, amortised across every event that worker processes.
3. ``_worker_build_episode`` runs the full ``DataCollection`` +
   ``generate_frames`` pipeline with ``defer_image_decode=True``, so video
   keys travel back across IPC as ``_DeferredFrame`` sentinels carrying raw
   encoded bytes rather than decoded HWC RGB arrays. The main process runs
   ``materialize_deferred`` before each ``writer.add_frame`` call.

Pickling raw bytes is far cheaper than pickling decoded numpy arrays, which
matters because every video frame crosses the IPC boundary. Tabular numpy
arrays (observations / actions) are small enough to pickle as-is.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import roboto

from .contract_utils import (
    _TOPIC_FETCH_THREADS,
    Contract,
    DataCollection,
    fps_to_time_step_ns,
    resolve_task_label,
)
from .lerobot import generate_frames

# Concurrency cap for the per-worker ``Topic.from_id`` storm. Spawn-mode
# workers don't inherit the parent's connection pool, so every ``from_id``
# pays full TLS + RTT; serialising N of them makes warm-up latency-bound
# in N rather than in one round-trip. Reusing the per-event topic-fetch
# thread budget keeps the global socket ceiling at
# ``pool_size × _TOPIC_FETCH_THREADS`` whether init or a steady-state
# fetch wave is active, not the sum of both.
_WORKER_INIT_THREADS = _TOPIC_FETCH_THREADS


class TopicDescriptor(NamedTuple):
    """Picklable subset of ``roboto.Topic`` state needed to fetch fresh.

    ``Topic.from_id`` returns a fully-functional Topic given just the id, so
    the descriptor only has to carry enough metadata for the work-list
    pre-filter (``_topic_intersects_window``) plus a stable file/topic id pair
    for logging. The actual fetch pulls data over the wire.
    """

    topic_id: str
    file_id: str | None
    topic_name: str
    start_time_ns: int | None
    end_time_ns: int | None
    message_count: int | None


class EventTask(NamedTuple):
    """One unit of work shipped to a worker.

    ``event_idx`` is the chronological index of this event among all events
    in the run. The main process drains futures in ``event_idx`` order, which
    keeps the writer's episode_index assignment deterministic regardless of
    worker completion order.

    ``task_label`` is the *raw* Roboto event ``task`` metadata only (``None``
    when absent/empty) — it is not yet the final LeRobot task string.
    ``_worker_build_episode`` resolves the final label via
    ``resolve_task_label`` once the episode's ``DataCollection`` (and thus
    any contract ``tasks:`` stream) is in hand, and that resolved value is
    what ends up on ``WorkerEpisode.task_label``.
    """

    event_idx: int
    event_id: str | None
    src_ds_id: str
    start_time_ns: int
    end_time_ns: int
    task_label: str | None
    num_frames: int
    buffer_ns: int
    action_lead_ns: int


# Module-level state because ``ProcessPoolExecutor`` calls the initializer
# once per worker and stores nothing of its own — the dict is how the
# initializer hands state to subsequent task calls in the same worker.
_WORKER_STATE: dict[str, Any] = {}


def _worker_init(
    descriptors_by_ds: dict[str, dict[str, list[TopicDescriptor]]],
    contracts_by_ds: dict[str, Contract],
) -> None:
    """Reconstruct per-worker ``Topic`` objects and stash them on the module.

    Issues one HTTP GET per (ds_id, topic_id), fanned out across a
    ``ThreadPoolExecutor`` so warm-up cost scales with one round-trip
    rather than with descriptor count. The thread budget is capped at
    ``_TOPIC_FETCH_THREADS`` so init and steady-state event fetches share
    the same per-worker socket ceiling.
    """
    client = roboto.RobotoClient.defaulted()

    # Flatten so one thread pool parallelises across datasets, topic names,
    # and chunked descriptors at once rather than per-bucket.
    work: list[tuple[str, str, int, TopicDescriptor]] = []
    for ds_id, by_name in descriptors_by_ds.items():
        for name, descriptors in by_name.items():
            for idx, descriptor in enumerate(descriptors):
                work.append((ds_id, name, idx, descriptor))

    topics_by_ds: dict[str, dict[str, list[roboto.Topic | None]]] = {
        ds_id: {
            name: [None] * len(descriptors) for name, descriptors in by_name.items()
        }
        for ds_id, by_name in descriptors_by_ds.items()
    }

    if work:
        max_workers = min(_WORKER_INIT_THREADS, len(work))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    roboto.Topic.from_id,
                    descriptor.topic_id,
                    roboto_client=client,
                ): (ds_id, name, idx)
                for ds_id, name, idx, descriptor in work
            }
            for fut in futures:
                ds_id, name, idx = futures[fut]
                topics_by_ds[ds_id][name][idx] = fut.result()

    _WORKER_STATE["topics"] = topics_by_ds
    _WORKER_STATE["contracts"] = contracts_by_ds


def _worker_build_episode(
    task: EventTask,
) -> WorkerEpisode:
    """Build one episode in the worker.

    Returns the event index (for in-order drain on the main process), the
    resolved task label, and the list of frame dicts. Video entries in each
    frame carry ``_DeferredFrame`` sentinels — the main process decodes them
    just before handing each frame to ``writer.add_frame``.
    """
    contract = _WORKER_STATE["contracts"][task.src_ds_id]
    topics = _WORKER_STATE["topics"][task.src_ds_id]

    episode_data = DataCollection(
        contract=contract,
        topics=topics,
        start_time_ns=task.start_time_ns - task.buffer_ns,
        end_time_ns=task.end_time_ns + task.buffer_ns + task.action_lead_ns,
    )

    # ``task.task_label`` is only the raw event metadata (may be ``None``);
    # resolve the final label now that ``episode_data`` — and thus any
    # contract ``tasks:`` stream — is available. The episode window here is
    # the unbuffered event boundary, not the buffered fetch range above.
    resolved_task_label = resolve_task_label(
        contract=contract,
        episode_data=episode_data,
        start_time_ns=task.start_time_ns,
        end_time_ns=task.end_time_ns,
        metadata_task=task.task_label,
    )

    time_step_ns = fps_to_time_step_ns(contract.fps)
    reference_timestamps = pd.Series(
        np.arange(task.num_frames, dtype=np.int64) * time_step_ns + task.start_time_ns,
        name="timestamp",
    )

    frames = list(
        generate_frames(
            contract,
            episode_data,
            reference_timestamps,
            task=resolved_task_label,
            defer_image_decode=True,
        )
    )
    return WorkerEpisode(
        event_idx=task.event_idx,
        task_label=resolved_task_label,
        frames=frames,
    )


class WorkerEpisode(NamedTuple):
    """Return value of :func:`_worker_build_episode`.

    Lives outside the function so tests can construct it directly when
    monkeypatching the worker call.
    """

    event_idx: int
    task_label: str
    frames: list[dict[str, Any]]


def topic_descriptors_from_topics(
    topics_dict: dict[str, list[Any]],
) -> dict[str, list[TopicDescriptor]]:
    """Project a ``{name: [Topic, ...]}`` mapping to picklable descriptors.

    Pulled out of the orchestrator so the test for ``_worker_init`` can
    feed it a parallel structure of stubs and check the reconstruction
    behaviour without going through Roboto.
    """
    out: dict[str, list[TopicDescriptor]] = {}
    for name, topic_list in topics_dict.items():
        out[name] = [
            TopicDescriptor(
                topic_id=t.topic_id,
                file_id=getattr(t, "file_id", None),
                topic_name=getattr(t, "name", name),
                start_time_ns=getattr(t, "start_time", None),
                end_time_ns=getattr(t, "end_time", None),
                message_count=getattr(t, "message_count", None),
            )
            for t in topic_list
        ]
    return out
