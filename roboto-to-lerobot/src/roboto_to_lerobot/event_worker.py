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
2. ``_worker_init`` builds ``roboto.Topic`` objects from the topic records
   the main process already holds, shipped as ``TopicDescriptor`` payloads,
   so a worker makes no API call to start.
3. ``_worker_build_episode`` runs the full ``DataCollection`` +
   ``generate_frames`` pipeline with ``defer_image_decode=True``, so video
   keys travel back across IPC as ``_DeferredFrame`` sentinels carrying raw
   encoded bytes, or for compressed video a reference into the episode's
   packets, rather than decoded HWC RGB arrays. The main process runs
   ``materialize_deferred`` before each ``writer.add_frame`` call.

Pickling raw bytes is far cheaper than pickling decoded numpy arrays, which
matters because every video frame crosses the IPC boundary. Tabular numpy
arrays (observations / actions) are small enough to pickle as-is.

Given a spool directory, a worker writes each episode's camera payloads to a
spool file of its own there as they are fetched (see the ``spool`` module),
and the frames carry references into that file instead of the bytes. The
episode then pickles to a size that does not depend on its camera payload.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import roboto

from .contract_utils import (
    Contract,
    DataCollection,
    fps_to_time_step_ns,
    resolve_task_label,
)
from .lerobot import generate_frames
from .spool import SpoolWriter


class TopicDescriptor(NamedTuple):
    """Picklable form of one ``roboto.Topic``: its record as JSON-shaped data.

    The main process already holds the full record of every contract topic,
    listed while matching the contract against the dataset's files. Shipping
    those records lets each worker build the same ``Topic`` objects without
    an API call. Fetching each topic again would cost one call per topic per
    worker, which adds up on recordings chunked across many files.
    """

    record: dict[str, Any]


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
    spool_dir: str | None = None,
) -> None:
    """Build per-worker ``Topic`` objects and stash them on the module.

    Makes no API call: each ``Topic`` is built from the record its
    descriptor carries, against this worker's own ``RobotoClient``, which
    later fetches the topic's data. ``spool_dir``, an existing directory,
    is where each episode's spool file goes; without it, camera payloads
    stay in memory.
    """
    client = roboto.RobotoClient.defaulted()
    _WORKER_STATE["topics"] = {
        ds_id: {
            name: [
                roboto.Topic(roboto.TopicRecord.model_validate(d.record), roboto_client=client)
                for d in descriptors
            ]
            for name, descriptors in by_name.items()
        }
        for ds_id, by_name in descriptors_by_ds.items()
    }
    _WORKER_STATE["contracts"] = contracts_by_ds
    _WORKER_STATE["spool_dir"] = spool_dir


def _worker_build_episode(
    task: EventTask,
) -> WorkerEpisode:
    """Build one episode in the worker.

    Returns the event index (for in-order drain on the main process), the
    resolved task label, the list of frame dicts, and the reference-timeline
    positions of the frames left out of that list. Video entries in each
    frame carry ``_DeferredFrame`` sentinels — the main process decodes them
    just before handing each frame to ``writer.add_frame``.

    When :func:`_worker_init` was given a spool directory, the episode's
    camera payloads are in the spool file named by
    ``WorkerEpisode.spool_path``. The file is closed when this returns, and
    the caller deletes it once the episode is drained. If building the
    episode fails, the file is deleted here before the exception propagates.
    """
    spool_dir = _WORKER_STATE.get("spool_dir")
    if not spool_dir:
        return _build_episode(task, None)
    spool = SpoolWriter(spool_dir, prefix=f"event-{task.event_idx}-")
    try:
        episode = _build_episode(task, spool)
    except BaseException:
        spool.discard()
        raise
    spool.close()
    return episode


def _build_episode(task: EventTask, spool: SpoolWriter | None) -> WorkerEpisode:
    """Build ``task``'s episode, writing its camera payloads to ``spool`` if given."""
    contract = _WORKER_STATE["contracts"][task.src_ds_id]
    topics = _WORKER_STATE["topics"][task.src_ds_id]

    episode_data = DataCollection(
        contract=contract,
        topics=topics,
        start_time_ns=task.start_time_ns - task.buffer_ns,
        end_time_ns=task.end_time_ns + task.buffer_ns + task.action_lead_ns,
        spool=spool,
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

    dropped_frames: list[int] = []
    frames = list(
        generate_frames(
            contract,
            episode_data,
            reference_timestamps,
            task=resolved_task_label,
            defer_image_decode=True,
            dropped_frames=dropped_frames,
        )
    )
    return WorkerEpisode(
        event_idx=task.event_idx,
        task_label=resolved_task_label,
        frames=frames,
        dropped_frames=tuple(dropped_frames),
        spool_path=spool.path if spool is not None else None,
    )


class WorkerEpisode(NamedTuple):
    """Return value of :func:`_worker_build_episode`.

    Lives outside the function so tests can construct it directly when
    monkeypatching the worker call.

    ``dropped_frames`` are the positions, on the episode's reference
    timeline, of the frames left out of ``frames`` because a stream's value
    for them is missing; ``frames`` holds the remaining positions in order.

    ``spool_path`` is the spool file that holds the camera payloads
    ``frames`` refer to, or ``None`` when the episode was built without a
    spool. Whoever drains the episode deletes the file.
    """

    event_idx: int
    task_label: str
    frames: list[dict[str, Any]]
    dropped_frames: tuple[int, ...] = ()
    spool_path: str | None = None


def topic_descriptors_from_topics(
    topics_dict: dict[str, list[roboto.Topic]],
) -> dict[str, list[TopicDescriptor]]:
    """Project a ``{name: [Topic, ...]}`` mapping to picklable descriptors."""
    return {
        name: [TopicDescriptor(record=t.record.model_dump(mode="json")) for t in topic_list]
        for name, topic_list in topics_dict.items()
    }
