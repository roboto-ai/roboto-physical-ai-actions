from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import shutil
import traceback
from collections import Counter, defaultdict
from concurrent.futures import Future, ProcessPoolExecutor
from datetime import UTC, datetime
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import roboto
from roboto.updates import MetadataChangeset

from .contract_utils import (
    _TOPIC_FETCH_THREADS,
    Contract,
    DataCollection,
    _compute_buffer_ns,
    collect_topics_from_dataset,
    fps_to_time_step_ns,
    load_contract,
    resolve_role_bindings,
    resolve_task_label,
)
from .event_worker import (
    EventTask,
    WorkerEpisode,
    _worker_build_episode,
    _worker_init,
    topic_descriptors_from_topics,
)
from .lerobot import generate_frames, materialize_deferred
from .logger import logger
from .stderr_filter import (
    disable_progress_bars,
    install_svt_stderr_filter,
    set_libav_log_level,
)
from .writers import LeRobotWriter

# Writer-encoder knobs forwarded to ``LeRobotDataset.create``. The v2_1
# adapter accepts the same kwargs and silently drops 0.5.x-only ones, so
# callers do not need to branch on lerobot version. ``vcodec`` is hard-coded
# inside the v3_0 adapter — h264 worsened downstream training.
#
# ``_STREAMING_ENCODING=True`` pipes pixels straight into ffmpeg instead of
# writing PNGs to disk for ffmpeg to read back, saving wall time per run.
# Trade-off: lerobot 0.5.x routes video stats through the streaming
# encoder's ``finish_episode()`` rather than ``compute_episode_stats``, so
# the two aggregators disagree on:
#   * ``count`` — pixel-counted, not frame-counted. Welford-merged stats
#     will silently corrupt if this dataset is concatenated with one
#     captured under the non-streaming path. Single-source training is
#     unaffected.
#   * ``mean``/``std``/quantiles — small drift because the reductions
#     differ.
#   * ``meta/episodes.parquet`` — image-stat columns appear at the tail
#     (patched in after the non-video pass returns).
# Encoded mp4 frames stay pixel-identical regardless. Flip to ``False`` to
# re-incur the PNG round-trip if a non-streaming-compatible stats path is
# required.
_IMAGE_WRITER_THREADS = 8
# lerobot 0.5.x's ``DatasetWriter.save_episode`` runs both the streaming
# branch and the batched-flush branch when both knobs are on; the batched
# flush then crashes because ``EpisodesMeta.save_episode`` never refreshes
# the in-memory cache that the empty-dataset create path initialises to
# ``None``. Even absent the crash, the batched flush would redundantly
# re-encode every Nth episode from scratch. ``batch_encoding_size=1``
# disables the batched branch so only the streaming branch runs.
_BATCH_ENCODING_SIZE = 1
_STREAMING_ENCODING = True
# SVT-AV1 level-of-parallelism per camera stream. ``None`` selects SVT
# auto-LP, which has allowed the bounded encoder queue to overflow under
# bursty ``add_frame`` pressure and silently drop frames. Pin to >= 2;
# ``encoder_threads`` is exposed as an action parameter for tuning.
_ENCODER_THREADS: int | None = 2
# Bounded encoder queue inside lerobot's writer. Larger values absorb
# bursts at the episode boundary; the cost is resident memory roughly
# equal to maxsize × frame_size × camera_count.
_ENCODER_QUEUE_MAXSIZE = 300

# Process-pool knobs. The ``pool_size`` action parameter is the user-facing
# knob; ``ROBOTO_TO_LEROBOT_POOL_SIZE`` remains as an env-var override for
# external benchmark sweeps. The auto-detected default is capped at
# ``_POOL_SIZE_CAP`` because each worker spawns its own ThreadPoolExecutor up
# to ``_TOPIC_FETCH_THREADS`` sockets, and the product bounds per-host fds.
# Explicit values from either the parameter or env var bypass the cap.
_POOL_SIZE_ENV = "ROBOTO_TO_LEROBOT_POOL_SIZE"
_POOL_SIZE_CAP = 16
_ENCODER_THREADS_ENV = "ROBOTO_TO_LEROBOT_ENCODER_THREADS"
# Sharded-writer knobs. ``shard_count > 1`` spawns K sibling shard processes
# that each write an independent LeRobotDataset to a temp dir; the main
# process then merges them with ``lerobot.datasets.aggregate.aggregate_datasets``
# (stream-copy concat, no re-encode). ``shard_count × encoder_threads``
# should not exceed host vCPUs or the writers contend on the encoder pool;
# the run warns rather than clamps so operators can override deliberately.
# ``ROBOTO_TO_LEROBOT_SHARD_COUNT`` mirrors the pool-size env-var escape
# hatch for external benchmark sweeps.
_SHARD_COUNT_ENV = "ROBOTO_TO_LEROBOT_SHARD_COUNT"
# Headroom on the in-flight submission window so the main thread can drain
# the head-of-line future while workers continue churning. Resident memory
# peaks around (pool_size + this) × episode_payload_size.
_POOL_IN_FLIGHT_HEADROOM = 4


def _parse_pool_size(raw: object, *, source: str) -> int:
    try:
        pool_size = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError) as e:
        raise ValueError(f"{source}={raw!r} is not an integer") from e
    if pool_size < 1:
        raise ValueError(f"{source}={raw!r} must be >= 1")
    return pool_size


def _resolve_encoder_threads(param_value: object = None) -> int:
    """Resolve SVT-AV1's level-of-parallelism (per stream).

    Precedence: explicit ``encoder_threads`` action parameter > the
    ``ROBOTO_TO_LEROBOT_ENCODER_THREADS`` env var > the compiled-in
    default (``_ENCODER_THREADS``). ``None`` and empty string are treated
    as unset at every layer. Bad values raise so a typo does not silently
    fall back to SVT auto-LP, which has dropped frames at high pool sizes.

    The parameter path only accepts a positive integer; opt in to SVT
    auto-LP by changing ``_ENCODER_THREADS`` to ``None`` in source.
    """
    if param_value is not None and param_value != "":
        return _parse_pool_size(param_value, source="encoder_threads parameter")
    raw = os.environ.get(_ENCODER_THREADS_ENV)
    if raw is None or raw == "":
        if _ENCODER_THREADS is None:
            raise ValueError(
                "_ENCODER_THREADS is None (SVT auto-LP) and no override was "
                "passed; this configuration has dropped frames at high pool "
                "sizes — set encoder_threads explicitly."
            )
        return _ENCODER_THREADS
    return _parse_pool_size(raw, source=_ENCODER_THREADS_ENV)


def _resolve_pool_size(param_value: object = None, shard_count: int = 1) -> int:
    """Resolve the per-shard worker count for the process pool.

    Precedence: explicit ``pool_size`` action parameter > the
    ``ROBOTO_TO_LEROBOT_POOL_SIZE`` env var > ``min(_POOL_SIZE_CAP,
    (os.cpu_count() or 4) // max(1, shard_count))``. ``None`` and empty
    string are treated as unset at every layer so external runners can
    unconditionally forward an empty env var and missing optional parameters
    fall through cleanly. Bad values (non-int, ``<1``) raise so a typo does
    not silently fall back to a serial-equivalent run.

    With ``shard_count > 1`` the semantics shift to "workers per shard" — K
    shards each spin up their own pool, so the auto-default scales the vCPU
    budget down by the shard count to keep total worker processes near
    vCPU count. Explicit values from the parameter or env var still pass
    through unchanged; the operator may want to over-subscribe deliberately.
    """
    if param_value is not None and param_value != "":
        return _parse_pool_size(param_value, source="pool_size parameter")
    raw = os.environ.get(_POOL_SIZE_ENV)
    if raw is None or raw == "":
        per_shard_budget = max(1, (os.cpu_count() or 4) // max(1, shard_count))
        return min(_POOL_SIZE_CAP, per_shard_budget)
    return _parse_pool_size(raw, source=_POOL_SIZE_ENV)


def _resolve_shard_count(param_value: object = None) -> int:
    """Resolve the number of writer shards (default 1 = legacy path).

    Precedence mirrors :func:`_resolve_pool_size`: explicit ``shard_count``
    action parameter > the ``ROBOTO_TO_LEROBOT_SHARD_COUNT`` env var > ``1``.
    ``None`` and empty string are treated as unset. Bad values raise so a
    typo does not silently regress to single-writer.
    """
    if param_value is not None and param_value != "":
        return _parse_pool_size(param_value, source="shard_count parameter")
    raw = os.environ.get(_SHARD_COUNT_ENV)
    if raw is None or raw == "":
        return 1
    return _parse_pool_size(raw, source=_SHARD_COUNT_ENV)


def _load_contract_from_context(
    context: roboto.InvocationContext,
) -> tuple[Contract, str, str]:
    """Load the contract YAML from the invocation dataset.

    Returns (contract, contract_relative_path, contract_sha256) so every
    output file can be traced back to the exact contract bytes.
    """
    contract_param = context.get_optional_parameter("contract")
    if contract_param is not None:
        logger.info("Contract file specified: %s", contract_param)
        contract_file = context.dataset.get_file_by_path(contract_param)
    else:
        yamls = context.dataset.list_files(include_patterns=["contract.yaml"])
        found_yaml = next(yamls, None)
        if found_yaml is None:
            raise ValueError("No contract file found in dataset")
        if next(yamls, None) is not None:
            logger.warning(
                "Multiple contract files found in dataset. Using %s",
                found_yaml.relative_path,
            )
        contract_file = context.dataset.get_file_by_path(found_yaml.relative_path)

    contract_path = context.input_dir / contract_file.relative_path
    contract_file.download(contract_path)
    contract = load_contract(contract_path)
    contract_sha256 = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    logger.info(
        "Loaded contract: %s (version %d) from %s (sha256=%s)",
        contract.name, contract.version, contract_file.relative_path, contract_sha256,
    )
    return contract, contract_file.relative_path, contract_sha256


def _stamp_output_files(
    context: roboto.InvocationContext,
    subfolder: Path,
    stamp: dict[str, str],
) -> None:
    """Annotate every file under ``subfolder`` with provenance metadata.

    Only wired up by the hosted Roboto runtime; ``invoke-local`` does not
    create the file-metadata changeset file. We silently skip in that case
    so local runs can still produce the LeRobot dataset and manifest.
    """
    try:
        changeset_manager = context.file_changeset_manager
    except ValueError as e:
        logger.warning(
            "Skipping per-file metadata stamping (local invocation): %s", e,
        )
        return

    for path in subfolder.rglob("*"):
        if not path.is_file():
            continue
        relative_path = str(path.relative_to(context.output_dir))
        changeset_manager.put_fields(relative_path, stamp)


class _DrainAccum:
    """Mutable counters shared between the drain loop and its caller.

    Lets :func:`_drain_one_episode` and :func:`_run_pool_drain` stay pure
    functions instead of inner closures over ``main``'s locals so the same
    drain code runs inside a shard subprocess. Plain attribute access (no
    dataclass) keeps the object picklable across spawn.
    """

    def __init__(self) -> None:
        self.total_frames: int = 0
        self.total_episodes: int = 0
        self.per_dataset_counts: dict[str, int] = defaultdict(int)
        self.episode_to_event: list[dict[str, object]] = []
        # Events the conversion soft-dropped. Each entry is a plain dict so
        # the accumulator survives the spawn pickle into a shard subprocess
        # unchanged; see _record_skip for the entry shape.
        self.skipped_events: list[dict[str, object]] = []


def _build_skipped_summary(skipped_events: list[dict]) -> dict:
    """Rollup of per-event skip records — shape mirrors ``dedup_summary``.

    Pulled out so the manifest assembly stays declarative and the rollup
    is unit-testable without instantiating an ``InvocationContext``.
    """
    return {
        "count": len(skipped_events),
        "by_stage": dict(Counter(s["stage"] for s in skipped_events)),
        "by_error_class": dict(Counter(s["error_class"] for s in skipped_events)),
        "by_source_dataset": dict(Counter(
            s["source_dataset_id"] for s in skipped_events
        )),
        "event_ids": sorted({
            str(s["event_id"]) for s in skipped_events
            if s["event_id"] is not None
        }),
    }


def _summarize_skip_reasons(skipped_events: list[dict]) -> str:
    """One-line, human-readable rollup of drop reasons for log/error text.

    Thin wrapper over :func:`_build_skipped_summary` — that function shapes
    the rollup for the JSON manifest; this one flattens it to a string so
    callers that need the reasons inline in a log message or an exception
    (e.g. :func:`_drop_empty_shards`) don't each reinvent the formatting.
    """
    if not skipped_events:
        return "no drop reasons recorded"
    summary = _build_skipped_summary(skipped_events)
    return (
        f"{summary['count']} event(s) dropped "
        f"(by_stage={summary['by_stage']}, by_error_class={summary['by_error_class']}, "
        f"event_ids={summary['event_ids']})"
    )


def _safe_discard_episode(writer) -> None:
    """Discard the writer's in-progress episode buffer, swallowing any
    failure. Soft-drop must be best-effort — re-raising here would convert
    a recoverable per-event failure back into a run-killing one.
    """
    try:
        writer.discard_episode()
    except Exception:
        logger.exception("discard_episode failed after soft-drop")


def _shard_tag(shard_idx: int | None) -> str:
    """Format the bracketed shard prefix shared by every episode log line.

    ``-`` is used in the single-writer path (no shards spawned) so the
    column stays present and greppable across both code paths.
    """
    return f"shard={shard_idx}" if shard_idx is not None else "shard=-"


def _record_skip(
    *,
    accumulator: _DrainAccum,
    meta: dict,
    exc: BaseException,
    stage: Literal["worker", "drain", "preload"],
    shard_idx: int | None = None,
) -> None:
    """Append a soft-drop entry for one failed event and log a WARNING.

    Must be called from inside an ``except`` block — ``traceback.format_exc()``
    reads ``sys.exc_info()`` and would otherwise return ``"NoneType: None\n"``.

    ``stage`` ∈ ``{"worker", "drain", "preload"}`` records which surface
    raised, so post-hoc analysis can tell apart episode-build failures (the
    worker process) from writer-side failures (frame iteration / save_episode)
    and the inline preload path.
    """
    event_id = meta.get("event_id")
    event_idx = meta.get("event_idx")
    src_ds_id = meta.get("src_ds_id")
    error_class = exc.__class__.__name__
    error_message = str(exc)
    accumulator.skipped_events.append({
        "event_id": event_id,
        "source_dataset_id": src_ds_id,
        "start_time_ns": int(meta.get("start_time_ns", 0)),
        "end_time_ns": int(meta.get("end_time_ns", 0)),
        "task_label": meta.get("task_label"),
        "stage": stage,
        "error_class": error_class,
        "error_message": error_message,
        "traceback": traceback.format_exc(),
    })
    logger.warning(
        "[%s] Dropping event event_idx=%s (event_id=%s, dataset=%s, stage=%s): %s: %s",
        _shard_tag(shard_idx), event_idx, event_id, src_ds_id, stage,
        error_class, error_message,
    )


def _drain_one_episode(
    *,
    writer,
    video_specs_by_key: dict,
    src_ds_id: str,
    event_id: object,
    start_time_ns: int,
    end_time_ns: int,
    episode,
    accumulator: _DrainAccum,
    shard_idx: int | None = None,
) -> None:
    """Push one episode through the writer.

    Module-level so the shard subprocess can call the same code path the
    single-writer drain uses.
    """
    shard_tag = _shard_tag(shard_idx)
    logger.info(
        "[%s] Processing episode %s (event_id=%s, dataset=%s, task=%r)",
        shard_tag, episode.event_idx, event_id, src_ds_id, episode.task_label,
    )
    frame_count = 0
    for frame in episode.frames:
        materialize_deferred(frame, video_specs_by_key)
        writer.add_frame(frame, task=episode.task_label)
        frame_count += 1
    writer.save_episode()
    accumulator.episode_to_event.append({
        "episode_index": accumulator.total_episodes,
        "event_id": event_id,
        "source_dataset_id": src_ds_id,
        "start_time_ns": int(start_time_ns),
        "end_time_ns": int(end_time_ns),
        "n_frames": frame_count,
        "task": episode.task_label,
    })
    accumulator.total_frames += frame_count
    accumulator.total_episodes += 1
    accumulator.per_dataset_counts[src_ds_id] += 1
    logger.info(
        "[%s] Saved episode %s (event_id=%s, frames=%d, writer_episode=%d)",
        shard_tag, episode.event_idx, event_id, frame_count, accumulator.total_episodes,
    )


def _run_pool_drain(
    *,
    writer,
    video_specs_by_key: dict,
    pool_tasks: list,
    drain_order: list[dict],
    descriptors_by_ds: dict,
    contracts_by_ds: dict,
    pool_size: int,
    accumulator: _DrainAccum,
    preloaded_event_idx: int | None = None,
    preloaded_episode=None,
    shard_idx: int | None = None,
) -> None:
    """Drive a ``ProcessPoolExecutor`` and drain ``drain_order`` in order.

    ``drain_order`` is the chronological per-event metadata for everything
    the writer will consume (both preloaded and pool-submitted). The pool
    is a top-up window: we submit ``pool_size + _POOL_IN_FLIGHT_HEADROOM``
    futures ahead, then block on each event_idx's specific future so the
    writer sees episodes in chronological order regardless of worker
    completion order.

    If ``drain_order`` has a single entry that matches ``preloaded_event_idx``
    and ``pool_tasks`` is empty, no pool is spun up at all (this preserves
    the single-event optimisation from the pre-shard code).
    """
    only_preloaded = (
        preloaded_episode is not None
        and not pool_tasks
        and len(drain_order) == 1
        and drain_order[0]["event_idx"] == preloaded_event_idx
    )
    if only_preloaded:
        meta = drain_order[0]
        try:
            _drain_one_episode(
                writer=writer,
                video_specs_by_key=video_specs_by_key,
                src_ds_id=meta["src_ds_id"],
                event_id=meta["event_id"],
                start_time_ns=meta["start_time_ns"],
                end_time_ns=meta["end_time_ns"],
                episode=preloaded_episode,
                accumulator=accumulator,
                shard_idx=shard_idx,
            )
        except Exception as exc:
            # Writer may have buffered frames before the failure — discard
            # so a subsequent episode can't inherit them. Symmetric with the
            # loop body even though no follow-on episode exists here.
            _record_skip(
                accumulator=accumulator, meta=meta, exc=exc, stage="drain",
                shard_idx=shard_idx,
            )
            _safe_discard_episode(writer)
        return

    # ``mp_context="spawn"`` not ``fork``: half-initialised lock state and a
    # shared ffmpeg pipe fd are bear-traps when fork-cloning a process that
    # already created a writer. Spawn re-imports the module from scratch in
    # each worker so ``_worker_init`` can re-read client config cleanly;
    # per-worker startup cost is paid once at pool warm-up.
    ctx = mp.get_context("spawn")
    max_in_flight = pool_size + _POOL_IN_FLIGHT_HEADROOM
    pool = ProcessPoolExecutor(
        max_workers=pool_size,
        mp_context=ctx,
        initializer=_worker_init,
        initargs=(descriptors_by_ds, contracts_by_ds),
    )
    submitted: dict[int, Future] = {}
    next_to_submit = 0
    try:
        for meta in drain_order:
            event_idx = meta["event_idx"]
            if (
                preloaded_event_idx is not None
                and event_idx == preloaded_event_idx
            ):
                try:
                    _drain_one_episode(
                        writer=writer,
                        video_specs_by_key=video_specs_by_key,
                        src_ds_id=meta["src_ds_id"],
                        event_id=meta["event_id"],
                        start_time_ns=meta["start_time_ns"],
                        end_time_ns=meta["end_time_ns"],
                        episode=preloaded_episode,
                        accumulator=accumulator,
                        shard_idx=shard_idx,
                    )
                except Exception as exc:
                    _record_skip(
                        accumulator=accumulator, meta=meta, exc=exc, stage="drain",
                        shard_idx=shard_idx,
                    )
                    _safe_discard_episode(writer)
                continue
            while (
                next_to_submit < len(pool_tasks)
                and len(submitted) < max_in_flight
            ):
                t = pool_tasks[next_to_submit]
                submitted[t.event_idx] = pool.submit(
                    _worker_build_episode, t,
                )
                next_to_submit += 1
            # Two failure surfaces, two except blocks: a worker-side raise
            # surfaces here at ``.result()`` and leaves the writer clean (no
            # discard needed); a drain-side raise happens inside
            # ``_drain_one_episode`` after frames have already been pushed
            # into the writer's buffer, so a ``discard_episode`` is required
            # before the next event can be saved.
            try:
                episode = submitted.pop(event_idx).result()
            except Exception as exc:
                _record_skip(
                    accumulator=accumulator, meta=meta, exc=exc, stage="worker",
                    shard_idx=shard_idx,
                )
                continue
            try:
                _drain_one_episode(
                    writer=writer,
                    video_specs_by_key=video_specs_by_key,
                    src_ds_id=meta["src_ds_id"],
                    event_id=meta["event_id"],
                    start_time_ns=meta["start_time_ns"],
                    end_time_ns=meta["end_time_ns"],
                    episode=episode,
                    accumulator=accumulator,
                    shard_idx=shard_idx,
                )
            except Exception as exc:
                _record_skip(
                    accumulator=accumulator, meta=meta, exc=exc, stage="drain",
                    shard_idx=shard_idx,
                )
                _safe_discard_episode(writer)
    except BaseException:
        # ``cancel_futures=True`` drops pending tasks immediately rather
        # than waiting for in-flight ones; ``wait=False`` lets the process
        # exit without joining workers that may still be holding large
        # episode payloads. The default ``shutdown()`` would block the
        # abort path for minutes at high pool sizes.
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)


def _partition_shards(
    matched: list,
    matched_events: list,
    *,
    shard_count: int,
    buffer_ns: int,
    action_lead_ns: int,
) -> tuple[list[list[dict]], list[list]]:
    """Frame-balanced LPT bin-pack with staggered-heavy within-shard rotation.

    Combines two properties that simpler partitioners satisfy only one of:

    1. **Frame balance across shards** — events are sorted heaviest-first
       and LPT-assigned to whichever shard currently holds the smallest
       running frame total. This keeps long-pole shard work within a few
       percent of the smallest, where contiguous-chunking can leave a
       skew of well over 1.5×.

    2. **Staggered heavy-event timing across shards** — within each shard
       the events are sorted heavy-first and then rotated by
       ``k * L // K`` so shard ``k`` starts drainage ``k/K`` of the way
       around its sorted ring. Shard 0 leads with its heaviest event;
       shard ``K-1`` reaches its heaviest near the end of its drain.
       Peak-bandwidth decode windows are spread along wall time rather
       than synchronising at ``t=0`` (which OOMs at high K) or at
       ``t=end`` (same failure mode).

    Side effect: the merged dataset's episode order is no longer
    chronological by ``start_time``. The manifest still carries
    ``start_time_ns``/``end_time_ns`` per episode for any consumer that
    needs wall-clock order.

    Determinism: LPT and sort both tiebreak on the original slot index,
    so identical inputs produce identical partitions across runs.

    Edge cases: ``shard_count > len(events)`` clamps to one event per
    shard and drops the rest; empty ``matched_events`` returns empty
    chunk lists.
    """
    n = len(matched_events)
    if n == 0:
        return [], []
    effective_k = max(1, min(shard_count, n))

    # Step 1 — LPT bin-pack: assign events (heaviest first) to whichever
    # shard currently has the smallest frame total. Tiebreak by lowest
    # shard index so output is deterministic across runs.
    slots_by_frames_desc = sorted(
        range(n), key=lambda s: (-matched[s][3], s),
    )
    shard_slots: list[list[int]] = [[] for _ in range(effective_k)]
    shard_totals: list[int] = [0] * effective_k
    for slot in slots_by_frames_desc:
        target = min(
            range(effective_k),
            key=lambda i: (shard_totals[i], i),
        )
        shard_slots[target].append(slot)
        shard_totals[target] += matched[slot][3]

    # Step 2 — within each shard, sort heavy-first then rotate so shard k's
    # single heaviest event lands at relative drain position
    # ``target_pos = k * L // K``. Shard 0 leads with its heaviest
    # (target_pos=0); shard K-1 reaches its heaviest near the end of its
    # drain (target_pos ≈ L-L/K). The peak-bandwidth windows are spread
    # along wall time instead of collapsing onto a single instant across
    # shards. The left-rotation amount that achieves this is
    # ``offset = (L - target_pos) mod L``: ``sorted_desc[0]`` (the
    # heaviest) ends up at position ``L - offset`` in ``ordered``.
    # Integer division is intentional: small shards (L < K) collapse
    # target_pos to 0 and the heaviest simply leads — the synchronisation
    # problem only bites when several shards each hold a comparably-heavy
    # event.
    chunks_drain: list[list[dict]] = []
    chunks_tasks: list[list] = []
    for k in range(effective_k):
        slots = shard_slots[k]
        if not slots:
            continue
        sorted_desc = sorted(slots, key=lambda s: (-matched[s][3], s))
        L = len(sorted_desc)
        target_pos = (k * L) // effective_k
        offset = (L - target_pos) % L
        ordered = sorted_desc[offset:] + sorted_desc[:offset]
        drain: list[dict] = []
        tasks: list = []
        for slot in ordered:
            event_idx, src_ds_id, task_label, num_frames = matched[slot]
            ev = matched_events[slot]
            drain.append({
                "event_idx": event_idx,
                "src_ds_id": src_ds_id,
                "task_label": task_label,
                "event_id": getattr(ev, "event_id", None),
                "start_time_ns": int(ev.start_time),
                "end_time_ns": int(ev.end_time),
            })
            tasks.append(EventTask(
                event_idx=event_idx,
                event_id=getattr(ev, "event_id", None),
                src_ds_id=src_ds_id,
                start_time_ns=int(ev.start_time),
                end_time_ns=int(ev.end_time),
                task_label=task_label,
                num_frames=num_frames,
                buffer_ns=buffer_ns,
                action_lead_ns=action_lead_ns,
            ))
        chunks_drain.append(drain)
        chunks_tasks.append(tasks)
    return chunks_drain, chunks_tasks


def _run_shard_subprocess(
    *,
    shard_idx: int,
    log_level: int,
    shard_root: str,
    repo_id: str,
    fps: int,
    features: dict,
    robot_type: str,
    encoder_threads: int,
    pool_size: int,
    video_specs_by_key: dict,
    descriptors_by_ds: dict,
    contracts_by_ds: dict,
    pool_tasks: list,
    drain_order: list,
    result_path: str,
) -> None:
    """Entrypoint for a single shard process.

    Spawned by ``main`` with ``mp_context="spawn"`` when ``shard_count > 1``;
    builds an independent ``LeRobotWriter`` rooted at ``shard_root``, drains
    its slice of events through the same ``_run_pool_drain`` loop the
    single-writer path uses, then writes a JSON summary to ``result_path``.
    The main process reads the summary back to assemble the manifest and
    drives :func:`lerobot.datasets.aggregate.aggregate_datasets` to merge
    every shard into the final destination.

    Path/repo arguments are strings rather than ``Path`` objects to keep
    the kwargs cheaply picklable across spawn.
    """
    # Spawn re-imports the module from scratch, so the libav C-level
    # filter and tqdm/datasets disable state set in main() do not carry
    # over. Re-apply them here so the shard's encoder threads and any
    # tqdm-wrapped lerobot helpers stay quiet. The SVT-AV1 stderr filter
    # is fd-inherited from the parent, so no re-install is needed.
    set_libav_log_level()
    disable_progress_bars()
    logger.setLevel(log_level)
    shard_root_p = Path(shard_root)
    result_path_p = Path(result_path)
    result: dict[str, Any] = {
        "shard_idx": shard_idx,
        "shard_root": str(shard_root_p),
        "repo_id": repo_id,
        "status": "started",
    }
    try:
        logger.info(
            "Shard %d: writer root=%s, events=%d, pool_size=%d, encoder_threads=%d",
            shard_idx, shard_root_p, len(drain_order), pool_size, encoder_threads,
        )
        writer = LeRobotWriter.create(
            repo_id=repo_id,
            fps=fps,
            features=features,
            root=shard_root_p,
            robot_type=robot_type,
            image_writer_threads=_IMAGE_WRITER_THREADS,
            batch_encoding_size=_BATCH_ENCODING_SIZE,
            streaming_encoding=_STREAMING_ENCODING,
            encoder_threads=encoder_threads,
            encoder_queue_maxsize=_ENCODER_QUEUE_MAXSIZE,
        )
        accumulator = _DrainAccum()
        _run_pool_drain(
            writer=writer,
            video_specs_by_key=video_specs_by_key,
            pool_tasks=pool_tasks,
            drain_order=drain_order,
            descriptors_by_ds=descriptors_by_ds,
            contracts_by_ds=contracts_by_ds,
            pool_size=pool_size,
            accumulator=accumulator,
            shard_idx=shard_idx,
        )
        writer.finalize()
        result.update({
            "status": "ok",
            "total_episodes": accumulator.total_episodes,
            "total_frames": accumulator.total_frames,
            "per_dataset_counts": dict(accumulator.per_dataset_counts),
            "episode_to_event": accumulator.episode_to_event,
            "skipped_events": list(accumulator.skipped_events),
        })
    except BaseException as exc:
        # Capture traceback so the parent can surface it instead of opaque
        # exit-code messaging when a shard dies inside lerobot.
        result.update({
            "status": "error",
            "error_class": exc.__class__.__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        })
        logger.exception("Shard %d failed: %s", shard_idx, exc)
        raise
    finally:
        result_path_p.write_text(json.dumps(result, default=str))


def _spawn_shards(
    *,
    chunks_drain: list[list[dict]],
    chunks_tasks: list[list],
    shard_workdir: Path,
    fps: int,
    features: dict,
    robot_type: str,
    encoder_threads: int,
    pool_size: int,
    video_specs_by_key: dict,
    descriptors_by_ds: dict,
    contracts_by_ds: dict,
    log_level: int,
) -> list[dict]:
    """Spawn one process per shard, wait for all, return per-shard results.

    Failures (non-zero exit, missing result file, ``status != "ok"``) are
    collected and surfaced as a single ``RuntimeError`` so the operator
    sees every shard's traceback in one place instead of chasing
    stale-pid output. The result list is sorted by ``shard_idx`` so
    downstream aggregation runs in original chronological order.
    """
    shard_count = len(chunks_drain)
    ctx = mp.get_context("spawn")
    processes: list[mp.Process] = []
    result_paths: list[Path] = []
    shard_roots: list[Path] = []
    shard_workdir.mkdir(parents=True, exist_ok=True)
    for i in range(shard_count):
        shard_root = shard_workdir / f"shard-{i:03d}"
        result_path = shard_workdir / f"shard-{i:03d}.result.json"
        shard_roots.append(shard_root)
        result_paths.append(result_path)
        p = ctx.Process(
            target=_run_shard_subprocess,
            kwargs={
                "shard_idx": i,
                "log_level": log_level,
                "shard_root": str(shard_root),
                "repo_id": shard_root.name,
                "fps": fps,
                "features": features,
                "robot_type": robot_type,
                "encoder_threads": encoder_threads,
                "pool_size": pool_size,
                "video_specs_by_key": video_specs_by_key,
                "descriptors_by_ds": descriptors_by_ds,
                "contracts_by_ds": contracts_by_ds,
                "pool_tasks": chunks_tasks[i],
                "drain_order": chunks_drain[i],
                "result_path": str(result_path),
            },
            name=f"shard-{i:03d}",
        )
        p.start()
        processes.append(p)

    # Join every shard before raising — partial joins leak orphaned shard
    # children, and we want a clean shutdown either way.
    failures: list[str] = []
    for i, p in enumerate(processes):
        p.join()
        if p.exitcode != 0:
            failures.append(f"shard-{i:03d} exit={p.exitcode}")

    results: list[dict] = []
    for i, rp in enumerate(result_paths):
        if not rp.exists():
            failures.append(f"shard-{i:03d} missing result file {rp}")
            continue
        results.append(json.loads(rp.read_text()))

    bad = [r for r in results if r.get("status") != "ok"]
    if failures or bad:
        for r in bad:
            logger.error(
                "Shard %s failed (%s): %s\n%s",
                r.get("shard_idx"),
                r.get("error_class", "?"),
                r.get("error_message"),
                r.get("traceback", ""),
            )
        raise RuntimeError(
            f"Shards failed: process_failures={failures} "
            f"bad_results={[r.get('shard_idx') for r in bad]}"
        )

    results.sort(key=lambda r: r["shard_idx"])
    for r in results:
        r["root"] = str(shard_roots[r["shard_idx"]])
    return results


def _shard_path_supported() -> bool:
    """Whether the embedded lerobot exposes the sharded path's merge API.

    ``lerobot.datasets.aggregate.aggregate_datasets`` is a 0.5.x-only API.
    ``pyproject.toml`` allows ``lerobot >= 0.3.3, < 0.6`` and the Dockerfile
    pins ``LEROBOT_VERSION`` at build time, so a 0.3.x image (the v2_1
    variant) has no merge API and cannot shard. Probed before any shard
    subprocess is spawned; the alternative is an opaque
    ``ModuleNotFoundError`` raised inside ``_aggregate_shards`` after every
    shard has already done its full conversion work.
    """
    try:
        from lerobot.datasets.aggregate import aggregate_datasets  # noqa: F401
    except ImportError:
        return False
    return True


def _aggregate_shards(
    shard_results: list[dict],
    dataset_root: Path,
) -> None:
    """Merge per-shard datasets into ``dataset_root`` via lerobot's aggregator.

    ``aggregate_datasets`` uses stream-copy concatenation
    (``concatenate_video_files`` → ffmpeg's concat demuxer in copy mode)
    so the merge cost is proportional to disk I/O, not video re-encode.
    """
    from lerobot.datasets.aggregate import aggregate_datasets

    repo_ids = [r["repo_id"] for r in shard_results]
    roots = [Path(r["root"]) for r in shard_results]
    logger.info(
        "Aggregating %d shards into %s via lerobot.aggregate_datasets...",
        len(shard_results), dataset_root,
    )
    aggregate_datasets(
        repo_ids=repo_ids,
        aggr_repo_id=dataset_root.name,
        roots=roots,
        aggr_root=dataset_root,
    )


def _drop_empty_shards(shard_results: list[dict]) -> list[dict]:
    """Partition shard results and exclude the zero-episode ones from
    aggregation.

    A shard subprocess whose *every* event is soft-dropped during drain
    (see ``_record_skip``) still calls ``writer.finalize()`` and reports
    back ``status: "ok"`` with ``total_episodes: 0`` — ``_run_shard_subprocess``
    has no way to know "wrote nothing" should be treated differently from
    "wrote something". But an empty shard's ``meta/`` directory lacks
    ``tasks.parquet`` and ``meta/episodes/``, and lerobot's
    ``LeRobotDatasetMetadata.__init__`` (invoked from inside
    ``aggregate_datasets``) catches the resulting local ``FileNotFoundError``
    and silently falls back to fetching the repo from the Hugging Face Hub.
    That repo does not exist, so the run dies deep inside lerobot with an
    opaque ``RepositoryNotFoundError: 401`` for a repo named after the shard
    ("shard-000") — completely masking the real per-event drop errors that
    are sitting right there in ``skipped_events``.

    Filtering here, before ``_aggregate_shards`` ever runs, means an empty
    shard's drop reasons reach the operator as a WARNING (or, if every shard
    came back empty, a ``RuntimeError``) instead of a misleading HTTP 401
    several stack frames into a third-party library.

    Returns the non-empty shard results, in their original order, for
    ``_aggregate_shards``. Callers that fold per-shard counters into the
    manifest (``per_dataset_counts``, ``episode_to_event``,
    ``skipped_events``) should keep using the *original*, unfiltered
    ``shard_results`` list — empty shards contribute nothing to the
    episode/frame counts either way, and dropping them from the manifest's
    ``skipped_events`` would throw away the very diagnostics this function
    exists to surface.
    """
    non_empty = [r for r in shard_results if int(r.get("total_episodes", 0)) > 0]
    empty = [r for r in shard_results if int(r.get("total_episodes", 0)) == 0]
    for r in empty:
        logger.warning(
            "Shard %s produced 0 episodes and will be excluded from "
            "aggregation: %s",
            r.get("shard_idx"), _summarize_skip_reasons(r.get("skipped_events", [])),
        )
    if not non_empty:
        all_skipped = [s for r in shard_results for s in r.get("skipped_events", [])]
        raise RuntimeError(
            f"All {len(shard_results)} shard(s) produced 0 episodes; nothing "
            f"to aggregate. Dropped events: {_summarize_skip_reasons(all_skipped)}"
        )
    return non_empty


def _renumber_episode_to_event(shard_results: list[dict]) -> list[dict]:
    """Flatten per-shard episode_to_event lists into a single list whose
    ``episode_index`` matches the merged dataset's global numbering.

    Shards are processed in ``shard_idx`` order (= chronological order, by
    partition construction), and each shard's local episode_index runs
    0..total_episodes-1, so the global index is just an offset accumulator.
    """
    flat: list[dict] = []
    offset = 0
    for r in shard_results:
        for entry in r["episode_to_event"]:
            flat.append({**entry, "episode_index": offset + int(entry["episode_index"])})
        offset += int(r["total_episodes"])
    return flat


def _warn_oversubscription(
    *,
    shard_count: int,
    encoder_threads: int,
    pool_size: int,
) -> None:
    """Log a warning if shard fan-out would oversubscribe the host CPU.

    Invariant: ``shard_count × encoder_threads`` should not exceed host
    vCPUs — the encoder threads are the cores that genuinely run hot;
    decode workers are largely idle in steady state. Warn rather than
    clamp so operators on hosts with measured headroom can opt in
    deliberately.

    Worker processes get a softer informational log: they amplify the
    process count but are CPU-light in steady state.
    """
    vcpus = os.cpu_count() or 0
    enc_total = shard_count * encoder_threads
    if vcpus and enc_total > vcpus:
        logger.warning(
            "Oversubscription guard: shard_count=%d × encoder_threads=%d "
            "= %d encoder cores requested but host has %d vCPUs. "
            "Throughput is likely to flatten or regress; reduce shard_count "
            "or encoder_threads, or override deliberately.",
            shard_count, encoder_threads, enc_total, vcpus,
        )
    workers_total = shard_count * pool_size
    if vcpus and workers_total > 2 * vcpus:
        logger.info(
            "Heads up: shard_count=%d × pool_size=%d = %d worker processes "
            "vs %d vCPUs. The decode work is largely idle steady-state, but "
            "watch memory: each worker holds ~one episode's payload.",
            shard_count, pool_size, workers_total, vcpus,
        )


def main(context: roboto.InvocationContext) -> None:
    """Convert Roboto ingested files to a LeRobot dataset.

    Loads events from the input Roboto Collection, then writes one
    combined LeRobot dataset where each episode corresponds to one
    event.
    """
    # Install before any LeRobotWriter.create (i.e. before the libsvtav1
    # encoder starts) so SVT-AV1's fd-2 banner output is filtered out at
    # the source. Spawned shard/event-worker children inherit the redirect.
    install_svt_stderr_filter()
    # Pin libav's C-level threshold to WARNING and silence tqdm /
    # HuggingFace-datasets progress bars. Both states are process-local
    # and must be re-applied at the top of _run_shard_subprocess for
    # spawned shard children.
    set_libav_log_level()
    disable_progress_bars()

    logger.setLevel(context.log_level)
    logger.info(
        "Starting roboto-to-lerobot (invocation_id=%s)", context.invocation_id
    )
    # Single source of truth for "which variant is running" — the action
    # name picks the image, the image picks the lerobot version, this log
    # line surfaces it in the invocation log.
    logger.info("Embedded lerobot version: %s", _pkg_version("lerobot"))

    collection_id = context.get_parameter("collection_id")
    if not collection_id:
        raise ValueError("collection_id must be a non-empty string")

    contract, contract_rel_path, contract_sha256 = _load_contract_from_context(context)

    fps = int(contract.fps)
    time_step_ns = fps_to_time_step_ns(contract.fps)
    buffer_ns = _compute_buffer_ns(contract)
    action_lead_ns = contract.action_lead_steps * time_step_ns

    # ``Collection.from_id`` defaults to ``content_mode=Full``, which makes
    # the server hydrate every event resource and return full EventRecord
    # payloads in ``record.resources``. Building ``Event`` instances from
    # those records avoids an N-extra-RTT round trip per invocation that
    # ``collection.events`` → ``Event.from_id(eid)`` would otherwise pay.
    collection = roboto.Collection.from_id(collection_id)
    if collection.record.resource_type != roboto.CollectionResourceType.Event:
        raise ValueError(
            f"collection_id {collection_id!r} has resource_type="
            f"{collection.record.resource_type.value!r}; this action requires "
            f"a collection of events (resource_type='event')."
        )
    collection_version = collection.record.version

    hydrated_event_resources = collection.record.resources.get(
        roboto.CollectionResourceType.Event, []
    )
    if not hydrated_event_resources:
        raise ValueError(
            f"Collection {collection_id!r} (version {collection_version}) "
            f"contains no events."
        )
    missing_events = collection.record.missing.get(
        roboto.CollectionResourceType.Event, []
    )
    if missing_events:
        logger.warning(
            "Collection %s references %d event(s) that could not be hydrated "
            "(deleted or inaccessible) — skipping: %s",
            collection_id,
            len(missing_events),
            sorted(ref.resource_id for ref in missing_events),
        )
    events = sorted(
        (
            roboto.Event(
                roboto.EventRecord.model_validate(r) if isinstance(r, dict) else r,
            )
            for r in hydrated_event_resources
        ),
        key=lambda e: e.start_time,
    )
    logger.info(
        "Loaded %d events from collection %s (version %d)",
        len(events), collection_id, collection_version,
    )

    # Earliest event (chronologically) with at least one dataset
    # association — its DataCollection is reused for feature discovery
    # and (in the single-writer path) inlined as the first episode.
    distinct_ds_ids: set[str] = set()
    first_event: roboto.Event | None = None
    first_ds_id: str | None = None
    for event in events:
        ds_ids = list(event.dataset_ids() or [])
        if not ds_ids:
            logger.warning(
                "Event %s has no dataset_ids; skipping",
                getattr(event, "event_id", "?"),
            )
            continue
        if first_event is None:
            first_event = event
            first_ds_id = ds_ids[0]
        distinct_ds_ids.update(ds_ids)
    if first_event is None or first_ds_id is None:
        raise ValueError(
            f"None of the {len(events)} event(s) in collection {collection_id!r} "
            f"are associated with a dataset."
        )

    # Per-dataset role resolution before topic discovery: a contract that
    # binds specs to ``role:`` placeholders gets its topics filled in from
    # ``file.metadata["role"]`` on this dataset's files. Contracts that
    # already use literal ``topic:`` pass through ``resolve_role_bindings``
    # unchanged.
    prepared: dict[str, tuple[roboto.Dataset, Contract, dict]] = {}
    # Per-dataset dedup reports, flattened later into the manifest.
    dedup_by_dataset: dict[str, list[dict]] = {}
    logger.info(
        "Matching contract topics against files in %d dataset(s)...",
        len(distinct_ds_ids),
    )
    for ds_id in distinct_ds_ids:
        ds = roboto.Dataset.from_id(ds_id)
        resolved = resolve_role_bindings(ds, contract)
        topics, dedup_groups = collect_topics_from_dataset(ds, resolved)
        prepared[ds_id] = (ds, resolved, topics)
        if dedup_groups:
            dedup_by_dataset[ds_id] = dedup_groups

    # Write under output_dir/<invocation_id>/ so the runtime's auto-upload
    # lands it at <invocation_dataset>/<invocation_id>/..., keeping repeat
    # invocations on the same dataset from clobbering each other.
    subfolder = context.output_dir / context.invocation_id
    dataset_root = subfolder / "combined"
    dataset_root.parent.mkdir(parents=True, exist_ok=True)

    _, first_resolved, first_topics = prepared[first_ds_id]

    logger.info("Preloading first event to discover feature shapes...")
    first_data = DataCollection(
        contract=first_resolved,
        topics=first_topics,
        start_time_ns=first_event.start_time - buffer_ns,
        end_time_ns=first_event.end_time + buffer_ns + action_lead_ns,
    )

    encoder_threads = _resolve_encoder_threads(
        context.get_optional_parameter("encoder_threads"),
    )
    logger.info(
        "Encoder threads (LP per SVT-AV1 stream) resolved to %d.", encoder_threads,
    )
    # Feature discovery: compute once on main, hand the dict to either the
    # single-writer path or every shard process. The dict is plain
    # JSON-shaped Python so it survives spawn's pickle round-trip cheaply.
    features = first_resolved.get_lerobot_features(
        resolved_features=first_data.resolved_features,
    )

    # Index every event-with-a-matching-dataset in chronological order.
    # Events whose ``dataset_ids`` do not intersect ``prepared`` are
    # silently skipped; the index becomes the ``event_idx`` we ship to the
    # pool and drain by below.
    #
    # ``task_label`` here is the *raw* event metadata task only (``None``
    # when absent/empty) — final resolution against the contract's
    # ``tasks:`` stream (which needs the per-episode ``DataCollection``,
    # not yet built at this point) happens later via
    # ``resolve_task_label``, in ``_worker_build_episode`` for pool tasks
    # and inline for the preloaded episode below.
    matched: list[tuple[int, str, str | None, int]] = []  # (event_idx, src_ds_id, task_label, num_frames)
    matched_events: list = []
    for event in events:
        ds_ids = [d for d in (event.dataset_ids() or []) if d in prepared]
        if not ds_ids:
            continue
        src_ds_id = ds_ids[0]
        raw_task_meta = (event.metadata or {}).get("task")
        task_label = str(raw_task_meta) if raw_task_meta else None
        num_frames = int((event.end_time - event.start_time) / time_step_ns) + 1
        event_idx = len(matched)
        matched.append((event_idx, src_ds_id, task_label, num_frames))
        matched_events.append(event)

    if not matched_events:
        raise ValueError(
            "Every event was filtered by the prepared-datasets intersection; "
            "nothing to convert."
        )

    # Materialised _DeferredFrame entries on the main process need the video
    # spec for decoder dispatch + resize. Videos pass through
    # ``resolve_role_bindings`` unchanged, so a single mapping covers every
    # resolved contract.
    video_specs_by_key = {v.key: v for v in contract.videos}

    # Build per-dataset payloads for ``_worker_init``: descriptor lists so
    # each worker can rehydrate Topic objects via ``Topic.from_id`` against
    # its own RobotoClient.
    descriptors_by_ds: dict[str, dict[str, list]] = {}
    contracts_by_ds: dict[str, Contract] = {}
    for ds_id, (_, resolved_ds, topics_dict) in prepared.items():
        descriptors_by_ds[ds_id] = topic_descriptors_from_topics(topics_dict)
        contracts_by_ds[ds_id] = resolved_ds

    shard_count = _resolve_shard_count(
        context.get_optional_parameter("shard_count"),
    )
    if shard_count > 1 and not _shard_path_supported():
        # Both v3-only knobs behave the same way on a lerobot 0.3.x image:
        # accepted and ignored, never fatal. action.json is shared by both
        # deployed variants and defaults shard_count to 3 for v3_0's benefit,
        # so failing here would break every default invocation of v2_1.
        # Clamped before pool_size resolves so auto-sizing divides the vCPU
        # budget by the shard count actually used, not the one requested.
        # ``_pkg_version`` is ``importlib.metadata.version``, which raises
        # PackageNotFoundError when lerobot is not installed at all. This is a
        # recovery path, so it must not be the thing that crashes the run.
        try:
            embedded = _pkg_version("lerobot")
        except Exception:
            embedded = "not installed"
        logger.warning(
            "shard_count=%d ignored: sharded writing requires lerobot 0.5.x "
            "(lerobot.datasets.aggregate.aggregate_datasets is missing) and "
            "the embedded lerobot is %s. Falling back to the single-writer "
            "path (shard_count=1).",
            shard_count, embedded,
        )
        shard_count = 1
    pool_size = _resolve_pool_size(
        context.get_optional_parameter("pool_size"),
        shard_count=shard_count,
    )
    logger.info(
        "Shards=%d, pool=%d worker(s) per shard, encoder_threads=%d per shard. "
        "Each worker uses up to %d threads for topic fetches.",
        shard_count, pool_size, encoder_threads,
        _TOPIC_FETCH_THREADS,
    )

    accumulator = _DrainAccum()

    if shard_count == 1:
        # Single-writer path — keep the preloaded-event optimisation: the
        # ``DataCollection`` we already paid for feature discovery feeds
        # the first episode inline, no extra fetch round-trip.
        writer = LeRobotWriter.create(
            repo_id=dataset_root.name,
            fps=fps,
            features=features,
            root=dataset_root,
            robot_type=contract.robot_type or "unknown",
            image_writer_threads=_IMAGE_WRITER_THREADS,
            batch_encoding_size=_BATCH_ENCODING_SIZE,
            streaming_encoding=_STREAMING_ENCODING,
            encoder_threads=encoder_threads,
            encoder_queue_maxsize=_ENCODER_QUEUE_MAXSIZE,
        )

        # ``first_event`` was selected from the first prepared dataset's
        # chronologically-first event, so it is always in ``matched_events``.
        preloaded_idx = matched_events.index(first_event)
        preloaded_event_idx = matched[preloaded_idx][0]

        # Drain order is the chronological list of every event the writer
        # consumes; ``pool_tasks`` is everything except the preloaded slot
        # (drained inline by event_idx match inside ``_run_pool_drain``).
        drain_order: list[dict] = []
        pool_tasks: list[EventTask] = []
        for slot, (event_idx, src_ds_id, task_label, num_frames) in enumerate(matched):
            ev = matched_events[slot]
            drain_order.append({
                "event_idx": event_idx,
                "src_ds_id": src_ds_id,
                "task_label": task_label,
                "event_id": getattr(ev, "event_id", None),
                "start_time_ns": int(ev.start_time),
                "end_time_ns": int(ev.end_time),
            })
            if slot == preloaded_idx:
                continue
            pool_tasks.append(EventTask(
                event_idx=event_idx,
                event_id=getattr(ev, "event_id", None),
                src_ds_id=src_ds_id,
                start_time_ns=int(ev.start_time),
                end_time_ns=int(ev.end_time),
                task_label=task_label,
                num_frames=num_frames,
                buffer_ns=buffer_ns,
                action_lead_ns=action_lead_ns,
            ))

        # Build the preloaded episode (frames in memory, undeferred — the
        # writer accepts numpy arrays directly so materialize_deferred is a
        # no-op for this slot).
        preloaded_event = matched_events[preloaded_idx]
        _, _, preloaded_task, preloaded_num_frames = matched[preloaded_idx]
        preloaded_ref_ts = pd.Series(
            [int(preloaded_event.start_time) + i * time_step_ns
             for i in range(preloaded_num_frames)],
            name="timestamp",
        )
        try:
            resolved_preloaded_task = resolve_task_label(
                contract=first_resolved,
                episode_data=first_data,
                start_time_ns=int(preloaded_event.start_time),
                end_time_ns=int(preloaded_event.end_time),
                metadata_task=preloaded_task,
            )
            preloaded_frames = list(generate_frames(
                first_resolved, first_data, preloaded_ref_ts, task=resolved_preloaded_task,
            ))
            preloaded_episode = WorkerEpisode(
                event_idx=preloaded_event_idx,
                task_label=resolved_preloaded_task,
                frames=preloaded_frames,
            )
        except Exception as exc:
            # Demote the failed event to a pool task instead of dropping it
            # outright: the pool-stage retry exercises the same code path
            # every other event runs, so a deterministic preload bug shows
            # up as a second skip entry with stage="worker" and the
            # operator gets symmetric diagnostics.
            _record_skip(
                accumulator=accumulator,
                meta={
                    "event_idx": preloaded_event_idx,
                    "event_id": getattr(preloaded_event, "event_id", None),
                    "src_ds_id": matched[preloaded_idx][1],
                    "start_time_ns": int(preloaded_event.start_time),
                    "end_time_ns": int(preloaded_event.end_time),
                    "task_label": preloaded_task,
                },
                exc=exc,
                stage="preload",
            )
            pool_tasks.append(EventTask(
                event_idx=preloaded_event_idx,
                event_id=getattr(preloaded_event, "event_id", None),
                src_ds_id=matched[preloaded_idx][1],
                start_time_ns=int(preloaded_event.start_time),
                end_time_ns=int(preloaded_event.end_time),
                task_label=preloaded_task,
                num_frames=preloaded_num_frames,
                buffer_ns=buffer_ns,
                action_lead_ns=action_lead_ns,
            ))
            preloaded_event_idx = None
            preloaded_episode = None

        logger.info(
            "Single-writer drain: %d pool events + 1 preloaded (slot=%d), "
            "pool=%d worker(s)",
            len(pool_tasks), preloaded_idx, pool_size,
        )
        _run_pool_drain(
            writer=writer,
            video_specs_by_key=video_specs_by_key,
            pool_tasks=pool_tasks,
            drain_order=drain_order,
            descriptors_by_ds=descriptors_by_ds,
            contracts_by_ds=contracts_by_ds,
            pool_size=pool_size,
            accumulator=accumulator,
            preloaded_event_idx=preloaded_event_idx,
            preloaded_episode=preloaded_episode,
        )
        writer.finalize()
        # Same failure mode the sharded path guards against in
        # ``_drop_empty_shards``: every event can soft-drop during drain
        # (see ``_record_skip``) while ``writer.finalize()`` still succeeds
        # on an empty dataset. There is no downstream ``aggregate_datasets``
        # call on this path to crash with a misleading Hub 401, but a
        # 0-episode "successful" run is just as silently wrong — fail loudly
        # here with the real drop reasons instead of letting the manifest
        # quietly report ``total_episodes: 0``.
        if accumulator.total_episodes == 0:
            raise RuntimeError(
                "Conversion produced 0 episodes; nothing was written. "
                f"Dropped events: {_summarize_skip_reasons(accumulator.skipped_events)}"
            )
    else:
        # Sharded path: drop the preloaded-event optimisation (one event of
        # repeat fetch per invocation, amortised across K shards), spawn K
        # shards each with their own writer + pool, then merge with
        # ``aggregate_datasets`` (stream-copy concat, no re-encode).
        # Free the feature-discovery DataCollection before spawning so each
        # shard does not inherit a stale copy of the first event's payload.
        del first_data
        chunks_drain, chunks_tasks = _partition_shards(
            matched, matched_events,
            shard_count=shard_count,
            buffer_ns=buffer_ns,
            action_lead_ns=action_lead_ns,
        )
        effective_shard_count = len(chunks_drain)
        if effective_shard_count < shard_count:
            logger.info(
                "Requested shard_count=%d but only %d events qualify; "
                "running %d shard(s) instead.",
                shard_count, len(matched_events), effective_shard_count,
            )
        # Warn against the actual fan-out, not the request: with few events
        # the partitioner clamps to one event per shard, so a request of 8
        # against 2 events fans out to 2 shards and the request-based
        # warning would overstate the contention.
        _warn_oversubscription(
            shard_count=effective_shard_count,
            encoder_threads=encoder_threads,
            pool_size=pool_size,
        )
        # Per-shard temp datasets live under the invocation subfolder but
        # are removed before the auto-upload sweep — only the merged
        # ``combined/`` dataset should hit the dataset.
        shard_workdir = subfolder / "_shards"
        logger.info(
            "Sharded drain: %d shard(s) writing to %s, then aggregating "
            "into %s",
            effective_shard_count, shard_workdir, dataset_root,
        )
        shard_results = _spawn_shards(
            chunks_drain=chunks_drain,
            chunks_tasks=chunks_tasks,
            shard_workdir=shard_workdir,
            fps=fps,
            features=features,
            robot_type=contract.robot_type or "unknown",
            encoder_threads=encoder_threads,
            pool_size=pool_size,
            video_specs_by_key=video_specs_by_key,
            descriptors_by_ds=descriptors_by_ds,
            contracts_by_ds=contracts_by_ds,
            log_level=context.log_level,
        )
        # ``shard_results`` (unfiltered) still feeds every fold-in below —
        # only the aggregation call itself should see empty shards excluded.
        # See ``_drop_empty_shards`` for why this ordering matters (a masked
        # HF-Hub 401 vs. a loud, on-point RuntimeError).
        aggregatable_shards = _drop_empty_shards(shard_results)
        _aggregate_shards(aggregatable_shards, dataset_root)
        # ``aggregate_datasets`` has copied every video + parquet it needs
        # into ``dataset_root``. Drop the per-shard scratch so it does not
        # get auto-uploaded by the runtime.
        try:
            shutil.rmtree(shard_workdir)
        except OSError as e:
            logger.warning(
                "Could not remove shard workdir %s: %s", shard_workdir, e,
            )
        # Fold per-shard counters into the manifest-shaped accumulator the
        # downstream code expects. Per-shard episode indices are local
        # (0..n_shard); ``_renumber_episode_to_event`` rebases them.
        for r in shard_results:
            for ds_id, n in r["per_dataset_counts"].items():
                accumulator.per_dataset_counts[ds_id] += int(n)
        accumulator.total_episodes = sum(
            int(r["total_episodes"]) for r in shard_results
        )
        accumulator.total_frames = sum(
            int(r["total_frames"]) for r in shard_results
        )
        accumulator.episode_to_event = _renumber_episode_to_event(shard_results)
        # Skipped events have no episode_index — no renumbering needed; just
        # concatenate in shard_idx order so the manifest reflects how the
        # operator-facing aggregator stitched chronology back together.
        for r in shard_results:
            accumulator.skipped_events.extend(r.get("skipped_events", []))

    # Manifest assembly expects the pre-refactor names; rebind from the
    # accumulator so the rest of ``main`` stays untouched.
    total_frames = accumulator.total_frames
    total_episodes = accumulator.total_episodes
    per_dataset_counts = accumulator.per_dataset_counts
    episode_to_event = accumulator.episode_to_event

    # Archive the contract bytes so the run survives later edits to the source.
    archived_contract_filename = "contract.yaml"
    archived_contract_path = subfolder / archived_contract_filename
    shutil.copyfile(context.input_dir / contract_rel_path, archived_contract_path)
    logger.info("Contract archived: %s", archived_contract_path)

    lerobot_version = _pkg_version("lerobot")

    # Flatten per-dataset dedup reports; each entry carries enough context
    # (dataset_id + per-file file_ids/paths) to find the source upload.
    dedup_records: list[dict] = []
    dropped_file_ids: set[str] = set()
    for ds_id, groups in dedup_by_dataset.items():
        for g in groups:
            dedup_records.append({"source_dataset_id": ds_id, **g})
            for d in g["dropped"]:
                dropped_file_ids.add(d["file_id"])
    dedup_summary = {
        "duplicate_groups": len(dedup_records),
        "topics_dropped": sum(len(g["dropped"]) for g in dedup_records),
        "dropped_file_ids": sorted(dropped_file_ids),
    }

    # Soft-dropped events: full per-event entries (with traceback) plus a
    # rollup summary. Mirrors the dedup shape so operators have a single
    # place to look when a run produced fewer episodes than expected.
    skipped_events_records = list(accumulator.skipped_events)
    skipped_events_summary = _build_skipped_summary(skipped_events_records)

    # Self-describing manifest lives at the root of the invocation subfolder.
    manifest = {
        "invocation_id": context.invocation_id,
        "contract": {
            "name": contract.name,
            "version": contract.version,
            "path": contract_rel_path,
            "sha256": contract_sha256,
            "archived_filename": archived_contract_filename,
        },
        "collection_id": collection_id,
        "collection_version": collection_version,
        "source_datasets": sorted(prepared.keys()),
        "episodes_per_dataset": dict(per_dataset_counts),
        "episode_to_event": episode_to_event,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "dedup": dedup_records,
        "dedup_summary": dedup_summary,
        "skipped_events": skipped_events_records,
        "skipped_events_summary": skipped_events_summary,
        "codebase": {
            "lerobot": lerobot_version,
        },
        "generated_at": datetime.now(UTC).isoformat(),
    }
    (subfolder / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("Manifest written: %s", subfolder / "manifest.json")

    # Stamp every uploaded file with contract provenance so you can query
    # `metadata.contract_name == '<name>'` against the dataset later.
    per_file_stamp = {
        "invocation_id": context.invocation_id,
        "contract_name": contract.name,
        "contract_version": contract.version,
        "contract_sha256": contract_sha256,
        "contract_path": contract_rel_path,
        "collection_id": collection_id,
        "collection_version": collection_version,
        "lerobot_version": lerobot_version,
    }
    _stamp_output_files(context, subfolder, per_file_stamp)

    # The runtime auto-uploads ``output_dir`` to the invocation's upload
    # destination, which is NOT necessarily ``context.dataset_id`` (that is the
    # *input* dataset the action operates on — often a contract-only dataset,
    # with the actual source data supplied as a collection). When the invocation
    # is created with an explicit ``upload_destination``, outputs land there
    # instead. Resolve the real destination; fall back to the input dataset (the
    # runtime's default when no destination is set).
    output_dataset = context.dataset
    output_dataset_id = context.dataset_id
    try:
        upload_dest = context.invocation.upload_destination
        if (
            upload_dest is not None
            and upload_dest.is_dataset
            and upload_dest.destination_id != context.dataset_id
        ):
            output_dataset = roboto.Dataset.from_id(
                upload_dest.destination_id, roboto_client=context.roboto_client
            )
            output_dataset_id = upload_dest.destination_id
    except Exception as e:  # local runs / no invocation record
        logger.debug("Could not resolve upload destination: %s", e)

    # Dataset-level index of invocations, stamped on the dataset the converted
    # output actually lands in (co-located with the per-file provenance stamps),
    # not the contract-only input dataset. Queryable by id:
    #   metadata.invocations.<id>.contract_name = '<name>'
    # Discovery of datasets with any conversion:
    #   metadata.invocations IS NOT NULL  (or: the key exists)
    output_dataset.update(
        metadata_changeset=MetadataChangeset.Builder()
        .put_field(f"invocations.{context.invocation_id}", {
            "contract_name": contract.name,
            "contract_version": contract.version,
            "contract_sha256": contract_sha256,
            "contract_path": contract_rel_path,
            "source_dataset_id": context.dataset_id,
            "collection_id": collection_id,
            "collection_version": collection_version,
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "lerobot_version": lerobot_version,
            "had_dedup": bool(dedup_records),
            "dedup_groups": len(dedup_records),
        })
        .build()
    )

    if dedup_records:
        logger.warning(
            "Topic dedup: dropped %d topic copies across %d duplicate groups "
            "(see manifest.dedup for details).",
            dedup_summary["topics_dropped"], dedup_summary["duplicate_groups"],
        )

    if skipped_events_records:
        logger.warning(
            "Soft-drop: %d event(s) failed conversion and were skipped "
            "(by_stage=%s, by_error_class=%s, see manifest.skipped_events).",
            skipped_events_summary["count"],
            skipped_events_summary["by_stage"],
            skipped_events_summary["by_error_class"],
        )

    logger.info(
        "Done. %d episodes, %d frames across %d datasets. "
        "Output at %s/ inside dataset %s.",
        total_episodes, total_frames, len(prepared),
        context.invocation_id, output_dataset_id,
    )
