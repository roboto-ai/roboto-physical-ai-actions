from __future__ import annotations

import hashlib
import http.client
import importlib.util
import json
import multiprocessing as mp
import os
import resource
import shutil
import ssl
import tempfile
import time
import traceback
import urllib.error
from collections import Counter, defaultdict
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib.metadata import version as _pkg_version
from multiprocessing.connection import wait as _wait_for_processes
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
import roboto
from roboto.exceptions import (
    RobotoFeatureNotAvailableException,
    RobotoServiceException,
    RobotoServiceUnavailableException,
)
from roboto.updates import MetadataChangeset

from .contract_utils import (
    _TOPIC_FETCH_THREADS,
    Contract,
    DataCollection,
    _compute_buffer_ns,
    fps_to_time_step_ns,
    load_contract,
    resolve_task_label,
)
from .episode_cache import (
    MARKER_NAME,
    EpisodeCache,
    entry_path,
    episode_key,
    episode_summary,
    recipe_hash,
)
from .episode_sources import annotate_manifest_entry, resolve_episode_sources
from .event_worker import (
    EventTask,
    WorkerEpisode,
    _worker_build_episode,
    _worker_init,
    topic_descriptors_from_topics,
)
from .lerobot import generate_frames, materialize_deferred
from .logger import logger
from .merge_v21 import merge_v21_entries
from .output_publish import parse_tags, publish_tagged_output
from .stderr_filter import (
    disable_progress_bars,
    install_svt_stderr_filter,
    set_libav_log_level,
)
from .subtasks import apply_session_subtasks
from .writers import LeRobotWriter
from .writers.base import EpisodeWriteError

if TYPE_CHECKING:
    from collections.abc import Callable
    from multiprocessing.connection import Connection

    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

# Writer-encoder knobs forwarded to ``LeRobotDataset.create``. The v2_1
# adapter accepts the same kwargs and silently drops 0.6.x-only ones, so
# callers do not need to branch on lerobot version. ``vcodec`` is hard-coded
# inside the v3_0 adapter — h264 worsened downstream training.
#
# ``_STREAMING_ENCODING=True`` pipes pixels straight into ffmpeg instead of
# writing PNGs to disk for ffmpeg to read back, saving wall time per run.
# Trade-off: lerobot 0.6.x routes video stats through the streaming
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
# lerobot's ``DatasetWriter.save_episode`` (0.5.x and 0.6.x) runs both the
# streaming branch and the batched-flush branch when both knobs are on; on
# 0.5.x the batched flush then crashes because
# ``EpisodesMeta.save_episode`` never refreshes the in-memory cache that the
# empty-dataset create path initialises to ``None``. Even absent the crash,
# the batched flush would redundantly
# re-encode every Nth episode from scratch. ``batch_encoding_size=1``
# disables the batched branch so only the streaming branch runs.
_BATCH_ENCODING_SIZE = 1
_STREAMING_ENCODING = True
# Default SVT-AV1 level of parallelism per camera stream; the
# ``encoder_threads`` action parameter overrides it. A fixed value, rather
# than SVT's automatic choice, keeps the encoders' CPU use predictable across
# shards (see ``_warn_oversubscription``). A low value makes the export wait
# on the encoder but loses no frames (``writers/streaming_encoder.py``).
_ENCODER_THREADS = 2
# Frames each camera's streaming encoder may hold in its queue. When the
# queue is full, ``add_frame`` waits for the encoder, so a larger queue lets
# the export run further ahead of the encoder, at a cost in resident memory
# of roughly maxsize × frame_size × camera_count.
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
# The file in the dataset's ``meta/`` that records where the dataset and each
# of its episodes came from. The Lance converter copies ``meta/`` whole.
_LINEAGE_NAME = "roboto_lineage.json"
# The lineage fields, with their Arrow types, that a v3.0 dataset also gets
# as ``roboto/<field>`` columns of its episode metadata, so that
# ``LeRobotDataset(episode_filter=...)`` can select episodes on them. All but
# ``recipe_hash`` differ per episode. LeRobot's loaders, LeRobot 0.7's
# ``aggregate_datasets`` and the Lance converter keep the columns.
_LINEAGE_COLUMNS: dict[str, str] = {
    "event_id": "string",
    "session_id": "string",
    "source_dataset_id": "string",
    "start_time_ns": "int64",
    "end_time_ns": "int64",
    "cache_hit": "bool",
    "recipe_hash": "string",
}
# Values of the ``output_format`` parameter: the LeRobot dataset, its Lance
# layout in its place, or both.
_OUTPUT_FORMATS = ("lerobot", "lance", "both")
# Threads the episode cache's merge uses to load the entries' metadata,
# probe their videos and write packed video files. The disk limits how fast
# the video files are written whatever the count; the threads mainly speed
# up loading the metadata.
_MERGE_THREADS = 4


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
    as unset at every layer. A value that is not a positive integer raises
    instead of falling back to the default, so a typo is not silently
    ignored.
    """
    if param_value is not None and param_value != "":
        return _parse_pool_size(param_value, source="encoder_threads parameter")
    raw = os.environ.get(_ENCODER_THREADS_ENV)
    if raw is None or raw == "":
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


def _resolve_output_format(param_value: object = None) -> str:
    """Resolve the ``output_format`` parameter; unset means ``"lerobot"``.

    Ignores case and surrounding whitespace. Raises ``ValueError`` on an
    unknown value, and on ``"lance"`` or ``"both"`` when the Lance converter
    (``lerobot-lancedb``) is not installed, as on the v2_1 variant. ``main``
    resolves it before converting any event, so either error fails the run
    early.
    """
    value = str(param_value or "lerobot").strip().lower()
    if value not in _OUTPUT_FORMATS:
        raise ValueError(f"output_format must be one of {', '.join(_OUTPUT_FORMATS)}, got {param_value!r}")
    if value != "lerobot" and importlib.util.find_spec("lerobot_lancedb") is None:
        raise ValueError(
            f"output_format={value!r} needs the Lance converter (lerobot-lancedb), which only "
            "roboto-to-lerobot-v3_0 includes."
        )
    return value


def _open_episode_cache(context: roboto.InvocationContext) -> EpisodeCache | None:
    """Return the episode cache in the org's assets when ``episode_cache`` is on and usable.

    Unset, empty or ``false`` returns ``None`` without any API call. With
    ``true``, makes one listing call against the invocation org's assets
    before any conversion work. When the installed SDK has no
    ``Org.assets``, or the org's asset storage is not enabled, logs one
    warning and returns ``None``, so the export converts every event as it
    does without the cache. Any other value raises ``ValueError``, so a typo
    does not silently disable the cache.
    """
    raw = context.get_optional_parameter("episode_cache")
    if raw is None or raw == "":
        return None
    value = str(raw).strip().lower()
    if value not in ("true", "false"):
        raise ValueError(f"episode_cache must be 'true' or 'false', got {raw!r}")
    if value == "false":
        return None

    files = getattr(context.org, "assets", None)
    if files is None:
        reason = "the installed roboto SDK has no Org.assets (needs roboto >= 0.61.0)"
    else:
        cache = EpisodeCache(files)
        try:
            cache.probe()
            return cache
        except RobotoFeatureNotAvailableException:
            reason = "org asset storage is not enabled for this org"
    logger.warning("episode_cache ignored: %s. Converting every event.", reason)
    return None


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
        "session_ids": sorted({
            str(s["session_id"]) for s in skipped_events
            if s.get("session_id") is not None
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


# Failures of building an event's episode that do not come from the event's
# data: a worker pool that broke (a worker process died, or its initializer
# failed) and network or service errors that outlasted the SDK's retries.
# Running the same export again can succeed, so ``_run_pool_drain`` fails the
# run on one of these; skipping the event would let an export that lacks it
# report success. Every other exception from a worker skips the event:
# ``_record_skip`` lists it in the manifest's ``skipped_events``.
_INFRASTRUCTURE_ERRORS: tuple[type[BaseException], ...] = (
    BrokenProcessPool,
    ConnectionError,  # includes http.client.RemoteDisconnected
    TimeoutError,
    ssl.SSLError,  # includes ssl.SSLEOFError
    urllib.error.URLError,
    http.client.HTTPException,
    RobotoServiceException,  # a 5xx response without a Roboto error body
    RobotoServiceUnavailableException,
)


def _infrastructure_failure(
    exc: BaseException, meta: dict, shard_idx: int | None,
) -> RuntimeError:
    """Build the error that fails the run when ``exc``, one of ``_INFRASTRUCTURE_ERRORS``, stops an event.

    ``meta`` is the event's ``drain_order`` entry or its ``EventTask`` as a
    dict; the message names the event by its ``event_idx``, ``event_id``
    and ``src_ds_id``.
    """
    if isinstance(exc, BrokenProcessPool):
        reason = (
            "the worker pool broke: a worker process died, or its initializer "
            "failed and logged its traceback above"
        )
    else:
        reason = "a network or service error outlasted the SDK's retries"
    return RuntimeError(
        f"[{_shard_tag(shard_idx)}] Event event_idx={meta.get('event_idx')} "
        f"(event_id={meta.get('event_id')}, dataset={meta.get('src_ds_id')}) "
        f"could not be converted because {reason}. The event's data is not at "
        "fault, so the run fails instead of dropping the event; running the "
        f"export again may succeed. Cause: {exc.__class__.__name__}: {exc}"
    )


def _preload_episode(
    *,
    contract,
    episode_data,
    event,
    event_idx: int,
    metadata_task,
    ref_ts: pd.Series,
) -> WorkerEpisode | None:
    """Build the first event's episode in the main process, or return ``None`` if that fails.

    The first event's data is already loaded to discover the features, so
    building its episode here saves fetching it again in a worker. A failure
    is logged and not recorded as a skip: the caller then submits the event
    to the pool like every other event, and what the pool does with it
    (saves it, skips it, or fails the run) is what the manifest reports.
    """
    try:
        task = resolve_task_label(
            contract=contract,
            episode_data=episode_data,
            start_time_ns=int(event.start_time),
            end_time_ns=int(event.end_time),
            metadata_task=metadata_task,
        )
        frames = list(generate_frames(contract, episode_data, ref_ts, task=task))
    except Exception:
        logger.warning(
            "Could not build the first event (event_id=%s) in the main process; converting it in the pool instead.",
            getattr(event, "event_id", None),
            exc_info=True,
        )
        return None
    return WorkerEpisode(event_idx=event_idx, task_label=task, frames=frames)


def _record_skip(
    *,
    accumulator: _DrainAccum,
    meta: dict,
    exc: BaseException,
    stage: Literal["worker", "drain"],
    shard_idx: int | None = None,
) -> None:
    """Append a soft-drop entry for one failed event and log a WARNING.

    Must be called from inside an ``except`` block — ``traceback.format_exc()``
    reads ``sys.exc_info()`` and would otherwise return ``"NoneType: None\n"``.

    ``stage`` ∈ ``{"worker", "drain"}`` records which surface raised, so
    post-hoc analysis can tell apart episode-build failures (the worker
    process) from writer-side failures (frame iteration / save_episode).
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
    on_episode_saved: Callable[[dict], None] | None = None,
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

    An event whose worker raises is recorded as a skip, unless the exception
    is one of ``_INFRASTRUCTURE_ERRORS``: then the drain raises the
    ``RuntimeError`` from :func:`_infrastructure_failure`, which names the
    event. A pool that is already broken when the next task is submitted
    fails the run the same way, naming that task's event. A failure to
    write an event's episode is recorded as a skip too, unless the writer
    raises :class:`EpisodeWriteError`, which fails the run.

    ``on_episode_saved``, if given, is called with an event's
    ``drain_order`` entry right after its episode is saved. An exception it
    raises is not recorded as a skip; it ends the drain.
    """

    def _drain(meta: dict, episode) -> None:
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
        except EpisodeWriteError:
            # An episode the writer could not save whole fails the run: in a
            # shared dataset, skipping it would leave rows and videos
            # misaligned, and a short video must never reach the episode cache.
            raise
        except Exception as exc:
            # The writer may have buffered frames before the failure;
            # discard them so the next episode cannot inherit them.
            _record_skip(
                accumulator=accumulator, meta=meta, exc=exc, stage="drain",
                shard_idx=shard_idx,
            )
            _safe_discard_episode(writer)
        else:
            if on_episode_saved is not None:
                on_episode_saved(meta)

    only_preloaded = (
        preloaded_episode is not None
        and not pool_tasks
        and len(drain_order) == 1
        and drain_order[0]["event_idx"] == preloaded_event_idx
    )
    if only_preloaded:
        _drain(drain_order[0], preloaded_episode)
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
                _drain(meta, preloaded_episode)
                continue
            while (
                next_to_submit < len(pool_tasks)
                and len(submitted) < max_in_flight
            ):
                t = pool_tasks[next_to_submit]
                try:
                    submitted[t.event_idx] = pool.submit(
                        _worker_build_episode, t,
                    )
                except BrokenProcessPool as exc:
                    raise _infrastructure_failure(exc, t._asdict(), shard_idx) from exc
                next_to_submit += 1
            # ``.result()`` raises a worker's exception before any of the
            # event's frames reach the writer, so skipping the event here
            # needs no discard. A failure after frames reach the writer is
            # handled, and discarded, by ``_drain``.
            try:
                episode = submitted.pop(event_idx).result()
            except _INFRASTRUCTURE_ERRORS as exc:
                raise _infrastructure_failure(exc, meta, shard_idx) from exc
            except Exception as exc:
                _record_skip(
                    accumulator=accumulator, meta=meta, exc=exc, stage="worker",
                    shard_idx=shard_idx,
                )
                continue
            _drain(meta, episode)
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


class _DatasetPerEpisodeWriter:
    """Writer that puts every episode in its own one-episode dataset.

    An episode's first frame creates a ``LeRobotWriter`` at
    ``<root>/episode-NNNN``; :meth:`finish_episode` finalizes that dataset
    after its episode is saved and returns its root. Each dataset gets the
    calls (create, frames, save, finalize) a writer converting only that
    episode would get, so its files are the same.
    """

    def __init__(self, root: Path, writer_kwargs: dict[str, Any]) -> None:
        self._root = root
        self._writer_kwargs = writer_kwargs
        self._count = 0
        self._writer: LeRobotWriter | None = None
        self._episode_root = root

    def add_frame(self, frame: dict[str, Any], task: str) -> None:
        if self._writer is None:
            self._episode_root = self._root / f"episode-{self._count:04d}"
            self._count += 1
            self._writer = LeRobotWriter.create(
                repo_id=self._episode_root.name, root=self._episode_root, **self._writer_kwargs,
            )
        self._writer.add_frame(frame, task=task)

    def save_episode(self) -> None:
        self._writer.save_episode()

    def discard_episode(self) -> None:
        """Drop the current episode, close its writer and delete its dataset, if it has one.

        Closing the writer stops its image-writer threads, which would
        otherwise run until the shard process exits.
        """
        if self._writer is None:
            return
        writer, self._writer = self._writer, None
        writer.discard_episode()
        writer.finalize()
        shutil.rmtree(self._episode_root, ignore_errors=True)

    def finish_episode(self) -> Path:
        writer, self._writer = self._writer, None
        writer.finalize()
        return self._episode_root

    def finalize(self) -> None:
        """Nothing to do: :meth:`finish_episode` finalized every dataset."""


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
    episode_conn: Connection | None = None,
) -> None:
    """Entrypoint for a single shard process.

    Spawned by ``main`` with ``mp_context="spawn"`` when ``shard_count > 1``;
    builds an independent ``LeRobotWriter`` rooted at ``shard_root``, drains
    its slice of events through the same ``_run_pool_drain`` loop the
    single-writer path uses, then writes a JSON summary to ``result_path``.
    The main process reads the summary back to assemble the manifest and
    drives :func:`lerobot.datasets.aggregate.aggregate_datasets` to merge
    every shard into the final destination.

    With ``episode_conn``, the shard instead writes each episode as its own
    one-episode dataset under ``shard_root`` and, as soon as one is
    finalized, sends ``{"event_idx", "root", "episode"}`` on
    ``episode_conn``, where ``episode`` is its ``episode_to_event`` entry.
    One process and one worker pool still convert all of the shard's events;
    only the output is split by episode.

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
        writer_kwargs = {
            "fps": fps,
            "features": features,
            "robot_type": robot_type,
            "image_writer_threads": _IMAGE_WRITER_THREADS,
            "batch_encoding_size": _BATCH_ENCODING_SIZE,
            "streaming_encoding": _STREAMING_ENCODING,
            "encoder_threads": encoder_threads,
            "encoder_queue_maxsize": _ENCODER_QUEUE_MAXSIZE,
        }
        accumulator = _DrainAccum()
        on_episode_saved = None
        if episode_conn is None:
            writer = LeRobotWriter.create(repo_id=repo_id, root=shard_root_p, **writer_kwargs)
        else:
            writer = _DatasetPerEpisodeWriter(shard_root_p, writer_kwargs)

            def on_episode_saved(meta: dict) -> None:
                # ``_drain_one_episode`` appended the saved episode's entry last.
                episode_conn.send({
                    "event_idx": meta["event_idx"],
                    "root": str(writer.finish_episode()),
                    "episode": accumulator.episode_to_event[-1],
                })

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
            on_episode_saved=on_episode_saved,
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
    on_episodes: Callable[[list[dict]], None] | None = None,
) -> list[dict]:
    """Spawn one process per shard, wait for all, return per-shard results.

    Each shard receives the topic descriptors and contracts of only the
    datasets its events come from. Spawn pickles them into the shard
    process and again into each of its pool workers, so a shard does not
    carry the topic records of datasets it never reads.

    With ``on_episodes``, every shard writes each episode as its own
    one-episode dataset (see :func:`_run_shard_subprocess`) and reports it
    once finalized. ``on_episodes`` receives the reports in batches while
    the shards run. An exception it raises is logged and does not stop the
    shards; the ``RuntimeError`` described below lists it once every shard
    has exited.

    Failures (non-zero exit, missing result file, ``status != "ok"``) are
    collected and surfaced as a single ``RuntimeError`` so the operator
    sees every shard's traceback in one place instead of chasing
    stale-pid output. The result list is sorted by ``shard_idx`` so
    downstream aggregation runs in original chronological order.
    """
    shard_count = len(chunks_drain)
    ctx = mp.get_context("spawn")
    running: list[mp.Process] = []
    readers: list[Connection] = []
    failures: list[str] = []
    result_paths: list[Path] = []
    shard_roots: list[Path] = []

    def _collect_reports(ready: list) -> None:
        reports: list[dict] = []
        for reader in [r for r in readers if r in ready]:
            try:
                while reader.poll():
                    reports.append(reader.recv())
            except EOFError:
                readers.remove(reader)
                reader.close()
        if reports:
            try:
                on_episodes(reports)
            except Exception as exc:
                logger.exception("Handling %d finished episode(s) failed", len(reports))
                failures.append(f"on_episodes {exc.__class__.__name__}: {exc}")

    shard_workdir.mkdir(parents=True, exist_ok=True)
    for i in range(shard_count):
        shard_ds_ids = {t.src_ds_id for t in chunks_tasks[i]}
        shard_root = shard_workdir / f"shard-{i:03d}"
        result_path = shard_workdir / f"shard-{i:03d}.result.json"
        shard_roots.append(shard_root)
        result_paths.append(result_path)
        reader, episode_conn = ctx.Pipe(duplex=False) if on_episodes else (None, None)
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
                "descriptors_by_ds": {ds: descriptors_by_ds[ds] for ds in shard_ds_ids},
                "contracts_by_ds": {ds: contracts_by_ds[ds] for ds in shard_ds_ids},
                "pool_tasks": chunks_tasks[i],
                "drain_order": chunks_drain[i],
                "result_path": str(result_path),
                "episode_conn": episode_conn,
            },
            name=f"shard-{i:03d}",
        )
        p.start()
        running.append(p)
        if reader is not None:
            # Only the shard keeps the sending end, so the reader reaches
            # EOF when the shard exits.
            episode_conn.close()
            readers.append(reader)

    # Join every shard before raising — partial joins leak orphaned shard
    # children, and we want a clean shutdown either way.
    while running:
        ready = _wait_for_processes([p.sentinel for p in running] + readers)
        _collect_reports(ready)
        for p in [p for p in running if p.exitcode is not None]:
            p.join()
            running.remove(p)
            if p.exitcode != 0:
                failures.append(f"{p.name} exit={p.exitcode}")
    # Read the reports that were still in a pipe when the last shard exited.
    _collect_reports(list(readers))

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
            + "".join(
                f"\nshard {r.get('shard_idx')}: {r.get('error_class')}: {r.get('error_message')}"
                for r in bad
            )
        )

    results.sort(key=lambda r: r["shard_idx"])
    for r in results:
        r["root"] = str(shard_roots[r["shard_idx"]])
    return results


def _shard_path_supported() -> bool:
    """Whether the embedded lerobot has ``lerobot.datasets.aggregate.aggregate_datasets``.

    lerobot 0.6 and later have it; 0.3.x does not. ``pyproject.toml`` allows
    ``lerobot >= 0.3.3, < 0.8`` without 0.4 and 0.5 and the Dockerfile pins ``LEROBOT_VERSION``
    at build time, so this tells the v3_0 variant from the v2_1 variant,
    whose 0.3.x writes LeRobot 2.1 and cannot shard. ``main`` probes it
    before spawning any shard subprocess, so that a v2_1 image ignores
    ``shard_count`` instead of raising ``ModuleNotFoundError`` in
    ``_aggregate_shards`` after every shard has converted its events.
    ``_convert_with_episode_cache`` uses it to choose the v3.0 or v2.1 merge.
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
    so the merge cost is proportional to disk I/O, not to re-encoding video.
    :func:`_restat_numbering` then corrects the ``stats.json`` entries of
    the columns the merge renumbers.
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
    _restat_numbering(dataset_root)


def _restat_numbering(dataset_root: Path) -> None:
    """Recompute the ``stats.json`` entries of ``episode_index``, ``index`` and ``task_index`` after a merge.

    lerobot's merge renumbers these columns in the data and in each
    episode's stats. Its ``finalize_aggregation``, however, builds
    ``stats.json`` by combining the sources' own ``stats.json`` files, which
    describe each source's numbering from 0: a merge of one-episode
    datasets gets an ``episode_index`` maximum of 0.

    A writer that saves every episode into one dataset instead combines its
    stats so far with each episode's stats as it saves that episode
    (``aggregate_stats([stats, episode_stats])``). This function repeats
    those steps with lerobot's functions, over the merged data in episode
    order, so the three entries equal those that such a writer produces for
    the merged episodes. Every other entry is left as it is.

    lerobot-merge keeps a copy of this function for lerobot 0.5, whose
    ``load_info`` returns a dict; the two packages are deployed separately
    and cannot import each other.
    """
    from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
    from lerobot.datasets.io_utils import load_info, load_stats, write_stats

    columns = ["episode_index", "index", "task_index"]
    features = load_info(dataset_root).features
    stats = None
    # Each data file holds whole episodes, and the files are numbered in
    # episode order, so the sorted paths list the episodes in order.
    for path in sorted((dataset_root / "data").glob("*/*.parquet")):
        for _, episode in pd.read_parquet(path, columns=columns).groupby("episode_index", sort=True):
            episode_stats = compute_episode_stats({c: episode[c].to_numpy() for c in columns}, features)
            stats = episode_stats if stats is None else aggregate_stats([stats, episode_stats])
    write_stats({**load_stats(dataset_root), **stats}, dataset_root)


def _merge_entries(parts: list[dict], dataset_root: Path) -> None:
    """Merge the one-episode datasets at ``part["root"]``, in order, into ``dataset_root``.

    Numbers the episodes in the order of ``parts`` and packs their data and
    video files in that order up to lerobot's default size limits, as
    lerobot's writer packs the episodes of one conversion. Runs the loop of
    ``aggregate_datasets`` with lerobot's helpers for it
    (``validate_all_metadata``, ``merge_video_feature_info_for_aggregate``,
    ``aggregate_data``, ``update_meta_data`` and ``finalize_aggregation``,
    which lerobot does not document as public API), but not its video step
    or its episode metadata writes. ``aggregate_datasets`` appends each
    source video to the current packed file by rewriting that file, and
    appends each source's episode metadata by rewriting the metadata file.
    Here, :func:`_place_videos` first chooses the packed file of every
    source video and its offset in that file, so that each packed video
    file and the episode metadata are written once.
    """
    from lerobot.datasets import aggregate
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

    logger.info("Merging %d episodes into %s...", len(parts), dataset_root)
    with ThreadPoolExecutor(_MERGE_THREADS) as pool:
        metas = list(pool.map(_load_entry_metadata, [Path(p["root"]) for p in parts]))
        fps, robot_type, _ = aggregate.validate_all_metadata(metas)
        features = aggregate.merge_video_feature_info_for_aggregate(metas)
        video_keys = [k for k, f in features.items() if f["dtype"] == "video"]
        # The limits ``aggregate_datasets`` applies when given none.
        dst = LeRobotDatasetMetadata.create(
            repo_id=dataset_root.name,
            fps=fps,
            robot_type=robot_type,
            features=features,
            root=dataset_root,
            use_videos=bool(video_keys),
            chunks_size=aggregate.DEFAULT_CHUNK_SIZE,
            data_files_size_in_mb=aggregate.DEFAULT_DATA_FILE_SIZE_IN_MB,
            video_files_size_in_mb=aggregate.DEFAULT_VIDEO_FILE_SIZE_IN_MB,
        )
        tasks = pd.concat([m.tasks for m in metas]).index.unique()
        dst.tasks = pd.DataFrame({"task_index": range(len(tasks))}, index=pd.Index(tasks, name="task"))
        videos_idx, packed = _place_videos(metas, video_keys, dst.video_files_size_in_mb, dst.chunks_size, pool)
        _merge_data_and_metadata(dataset_root, metas, videos_idx, dst)
        staging = dataset_root / "_packing"
        staging.mkdir()
        with _temp_dir(staging):
            list(pool.map(lambda job: _write_video(dataset_root / job[0], job[1]), packed))
        staging.rmdir()


def _load_entry_metadata(root: Path) -> LeRobotDatasetMetadata:
    """Load the metadata of the one-episode dataset at ``root`` as a ``LeRobotDatasetMetadata``.

    Sets ``repo_id``, ``root`` and ``revision`` and loads ``info``,
    ``tasks``, ``stats`` and ``episodes`` without calling the constructor.
    The constructor loads the episode metadata as a Hugging Face
    ``Dataset``, which in pure Python takes most of the time loading a
    one-episode dataset's metadata takes, and drops its ``stats/`` columns.
    Pandas reads every column, so the merged episode metadata keeps each
    episode's stats.
    """
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.io_utils import load_info, load_stats, load_tasks

    meta = LeRobotDatasetMetadata.__new__(LeRobotDatasetMetadata)
    meta.repo_id, meta.root, meta.revision = root.name, root, None
    meta.info, meta.tasks, meta.stats = load_info(root), load_tasks(root), load_stats(root)
    meta.episodes = pd.concat(
        [pd.read_parquet(p) for p in sorted((root / "meta" / "episodes").glob("*/*.parquet"))], ignore_index=True,
    )
    return meta


def _merge_data_and_metadata(
    dataset_root: Path, metas: list[LeRobotDatasetMetadata], videos_idx: list[dict], dst: LeRobotDatasetMetadata,
) -> None:
    """Write the data, tasks, info, stats and episode metadata of the merge into ``dst``.

    Runs the loop of ``aggregate_datasets`` without its video step:
    ``aggregate_data`` writes each source's data to a file of its own, which
    :func:`_pack_data` then joins, and ``update_meta_data`` points each
    source's episodes at their packed video files and offsets in
    ``videos_idx``. The episode metadata is written once, after the data
    files are packed, and always to one file; lerobot's writer and
    ``aggregate_datasets`` start a new file when it reaches the data file
    size limit (100 MB, tens of thousands of episodes).
    """
    from lerobot.datasets import aggregate
    from lerobot.datasets.utils import DEFAULT_EPISODES_PATH

    data_idx: dict[str, Any] = {"chunk": 0, "file": 0}
    meta_idx = {"chunk": 0, "file": 0}
    episodes = []
    for src, src_videos_idx in zip(metas, videos_idx, strict=True):
        data_idx = aggregate.aggregate_data(
            src, dst, data_idx, dst.data_files_size_in_mb, dst.chunks_size, concatenate_data=False,
        )
        episodes.append(aggregate.update_meta_data(src.episodes.copy(), dst, meta_idx, data_idx, src_videos_idx))
        data_idx.pop("src_to_dst", None)
        dst.info.total_episodes += src.total_episodes
        dst.info.total_frames += src.total_frames
    episodes_df = pd.concat(episodes, ignore_index=True)
    _pack_data(dataset_root, episodes_df, dst)
    episodes_path = dataset_root / DEFAULT_EPISODES_PATH.format(chunk_index=0, file_index=0)
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    episodes_df.to_parquet(episodes_path)
    aggregate.finalize_aggregation(dst, metas)
    _restat_numbering(dataset_root)


@contextmanager
def _temp_dir(path: Path):
    """Make ``tempfile`` create its files under ``path``.

    ``concatenate_video_files`` writes each output to a temporary file and
    then moves it into place; under the dataset's own disk the move is a
    rename, and the file does not fill a small ``/tmp``.
    """
    previous = tempfile.tempdir
    tempfile.tempdir = str(path)
    try:
        yield
    finally:
        tempfile.tempdir = previous


def _pack_groups(
    sources: list[Any], sizes_mb: list[float], max_mb: float, chunks_size: int,
) -> list[tuple[tuple[int, int], list[Any]]]:
    """Split ``sources``, files in episode order, into packed files.

    Returns ``(destination, sources)`` pairs numbered from file 0 of chunk 0.
    A file is closed before the source that would bring its summed size to
    ``max_mb``, the rule lerobot's writer and ``aggregate_datasets`` apply.
    """
    from lerobot.datasets.utils import update_chunk_file_indices

    groups: list[tuple[tuple[int, int], list[Any]]] = []
    total = 0.0
    for source, size in zip(sources, sizes_mb, strict=True):
        if not groups or total + size >= max_mb:
            dst = update_chunk_file_indices(*groups[-1][0], chunks_size) if groups else (0, 0)
            groups.append((dst, []))
            total = 0.0
        groups[-1][1].append(source)
        total += size
    return groups


def _place_videos(
    metas: list[LeRobotDatasetMetadata], video_keys: list[str], max_mb: float, chunks_size: int,
    pool: ThreadPoolExecutor,
) -> tuple[list[dict], list[tuple[str, list[Path]]]]:
    """Choose the packed file of every source video file, and its offset in that file.

    Returns, per source dataset, the ``videos_idx`` that lerobot's
    ``update_meta_data`` takes to point the source's episodes at their packed
    files, and, per packed file, its path relative to the dataset root and
    its source files in order. A source's frames start at the summed
    durations of the sources before it in its packed file, the duration
    lerobot's writer uses to place an episode.
    """
    from lerobot.datasets.io_utils import get_file_size_in_mb
    from lerobot.datasets.utils import DEFAULT_VIDEO_PATH
    from lerobot.datasets.video_utils import get_video_duration_in_s

    videos_idx: list[dict] = [{} for _ in metas]
    packed: list[tuple[str, list[Path]]] = []
    for key in video_keys:
        sources = [
            (i, int(chunk), int(file))
            for i, meta in enumerate(metas)
            for chunk, file in sorted(set(zip(
                meta.episodes[f"videos/{key}/chunk_index"], meta.episodes[f"videos/{key}/file_index"], strict=True,
            )))
        ]
        paths = [
            metas[i].root / DEFAULT_VIDEO_PATH.format(video_key=key, chunk_index=chunk, file_index=file)
            for i, chunk, file in sources
        ]
        durations = list(pool.map(get_video_duration_in_s, paths))
        for idx in videos_idx:
            # ``update_meta_data`` reads ``chunk`` and ``file`` as the default for a
            # file missing from ``src_to_dst``, even though every file is in it.
            idx[key] = {"chunk": 0, "file": 0, "src_to_dst": {}, "src_to_offset": {}}
        for dst, group in _pack_groups(
            list(range(len(sources))), [get_file_size_in_mb(p) for p in paths], max_mb, chunks_size,
        ):
            elapsed = 0.0
            for s in group:
                i, chunk, file = sources[s]
                videos_idx[i][key]["src_to_dst"][(chunk, file)] = dst
                videos_idx[i][key]["src_to_offset"][(chunk, file)] = elapsed
                elapsed += durations[s]
            rel = DEFAULT_VIDEO_PATH.format(video_key=key, chunk_index=dst[0], file_index=dst[1])
            packed.append((rel, [paths[s] for s in group]))
    return videos_idx, packed


def _write_video(out: Path, sources: list[Path]) -> None:
    """Write the packed video file ``out`` from ``sources``, stream-copying their packets.

    A file packed alone is hard-linked, or copied where the link fails.
    """
    from lerobot.datasets.video_utils import concatenate_video_files

    out.parent.mkdir(parents=True, exist_ok=True)
    if len(sources) > 1:
        concatenate_video_files(sources, out)
        return
    try:
        os.link(sources[0], out)
    except OSError:
        shutil.copyfile(sources[0], out)


def _pack_data(root: Path, episodes: pd.DataFrame, meta: LeRobotDatasetMetadata) -> None:
    """Join the data files of ``root``, in episode order, up to the dataset's size limit.

    Sizes are uncompressed parquet sizes, which ``aggregate_datasets`` also
    compares. Rewrites ``data/{chunk,file}_index`` of the episode metadata
    ``episodes`` to match. A file packed alone is moved unchanged.
    """
    from lerobot.datasets.io_utils import (
        get_parquet_file_size_in_mb,
        to_parquet_one_row_group_per_episode,
    )

    def path(location: tuple[int, int]) -> Path:
        return root / meta.data_path.format(chunk_index=location[0], file_index=location[1])

    pairs = zip(episodes["data/chunk_index"], episodes["data/file_index"], strict=True)
    located = [(int(c), int(f)) for c, f in pairs]
    sources = list(dict.fromkeys(located))
    staging = root / "_packing" / "data"
    location: dict[tuple[int, int], tuple[int, int]] = {}
    for dst, group in _pack_groups(
        sources, [get_parquet_file_size_in_mb(path(s)) for s in sources],
        meta.data_files_size_in_mb, meta.chunks_size,
    ):
        out = staging / path(dst).relative_to(root / "data")
        out.parent.mkdir(parents=True, exist_ok=True)
        if len(group) == 1:
            path(group[0]).rename(out)
        else:
            to_parquet_one_row_group_per_episode(
                pd.concat([pd.read_parquet(path(s)) for s in group], ignore_index=True), out,
            )
        location.update(dict.fromkeys(group, dst))
    shutil.rmtree(root / "data")
    staging.rename(root / "data")
    staging.parent.rmdir()
    episodes["data/chunk_index"] = [location[s][0] for s in located]
    episodes["data/file_index"] = [location[s][1] for s in located]


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


def _convert_with_episode_cache(
    *,
    cache: EpisodeCache,
    recipe: str,
    matched: list,
    matched_events: list,
    topics_by_ds: dict,
    contracts_by_ds: dict,
    buffer_ns: int,
    action_lead_ns: int,
    dataset_root: Path,
    workdir: Path,
    max_parallel: int,
    accumulator: _DrainAccum,
    **spawn_kwargs: Any,
) -> None:
    """Build ``dataset_root`` from cached one-episode datasets plus new ones.

    Keys every matched event and downloads the entries the cache already
    holds. The remaining distinct keys are split across at most
    ``max_parallel`` shard processes, each converting its events one after
    another into one-episode datasets. Every entry is uploaded as soon as
    its shard reports it, so it stays in the cache when a later event or
    shard fails the export. Then merges every
    entry in ``matched`` order, the order a single-writer conversion writes.
    Fills ``accumulator`` like the other paths, adding to each episode its
    ``episode_key`` and whether it was a ``cache_hit``.

    ``spawn_kwargs`` are forwarded to :func:`_spawn_shards`.
    """
    keys = [
        episode_key(
            event_id=getattr(ev, "event_id", None),
            src_ds_id=src_ds_id,
            start_time_ns=int(ev.start_time),
            end_time_ns=int(ev.end_time),
            task_label=task_label,
            buffer_ns=buffer_ns,
            action_lead_ns=action_lead_ns,
            contract=contracts_by_ds[src_ds_id],
            topics=topics_by_ds[src_ds_id],
        )
        for (_, src_ds_id, task_label, _), ev in zip(matched, matched_events, strict=True)
    ]
    hits = cache.fetch(recipe, sorted(set(keys)), workdir)

    # One conversion per distinct key, at the slot where it first appears.
    miss_slots: dict[str, int] = {}
    for slot, key in enumerate(keys):
        if key not in hits:
            miss_slots.setdefault(key, slot)
    converted: set[str] = set()
    if miss_slots:
        slots = list(miss_slots.values())
        key_by_event_idx = {matched[s][0]: k for k, s in miss_slots.items()}

        def _store(reports: list[dict]) -> None:
            keys_done = []
            for report in reports:
                key = key_by_event_idx[report["event_idx"]]
                entry = workdir / entry_path(recipe, key)
                entry.parent.mkdir(parents=True, exist_ok=True)
                Path(report["root"]).rename(entry)
                # The index within the shard; the merge numbers the episodes again.
                summary = {k: v for k, v in report["episode"].items() if k != "episode_index"}
                (entry / MARKER_NAME).write_text(json.dumps(summary, default=str))
                keys_done.append(key)
            cache.upload(recipe, workdir, keys_done)
            converted.update(keys_done)

        chunks_drain, chunks_tasks = _partition_shards(
            [matched[s] for s in slots],
            [matched_events[s] for s in slots],
            shard_count=max_parallel,
            buffer_ns=buffer_ns,
            action_lead_ns=action_lead_ns,
        )
        results = _spawn_shards(
            chunks_drain=chunks_drain,
            chunks_tasks=chunks_tasks,
            shard_workdir=workdir / "_shards",
            contracts_by_ds=contracts_by_ds,
            on_episodes=_store,
            **spawn_kwargs,
        )
        for r in results:
            accumulator.skipped_events.extend(r.get("skipped_events", []))

    order = [s for s, key in enumerate(keys) if key in hits or key in converted]
    if not order:
        raise RuntimeError(
            "Conversion produced 0 episodes; nothing was written. "
            f"Dropped events: {_summarize_skip_reasons(accumulator.skipped_events)}"
        )
    # lerobot 0.3.x writes v2.1, one file per episode, and has no merge API.
    merge = _merge_entries if _shard_path_supported() else merge_v21_entries
    merge(
        [{"repo_id": keys[s], "root": str(workdir / entry_path(recipe, keys[s]))} for s in order],
        dataset_root,
    )

    for episode_index, slot in enumerate(order):
        key = keys[slot]
        summary = episode_summary(workdir / entry_path(recipe, key))
        accumulator.episode_to_event.append({
            "episode_index": episode_index, **summary, "episode_key": key, "cache_hit": key in hits,
        })
        accumulator.total_frames += int(summary["n_frames"])
        accumulator.per_dataset_counts[summary["source_dataset_id"]] += 1
    accumulator.total_episodes = len(order)
    logger.info(
        "Episode cache: %d episodes, %d hits, %d converted (recipe %s).",
        len(order), sum(keys[s] in hits for s in order), len(converted), recipe,
    )
    # The runtime uploads everything left under the output directory, and
    # ``workdir`` lies inside it; only ``dataset_root`` should be uploaded.
    try:
        shutil.rmtree(workdir)
    except OSError as e:
        logger.warning("Could not remove episode cache workdir %s: %s", workdir, e)


def _write_lineage(dataset_root: Path, run: dict, episodes: list[dict]) -> None:
    """Record in ``dataset_root/meta`` where the dataset and each of its episodes came from.

    ``run`` describes the export and must have a ``recipe_hash``.
    ``episodes`` are the manifest's ``episode_to_event`` entries, numbered
    as in the dataset.

    Writes ``run`` to ``meta/roboto_lineage.json`` with, per episode, its
    index, its event or its session (the other id is null), source dataset,
    bounds, whether it came from the episode cache and, when the cache was
    on, its ``episode_key``. Then appends a
    ``roboto/<field>`` column for each field of ``_LINEAGE_COLUMNS`` to every
    episode metadata file, ``meta/episodes/*/*.parquet``. A v2.1 dataset
    keeps its episode metadata in ``meta/episodes.jsonl`` instead, so it
    gets only the JSON file.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [
        {
            "episode_index": int(e["episode_index"]),
            "event_id": e["event_id"],
            "session_id": e.get("session_id"),
            "source_dataset_id": e["source_dataset_id"],
            "start_time_ns": int(e["start_time_ns"]),
            "end_time_ns": int(e["end_time_ns"]),
            "cache_hit": bool(e.get("cache_hit", False)),
            **({"episode_key": e["episode_key"]} if "episode_key" in e else {}),
        }
        for e in episodes
    ]
    meta = dataset_root / "meta"
    (meta / _LINEAGE_NAME).write_text(json.dumps({**run, "episodes": rows}, indent=2))
    by_index = {r["episode_index"]: {**r, "recipe_hash": run["recipe_hash"]} for r in rows}
    for path in sorted((meta / "episodes").glob("*/*.parquet")):
        table = pq.read_table(path)
        indices = table.column("episode_index").to_pylist()
        for field, dtype in _LINEAGE_COLUMNS.items():
            values = [by_index[i][field] for i in indices]
            table = table.append_column(f"roboto/{field}", pa.array(values, type=dtype))
        pq.write_table(table, path)


def _lance_convert(root: str, out: str) -> int:
    """Convert the LeRobot dataset at ``root`` to the Lance layout at ``out``.

    Runs in the process :func:`_write_lance` starts for it, and returns that
    process's peak resident memory in KiB (``ru_maxrss`` on Linux).
    """
    from lerobot_lancedb.convert import convert

    convert(Path(out), root=Path(root))
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def _write_lance(dataset_root: Path, out: Path, *, keep_lerobot: bool) -> None:
    """Write the Lance layout of the LeRobot dataset at ``dataset_root`` to ``out``.

    Then removes ``dataset_root``, unless ``keep_lerobot``. The converter,
    ``lerobot_lancedb.convert.convert``, copies ``meta/`` (and the lineage
    in it) and stores the frames in one Lance table and the bytes of every
    video file, unchanged, in another. It runs in a spawned process of its
    own, so that its memory returns to the system when the process exits
    and its peak can be logged. A converter error propagates and leaves
    ``dataset_root`` in place.
    """
    started = time.monotonic()
    with ProcessPoolExecutor(1, mp_context=mp.get_context("spawn")) as pool:
        peak_kib = pool.submit(_lance_convert, str(dataset_root), str(out)).result()
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    logger.info(
        "Lance dataset written to %s in %.0f s: %.2f GB; converter peak memory %.2f GB.",
        out, time.monotonic() - started, size / 1e9, peak_kib * 1024 / 1e9,
    )
    if not keep_lerobot:
        shutil.rmtree(dataset_root)


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

    Loads the events or sessions of the input Roboto Collection, then
    writes one combined LeRobot dataset where each episode corresponds to
    one event or one session.
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

    output_format = _resolve_output_format(context.get_optional_parameter("output_format"))
    contract, contract_rel_path, contract_sha256 = _load_contract_from_context(context)
    episode_cache = _open_episode_cache(context)

    fps = int(contract.fps)
    time_step_ns = fps_to_time_step_ns(contract.fps)
    buffer_ns = _compute_buffer_ns(contract)
    action_lead_ns = contract.action_lead_steps * time_step_ns

    collection = roboto.Collection.from_id(collection_id)
    collection_version = collection.record.version
    collection_resource_type = collection.record.resource_type.value
    # Every event, or every session, of the collection becomes one episode;
    # see ``episode_sources`` for how each kind is windowed and which files
    # its topics are read from. ``prepared`` is keyed by each source's
    # ``scope_key`` (dataset id for events, session id for sessions).
    sources, prepared = resolve_episode_sources(collection, contract)
    dedup_by_dataset: dict[str, list[dict]] = {
        key: scope.dedup_groups for key, scope in prepared.items() if scope.dedup_groups
    }
    # Earliest source (chronologically) — its DataCollection is reused for
    # feature discovery and (in the single-writer path) inlined as the first
    # episode.
    first_event = sources[0]
    first_ds_id = first_event.scope_key

    # Write under output_dir/<invocation_id>/ so the runtime's auto-upload
    # lands it at <invocation_dataset>/<invocation_id>/..., keeping repeat
    # invocations on the same dataset from clobbering each other.
    subfolder = context.output_dir / context.invocation_id
    dataset_root = subfolder / "combined"
    dataset_root.parent.mkdir(parents=True, exist_ok=True)

    first_resolved = prepared[first_ds_id].contract
    first_topics = prepared[first_ds_id].topics

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

    # Index every source in chronological order; the index becomes the
    # ``event_idx`` we ship to the pool and drain by below.
    #
    # ``task_label`` here is the *raw* event metadata task only (``None``
    # when absent/empty) — final resolution against the contract's
    # ``tasks:`` stream (which needs the per-episode ``DataCollection``,
    # not yet built at this point) happens later via
    # ``resolve_task_label``, in ``_worker_build_episode`` for pool tasks
    # and inline for the preloaded episode below.
    matched: list[tuple[int, str, str | None, int]] = []  # (event_idx, src_ds_id, task_label, num_frames)
    matched_events: list = []
    for event in sources:
        src_ds_id = event.scope_key
        raw_task_meta = (event.metadata or {}).get("task")
        task_label = str(raw_task_meta) if raw_task_meta else None
        num_frames = int((event.end_time - event.start_time) / time_step_ns) + 1
        event_idx = len(matched)
        matched.append((event_idx, src_ds_id, task_label, num_frames))
        matched_events.append(event)

    # Materialised _DeferredFrame entries on the main process need the video
    # spec for decoder dispatch + resize. Videos pass through
    # ``resolve_role_bindings`` unchanged, so a single mapping covers every
    # resolved contract.
    video_specs_by_key = {v.key: v for v in contract.videos}

    # Build the per-dataset arguments of ``_worker_init``: the records of the
    # contract topics, from which each worker builds its own Topic objects
    # without an API call, and the resolved contract.
    descriptors_by_ds: dict[str, dict[str, list]] = {}
    contracts_by_ds: dict[str, Contract] = {}
    for key, scope in prepared.items():
        descriptors_by_ds[key] = topic_descriptors_from_topics(scope.topics)
        contracts_by_ds[key] = scope.contract

    shard_count = _resolve_shard_count(
        context.get_optional_parameter("shard_count"),
    )
    if shard_count > 1 and episode_cache is None and not _shard_path_supported():
        # A v3-only knob on a lerobot 0.3.x image: accepted and ignored,
        # never fatal. With the episode cache, the shard count bounds the
        # processes that convert cache misses on either variant, so it is
        # kept. action.json is shared by both deployed variants and
        # defaults shard_count to 3 for v3_0's benefit, so failing here
        # would break every default invocation of v2_1.
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
            "shard_count=%d ignored: sharded writing requires lerobot 0.6 or later "
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
    recipe = recipe_hash(
        contract_path=context.input_dir / contract_rel_path,
        features=features,
        streaming_encoding=_STREAMING_ENCODING,
    )

    if episode_cache is not None:
        # Every episode is written by a shard process, so the
        # preloaded first event is only needed for feature discovery.
        del first_data
        _convert_with_episode_cache(
            cache=episode_cache,
            recipe=recipe,
            matched=matched,
            matched_events=matched_events,
            topics_by_ds={key: scope.topics for key, scope in prepared.items()},
            contracts_by_ds=contracts_by_ds,
            buffer_ns=buffer_ns,
            action_lead_ns=action_lead_ns,
            dataset_root=dataset_root,
            workdir=subfolder / "_episode_cache",
            max_parallel=shard_count,
            accumulator=accumulator,
            fps=fps,
            features=features,
            robot_type=contract.robot_type or "unknown",
            encoder_threads=encoder_threads,
            pool_size=pool_size,
            video_specs_by_key=video_specs_by_key,
            descriptors_by_ds=descriptors_by_ds,
            log_level=context.log_level,
        )
    elif shard_count == 1:
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
        preloaded_episode = _preload_episode(
            contract=first_resolved,
            episode_data=first_data,
            event=preloaded_event,
            event_idx=preloaded_event_idx,
            metadata_task=preloaded_task,
            ref_ts=preloaded_ref_ts,
        )
        if preloaded_episode is None:
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
    # accumulator so the rest of ``main`` stays untouched. The drain keys
    # every entry and count by ``scope_key``, which for a session is the
    # session id; relabel those with the session's real dataset ids.
    total_frames = accumulator.total_frames
    total_episodes = accumulator.total_episodes
    sources_by_key = {s.scope_key: s for s in sources}
    episode_to_event = [
        annotate_manifest_entry(e, sources_by_key) for e in accumulator.episode_to_event
    ]
    accumulator.skipped_events = [
        annotate_manifest_entry(e, sources_by_key) for e in accumulator.skipped_events
    ]
    per_dataset_counts: dict[str, int] = defaultdict(int)
    for e in episode_to_event:
        per_dataset_counts[e["source_dataset_id"]] += 1
    source_dataset_ids = sorted({d for s in sources for d in s.dataset_ids})
    subtasks_summary = apply_session_subtasks(
        dataset_root, sources, episode_to_event, contract.fps,
    )

    # Archive the contract bytes so the run survives later edits to the source.
    archived_contract_filename = "contract.yaml"
    archived_contract_path = subfolder / archived_contract_filename
    shutil.copyfile(context.input_dir / contract_rel_path, archived_contract_path)
    logger.info("Contract archived: %s", archived_contract_path)

    lerobot_version = _pkg_version("lerobot")
    _write_lineage(
        dataset_root,
        run={
            "invocation_id": context.invocation_id,
            "collection_id": collection_id,
            "collection_version": collection_version,
            "collection_resource_type": collection_resource_type,
            "contract": {"name": contract.name, "version": contract.version, "sha256": contract_sha256},
            "lerobot_version": lerobot_version,
            "recipe_hash": recipe,
        },
        episodes=episode_to_event,
    )
    # After ``_write_lineage``, so the Lance layout's copy of ``meta/`` has the lineage.
    if output_format != "lerobot":
        _write_lance(dataset_root, subfolder / "lance", keep_lerobot=output_format == "both")

    # Flatten per-dataset dedup reports; each entry carries enough context
    # (dataset_id + per-file file_ids/paths) to find the source upload.
    dedup_records: list[dict] = []
    dropped_file_ids: set[str] = set()
    for ds_id, groups in dedup_by_dataset.items():
        for g in groups:
            dedup_records.append(annotate_manifest_entry(
                {"source_dataset_id": ds_id, **g}, sources_by_key,
            ))
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
        "collection_resource_type": collection_resource_type,
        "source_datasets": source_dataset_ids,
        "episodes_per_dataset": dict(per_dataset_counts),
        "episode_to_event": episode_to_event,
        "subtasks": subtasks_summary,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "dedup": dedup_records,
        "dedup_summary": dedup_summary,
        "skipped_events": skipped_events_records,
        "skipped_events_summary": skipped_events_summary,
        "codebase": {
            "lerobot": lerobot_version,
        },
        "output_format": output_format,
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
        "collection_resource_type": collection_resource_type,
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
            "collection_resource_type": collection_resource_type,
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "lerobot_version": lerobot_version,
            "had_dedup": bool(dedup_records),
            "dedup_groups": len(dedup_records),
        })
        .build()
    )

    output_tags = parse_tags(context.get_optional_parameter("output_tags"))
    if output_tags and not context.is_dry_run:
        publish_tagged_output(output_dataset, context.output_dir, subfolder, output_tags)

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
        total_episodes, total_frames, len(source_dataset_ids),
        context.invocation_id, output_dataset_id,
    )
