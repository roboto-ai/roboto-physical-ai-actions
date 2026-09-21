from __future__ import annotations

import dataclasses
import logging
import pathlib
from typing import Any

import numpy as np
import pandas as pd
import roboto

from .._vendored.roboto_to_lerobot.contract_utils import (
    Contract,
    DataCollection,
    _compute_buffer_ns,
    collect_topics_from_dataset,
    fps_to_time_step_ns,
    load_contract,
)
from .._vendored.roboto_to_lerobot.lerobot import generate_frames
from ..core.types import EpisodeData, FeatureSpec, SourceDescriptor

logger = logging.getLogger(__name__)


def load_pre_conversion_episodes(
    dataset: roboto.Dataset,
    dataset_id: str,
    contract_path: pathlib.Path,
    collection_id: str,
    roboto_client: Any | None,
) -> tuple[list[EpisodeData], SourceDescriptor, dict[str, Any]]:
    """Audit signal == post-conversion signal.

    Events are pulled from the Roboto Collection identified by
    ``collection_id`` (same input contract as the ``roboto-to-lerobot``
    conversion action) and may span multiple source datasets; topics are
    prepared per source dataset and each event is routed to its own.
    For every event, drive the exact same pipeline the converter uses to
    emit LeRobot frames (``generate_frames``): fixed-fps reference timeline,
    per-topic pre/post transforms, AlignSpec merge onto the timeline,
    ``action_lead_steps`` timestamp shift, NaN-row drop. The per-frame
    state / action arrays are then stacked into ``EpisodeData``. Metrics
    therefore see *exactly* what the LeRobotDataset would contain after
    conversion — modulo the video/mp4 branch, which is stripped from the
    contract because the Roboto SDK currently serves degraded JPEG previews
    for Image/CompressedImage topics, so frame-level visual fidelity is not
    auditable here.
    """
    contract = load_contract(contract_path)
    logger.info("Loaded contract: %s (version %d)", contract.name, contract.version)

    # Audit mode: drop videos from the contract so DataCollection skips image
    # fetches entirely and generate_frames' video branch is a no-op. Scalar
    # state/action math (transforms, alignment, fps) is identical to the
    # conversion pipeline.
    audit_contract = dataclasses.replace(contract, videos=[])
    if contract.videos:
        logger.info(
            "Audit mode: skipping %d video topic(s) — scalar state/action only",
            len(contract.videos),
        )

    fps = int(audit_contract.fps)
    time_step_ns = fps_to_time_step_ns(audit_contract.fps)
    # NB: pass the *original* contract (with videos), not audit_contract.
    # _compute_buffer_ns scans align specs across observations, videos, and
    # actions. Stripping videos here would shrink the buffer when a video
    # spec carries the binding tolerance (or the tolerance_ms == 0 sentinel
    # that forces the 1 s fallback) and diverge from the converter's window.
    buffer_ns = _compute_buffer_ns(contract)
    action_lead_ns = audit_contract.action_lead_steps * time_step_ns

    # ``Collection.from_id`` defaults to ``content_mode=Full``, which makes
    # the server hydrate every event resource and return full EventRecord
    # payloads in ``record.resources``. Building ``Event`` instances from
    # those records avoids an N-extra-RTT round trip per invocation that
    # ``collection.events`` → ``Event.from_id(eid)`` would otherwise pay.
    # Mirrors the converter's collection-load pattern verbatim so the same
    # collection_id can drive both actions.
    collection = roboto.Collection.from_id(
        collection_id, roboto_client=roboto_client
    )
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

    # Cross-dataset routing mirrors roboto-to-lerobot/main.py: discover the
    # distinct dataset_ids referenced by the collection's events, prep each
    # source dataset's topics once, then route each event to its source
    # dataset's topics inside the per-event loop. Events with no
    # ``dataset_ids`` cannot be audited (no topics to bind to) and are
    # dropped with a log line.
    all_events = sorted(
        (
            roboto.Event(
                roboto.EventRecord.model_validate(r) if isinstance(r, dict) else r,
            )
            for r in hydrated_event_resources
        ),
        key=lambda e: e.start_time,
    )
    events: list[roboto.Event] = []
    distinct_ds_ids: set[str] = set()
    no_dataset_event_ids: list[str] = []
    for event in all_events:
        ds_ids = event.dataset_ids()
        if not ds_ids:
            no_dataset_event_ids.append(getattr(event, "event_id", "?"))
            continue
        events.append(event)
        distinct_ds_ids.update(ds_ids)
    if no_dataset_event_ids:
        logger.info(
            "Collection %s (version %d): %d of %d event(s) have no dataset "
            "association and will be skipped.",
            collection_id, collection_version,
            len(no_dataset_event_ids), len(all_events),
        )
    if not events:
        raise ValueError(
            f"Collection {collection_id!r} (version {collection_version}) "
            f"has no events with dataset associations."
        )

    # Prep each source dataset's topics once, keyed by ds_id, so the per-event
    # loop can look them up by ``event.dataset_ids()[0]``. The invocation
    # ``dataset`` is reused when it's one of the source datasets; others are
    # fetched via ``Dataset.from_id``.
    source_dataset_ids = sorted(distinct_ds_ids)
    prepared: dict[str, dict[str, list[roboto.Topic]]] = {}
    for ds_id in source_dataset_ids:
        ds = dataset if ds_id == dataset_id else roboto.Dataset.from_id(
            ds_id, roboto_client=roboto_client,
        )
        topics, _ = collect_topics_from_dataset(ds, audit_contract)
        prepared[ds_id] = topics
    logger.info(
        "Loaded %d events from collection %s (version %d) across %d source "
        "dataset(s): %s",
        len(events), collection_id, collection_version,
        len(prepared), source_dataset_ids,
    )

    episodes: list[EpisodeData] = []
    state_spec: FeatureSpec | None = None
    action_spec: FeatureSpec | None = None
    # Per-event failures are recorded and surfaced via metadata; the audit
    # then continues with the remaining events. The converter applies the
    # same soft-drop policy on its side (manifest entry per failed event),
    # so Mode A's "events analyzed" count matches Mode B's input count
    # while Mode B's output naturally omits the failed ones.
    failed_events: list[dict[str, Any]] = []

    for idx, event in enumerate(events):
        src_ds_id = event.dataset_ids()[0]
        logger.info(
            "Loading event %d/%d (dataset=%s) for audit",
            idx + 1, len(events), src_ds_id,
        )
        try:
            # Window padding mirrors the ``DataCollection(...)`` calls in the
            # converter (roboto_to_lerobot/main.py and event_worker.py). The
            # ``+ action_lead_ns`` tail is load-bearing: without it the final
            # ``action_lead_steps`` frames have no future-action samples to
            # pull from during merge and get dropped by NaN-row filtering.
            data = DataCollection(
                contract=audit_contract,
                topics=prepared[src_ds_id],
                start_time_ns=event.start_time - buffer_ns,
                end_time_ns=event.end_time + buffer_ns + action_lead_ns,
            )

            if state_spec is None:
                state_spec = _resolve_concat_spec(
                    audit_contract, data.resolved_features, "observation.state", is_action=False
                )
            if action_spec is None:
                action_spec = _resolve_concat_spec(
                    audit_contract, data.resolved_features, "action", is_action=True
                )

            # Reference-timeline construction mirrors the converter's per-event
            # loop (roboto_to_lerobot/main.py builds ``num_frames`` and
            # event_worker.py builds the ``reference_timestamps`` series). Drives
            # fps-resample alignment inside ``generate_frames``; must stay
            # byte-for-byte identical to the converter's formula or Mode A frame
            # counts diverge from Mode B.
            num_frames = int((event.end_time - event.start_time) / time_step_ns) + 1
            reference_timestamps = pd.Series(
                np.arange(num_frames, dtype=np.int64) * time_step_ns + event.start_time,
                name="timestamp",
            )

            state_rows: list[np.ndarray] = []
            action_rows: list[np.ndarray] = []
            for frame in generate_frames(
                audit_contract,
                data,
                reference_timestamps,
                task=(event.metadata or {}).get("task", "audit"),
            ):
                if "observation.state" in frame:
                    state_rows.append(np.asarray(frame["observation.state"], dtype=float))
                if "action" in frame:
                    action_rows.append(np.asarray(frame["action"], dtype=float))

            if not state_rows and not action_rows:
                logger.warning("Event %d yielded no state or action samples; skipping", idx)
                failed_events.append({
                    "episode_index": idx,
                    "event_id": getattr(event, "event_id", None),
                    "error_type": "EmptyEvent",
                    "error": "generate_frames produced no state or action rows",
                })
                continue

            # Cast to float32 to match LeRobot's parquet storage precision. Without
            # this, Mode A metrics see float64 values where the conversion pipeline
            # would have quantized to float32 — a "dead" sensor channel shows std=0
            # here but std≈1e-7 in Mode B, so flags like `zero_variance_dim` would
            # fire in Mode A and miss in Mode B on the same data.
            state_arr = (
                np.stack(state_rows, axis=0).astype(np.float32) if state_rows else None
            )
            action_arr = (
                np.stack(action_rows, axis=0).astype(np.float32) if action_rows else None
            )

            n_frames = (
                state_arr.shape[0] if state_arr is not None else action_arr.shape[0]  # type: ignore[union-attr]
            )

            # Timestamps relative to event start, in seconds — generate_frames
            # drops NaN rows so exact row count may be < num_frames. We reconstruct
            # a best-effort timestamp vector at the fixed cadence for the emitted
            # row count; metrics use this for velocity/jerk only and tolerate the
            # coarse approximation.
            timestamps_s = (np.arange(n_frames, dtype=np.float64) / float(fps))

            episodes.append(
                EpisodeData(
                    episode_index=idx,
                    fps=float(audit_contract.fps),
                    n_frames=int(n_frames),
                    state=state_arr,
                    state_spec=state_spec,
                    action=action_arr,
                    action_spec=action_spec,
                    timestamps=timestamps_s,
                )
            )
        except Exception as exc:
            # Broad except is deliberate: a malformed event, a missing
            # topic upload, a decoder failure, or an alignment edge case
            # should not abort the audit of the remaining events. We log
            # the full traceback at exception level (kept off the warning
            # line itself to keep the operator-facing log scannable) and
            # capture the error in metadata so the report can surface a
            # per-event failure list.
            event_id = getattr(event, "event_id", None)
            logger.warning(
                "Event %d/%d (id=%s) failed during pre-conversion audit: %s: %s — skipping",
                idx + 1, len(events), event_id, type(exc).__name__, exc,
            )
            logger.debug("Traceback for event %d:", idx, exc_info=True)
            failed_events.append({
                "episode_index": idx,
                "event_id": event_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            continue

    if not episodes:
        raise ValueError(
            f"Pre-conversion audit failed: 0 of {len(events)} event(s) "
            f"yielded usable data. See logs for per-event errors."
        )
    if failed_events:
        logger.warning(
            "Pre-conversion audit: %d of %d event(s) failed; continuing with %d episode(s).",
            len(failed_events), len(events), len(episodes),
        )

    source = SourceDescriptor(
        kind="roboto_events",
        identifier=f"{dataset_id}:{collection_id}@v{collection_version}",
        detail={
            "contract_name": contract.name,
            "contract_version": contract.version,
            "contract_fps": contract.fps,
            "action_lead_steps": contract.action_lead_steps,
            "collection_id": collection_id,
            "collection_version": collection_version,
            "n_events": len(events),
            "n_events_failed": len(failed_events),
            "n_events_skipped_no_dataset": len(no_dataset_event_ids),
            "n_source_datasets": len(prepared),
            "source_dataset_ids": source_dataset_ids,
        },
    )
    metadata = {
        "state_spec": state_spec,
        "action_spec": action_spec,
        "contract": contract,
        "contract_path": contract_path,
        # Preserve the ordered events so the orchestrator can map episode_index
        # (assigned from `enumerate(events)` above) back to the originating
        # event for downstream actions like tag writes. Failed events keep
        # their slot in this list — `failed_events` below carries the gaps.
        "events": events,
        # Per-event failure records: {episode_index, event_id, error_type,
        # error}. Empty list when all events succeeded. The report layer
        # surfaces this; the orchestrator should skip tag writes for any
        # episode_index that appears here.
        "failed_events": failed_events,
    }
    return episodes, source, metadata


def _resolve_concat_spec(
    contract: Contract,
    resolved_features: dict[str, tuple[int, list[str]]],
    base_key: str,
    is_action: bool,
) -> FeatureSpec | None:
    specs = contract.actions if is_action else contract.observations
    total_dim = 0
    names: list[str] = []
    for spec in specs:
        if spec.key != base_key:
            continue
        resolved = resolved_features.get(spec.unique_key)
        if resolved is None:
            continue
        dim, lr_names = resolved
        total_dim += dim
        if lr_names:
            names.extend(lr_names)
        else:
            names.extend([f"{spec.unique_key}.{i}" for i in range(dim)])
    if total_dim == 0:
        return None
    return FeatureSpec(key=base_key, names=names, dim=total_dim)
