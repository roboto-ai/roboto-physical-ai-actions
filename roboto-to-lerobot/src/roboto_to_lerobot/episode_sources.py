"""Resolve an input collection into the per-episode units the converter drains.

The action accepts two kinds of Roboto Collection, and every resource in
either becomes exactly one LeRobot episode:

* **Events** (``resource_type='event'``): the episode is the event's time
  window, read from the topics of the event's first associated dataset.
  Events sharing a dataset share one prepared topic set, keyed by
  ``dataset_id``.
* **Sessions** (``resource_type='session'``): the episode is the session's
  time window, read from the topics of the session's own files only, which
  may come from several datasets. Each session has its own prepared topic
  set, keyed by ``session_id``.

Both resolve to the same shape: a chronological list of
:class:`EpisodeSource` and a ``{scope_key: PreparedScope}`` mapping. From
there on the pipeline does not care which kind of collection it was given;
``EpisodeSource.scope_key`` is the opaque key that flows through
``EventTask.src_ds_id`` to pick the worker's contract + topics.

Session data is read through the file-scoped ``roboto.Topic`` API, the same
path events use, rather than through ``Session.list_topics``. That keeps the
two collection kinds on one fetch/alignment path, and it keeps working for
sessions whose files were ingested before the platform started writing the
normalized topic rows a session topic listing reads (those sessions list no
topics and can have no aggregate time bounds; see :func:`_session_window`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple

import roboto

from .contract_utils import (
    Contract,
    collect_topics_from_dataset,
    collect_topics_from_files,
    resolve_role_bindings,
    resolve_role_bindings_for_files,
)
from .logger import logger

SourceType = Literal["event", "session"]


@dataclass(frozen=True)
class EpisodeSource:
    """One collection resource that becomes one episode.

    ``scope_key`` selects the prepared contract + topics the episode is
    read from: the dataset id for an event, the session id for a session.
    ``dataset_ids`` are the datasets the episode's data comes from, for
    provenance only.
    """

    source_type: SourceType
    source_id: str
    scope_key: str
    dataset_ids: tuple[str, ...]
    start_time: int
    end_time: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def event_id(self) -> str:
        """The id the drain loop logs and records in its ``event_id`` slot.

        The source's id for both kinds; :func:`annotate_manifest_entry`
        moves a session's id to ``session_id`` when the manifest is written.
        """
        return self.source_id


class PreparedScope(NamedTuple):
    """Role-resolved contract and matched topics for one ``scope_key``."""

    contract: Contract
    topics: dict[str, list[roboto.Topic]]
    dedup_groups: list[dict]


class ResolvedSources(NamedTuple):
    sources: list[EpisodeSource]
    prepared: dict[str, PreparedScope]


def resolve_episode_sources(
    collection: roboto.Collection,
    contract: Contract,
) -> ResolvedSources:
    """Dispatch on the collection's resource type."""
    resource_type = collection.record.resource_type
    if resource_type == roboto.CollectionResourceType.Event:
        return _resolve_event_sources(collection, contract)
    if resource_type == roboto.CollectionResourceType.Session:
        return _resolve_session_sources(collection, contract)
    raise ValueError(
        f"collection_id {collection.collection_id!r} has resource_type="
        f"{resource_type.value!r}; this action requires a collection of "
        f"events (resource_type='event') or sessions (resource_type='session')."
    )


def _hydrated_resources(
    collection: roboto.Collection,
    resource_type: roboto.CollectionResourceType,
    noun: str,
) -> list:
    """Return the collection's hydrated records, warning about missing ones.

    ``Collection.from_id`` defaults to ``content_mode=Full``, which makes the
    server hydrate every resource and return full records in
    ``record.resources``. Building domain objects from those records avoids
    an extra round trip per resource.
    """
    record = collection.record
    hydrated = record.resources.get(resource_type, [])
    if not hydrated:
        raise ValueError(
            f"Collection {collection.collection_id!r} (version {record.version}) "
            f"contains no {noun}s."
        )
    missing = record.missing.get(resource_type, [])
    if missing:
        logger.warning(
            "Collection %s references %d %s(s) that could not be hydrated "
            "(deleted or inaccessible) — skipping: %s",
            collection.collection_id,
            len(missing),
            noun,
            sorted(ref.resource_id for ref in missing),
        )
    return hydrated


def _resolve_event_sources(
    collection: roboto.Collection,
    contract: Contract,
) -> ResolvedSources:
    events = sorted(
        (
            roboto.Event(
                roboto.EventRecord.model_validate(r) if isinstance(r, dict) else r,
            )
            for r in _hydrated_resources(
                collection, roboto.CollectionResourceType.Event, "event",
            )
        ),
        key=lambda e: e.start_time,
    )
    logger.info(
        "Loaded %d events from collection %s (version %d)",
        len(events), collection.collection_id, collection.record.version,
    )

    sources: list[EpisodeSource] = []
    distinct_ds_ids: set[str] = set()
    for event in events:
        ds_ids = list(event.dataset_ids() or [])
        if not ds_ids:
            logger.warning(
                "Event %s has no dataset_ids; skipping",
                getattr(event, "event_id", "?"),
            )
            continue
        distinct_ds_ids.update(ds_ids)
        # An event is read from its first dataset only.
        sources.append(EpisodeSource(
            source_type="event",
            source_id=event.event_id,
            scope_key=ds_ids[0],
            dataset_ids=(ds_ids[0],),
            start_time=int(event.start_time),
            end_time=int(event.end_time),
            metadata=dict(event.metadata or {}),
        ))
    if not sources:
        raise ValueError(
            f"None of the {len(events)} event(s) in collection "
            f"{collection.collection_id!r} are associated with a dataset."
        )

    # Per-dataset role resolution before topic discovery: a contract that
    # binds specs to ``role:`` placeholders gets its topics filled in from
    # ``file.metadata["role"]`` on this dataset's files. Contracts that
    # already use literal ``topic:`` pass through ``resolve_role_bindings``
    # unchanged.
    logger.info(
        "Matching contract topics against files in %d dataset(s)...",
        len(distinct_ds_ids),
    )
    prepared: dict[str, PreparedScope] = {}
    for ds_id in distinct_ds_ids:
        ds = roboto.Dataset.from_id(ds_id)
        resolved = resolve_role_bindings(ds, contract)
        topics, dedup_groups = collect_topics_from_dataset(ds, resolved)
        prepared[ds_id] = PreparedScope(resolved, topics, dedup_groups)
    return ResolvedSources(sources, prepared)


def _resolve_session_sources(
    collection: roboto.Collection,
    contract: Contract,
) -> ResolvedSources:
    sessions = [
        roboto.Session(
            roboto.SessionRecord.model_validate(r) if isinstance(r, dict) else r,
        )
        for r in _hydrated_resources(
            collection, roboto.CollectionResourceType.Session, "session",
        )
    ]
    logger.info(
        "Loaded %d sessions from collection %s (version %d)",
        len(sessions), collection.collection_id, collection.record.version,
    )

    logger.info(
        "Matching contract topics against the files of %d session(s)...",
        len(sessions),
    )
    sources: list[EpisodeSource] = []
    prepared: dict[str, PreparedScope] = {}
    for session in sessions:
        session_files = list(session.list_files())
        if not session_files:
            logger.warning("Session %s has no files; skipping", session.session_id)
            continue
        _check_no_epoch_offset(session, session_files)

        files = [roboto.File.from_id(sf.file_id) for sf in session_files]
        scope = f"session {session.session_id}"
        resolved = resolve_role_bindings_for_files(
            lambda files=files: files, contract, scope=scope,
        )
        topics, dedup_groups = collect_topics_from_files(
            files, resolved, scope=f"Session {session.session_id}",
        )
        start_time, end_time = _session_window(session, session_files, topics)
        prepared[session.session_id] = PreparedScope(resolved, topics, dedup_groups)
        sources.append(EpisodeSource(
            source_type="session",
            source_id=session.session_id,
            scope_key=session.session_id,
            dataset_ids=tuple(sorted({
                ds_id
                for ds_id in (
                    getattr(sf, "dataset_id", None) or f.dataset_id
                    for sf, f in zip(session_files, files, strict=True)
                )
                if ds_id
            })),
            start_time=start_time,
            end_time=end_time,
            metadata=dict(session.metadata or {}),
        ))
    if not sources:
        raise ValueError(
            f"None of the {len(sessions)} session(s) in collection "
            f"{collection.collection_id!r} have any files."
        )
    sources.sort(key=lambda s: (s.start_time, s.source_id))
    return ResolvedSources(sources, prepared)


def _check_no_epoch_offset(session: roboto.Session, session_files: list) -> None:
    """Refuse sessions that re-anchor their files onto a different clock.

    A session can shift a file's data onto the wall clock with a
    ``unix_epoch_offset_ns``; its bounds are then in the shifted clock while
    the file-scoped topic reads used here return the file's own timestamps,
    so the window would select the wrong data, or none.
    """
    offset_files = [
        sf.file_id for sf in session_files
        if getattr(sf, "unix_epoch_offset_ns", None)
    ]
    if offset_files:
        raise ValueError(
            f"Session {session.session_id} applies a unix_epoch_offset_ns to "
            f"file(s) {sorted(offset_files)}; sessions with re-anchored file "
            f"timelines are not supported."
        )


def _session_window(
    session: roboto.Session,
    session_files: list,
    topics: dict[str, list[roboto.Topic]],
) -> tuple[int, int]:
    """The ``[start, end]`` window, in ns since epoch, the episode covers.

    The session's aggregate bounds when it has them. Otherwise each file
    contributes the window the session holds of it, or, for a whole-file
    membership, the span of the contract topics matched on it. Sessions
    whose files predate the platform's normalized topic rows have no
    aggregate bounds, which is why the fallback exists.
    """
    if session.min_timestamp_ns is not None and session.max_timestamp_ns is not None:
        return int(session.min_timestamp_ns), int(session.max_timestamp_ns)

    topics_by_file: dict[str, list[roboto.Topic]] = {}
    for topic_list in topics.values():
        for t in topic_list:
            topics_by_file.setdefault(t.file_id, []).append(t)

    starts: list[int] = []
    ends: list[int] = []
    for sf in session_files:
        lo = getattr(sf, "min_wall_clock_timestamp_ns", None)
        hi = getattr(sf, "max_wall_clock_timestamp_ns", None)
        if lo is not None and hi is not None:
            starts.append(int(lo))
            ends.append(int(hi))
            continue
        for t in topics_by_file.get(sf.file_id, []):
            if t.start_time is not None and t.end_time is not None:
                starts.append(int(t.start_time))
                ends.append(int(t.end_time))
    if not starts:
        raise ValueError(
            f"Session {session.session_id} has no time bounds, and none of the "
            f"contract topics matched on its files carry a time range."
        )
    logger.info(
        "Session %s has no aggregate time bounds; derived [%d, %d] from its "
        "files' contract topics.",
        session.session_id, min(starts), max(ends),
    )
    return min(starts), max(ends)


def annotate_manifest_entry(
    entry: dict[str, Any],
    sources_by_key: dict[str, EpisodeSource],
) -> dict[str, Any]:
    """Rewrite one ``episode_to_event`` / ``skipped_events`` entry for sessions.

    Entries carry the episode's ``scope_key`` in ``source_dataset_id`` and
    its source id in ``event_id``. For an event those already are the
    dataset and event ids, so the entry is returned as is. For a session
    both are the session id: record it under ``session_id``, clear
    ``event_id``, and record the session's datasets instead.
    """
    source = sources_by_key.get(entry.get("source_dataset_id"))
    if source is None or source.source_type != "session":
        return entry
    out = dict(entry)
    out["event_id"] = None
    out["session_id"] = source.source_id
    out["source_dataset_id"] = source.dataset_ids[0] if source.dataset_ids else None
    out["source_dataset_ids"] = list(source.dataset_ids)
    return out
