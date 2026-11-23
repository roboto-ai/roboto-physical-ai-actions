"""Per-episode conversion cache kept in the org's asset storage.

Each cached entry is a complete one-episode LeRobot dataset, in the format
the embedded lerobot writes (v3.0 or v2.1), stored at
``.lerobot-cache/<recipe_hash>/<episode_key>/`` in ``Org.assets``. An
export merges cached entries (hits) with freshly converted ones (misses)
without re-encoding: on v3.0 with the helpers of
``lerobot.datasets.aggregate``, which stream-copy video packets and parquet
rows, and on v2.1 through
:func:`roboto_to_lerobot.merge_v21.merge_v21_entries`, which moves each
episode's files. Either way the merged output matches a full conversion of
the same events in the same order.

Both keys are sha256 hex digests of canonical JSON:

* ``recipe_hash`` covers everything that changes the bytes of an entry or
  whether two entries can be merged: the parsed contract YAML, this
  package's source, the lerobot version and codebase version, the video
  codec, the streaming-encoding flag and the resolved ``features`` dict.
* ``episode_key`` covers everything that selects one episode's input: the
  event (id, bounds, source dataset, raw task label), the fetch window
  padding, the dataset's resolved topic bindings, and a fingerprint of every
  topic chunk that intersects the padded window.

An entry is a hit only once its marker file exists. The marker is uploaded
after every other file of the entry, so an interrupted upload leaves an
entry that lookups ignore.

The cache never lists files: the platform's file listing leaves out every
path that starts with ``.``, so nothing under ``.lerobot-cache/`` is ever
listed. The marker names the entry's other files instead, and records where
each file's bytes are stored, so a lookup fetches the marker by its path and
then downloads the entry's files without one API call per file. An entry
whose marker lacks those locations, or whose stored bytes cannot be
downloaded, is fetched file by file by path instead.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any

import yaml
from roboto.exceptions import RobotoNotFoundException

from .contract_utils import Contract, _topic_intersects_window
from .logger import logger

CACHE_PREFIX = ".lerobot-cache"
# Uploaded after every other file of its entry. It holds the episode's
# manifest summary, so that a hit can be reported without opening the
# entry's parquet files; the paths of the entry's other files, under
# MARKER_FILES_KEY; and where each of those files is stored, under
# MARKER_OBJECTS_KEY (see object_ref).
MARKER_NAME = "_COMPLETE.json"
MARKER_FILES_KEY = "files"
MARKER_OBJECTS_KEY = "objects"
# Concurrent lookups of cache files by path, one API call each.
_FETCH_THREADS = 16


def _sha256_json(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _source_digest() -> str:
    """Hash every ``.py`` file in this package.

    The package version in ``pyproject.toml`` does not tell two builds
    apart; hashing the source invalidates the cache on any
    code change, including edits that would not change the output.
    """
    pkg_root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(pkg_root.rglob("*.py")):
        digest.update(path.relative_to(pkg_root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def recipe_hash(
    *, contract_path: Path, features: dict, streaming_encoding: bool, video_encoding: Mapping[str, Any],
) -> str:
    """Hash the settings that every entry merged into one export must share.

    The contract is hashed as parsed YAML, so comments, whitespace, key
    order and the file's path do not change the hash. Encoder thread count,
    shard count and the collection are left out: they do not change the
    bytes of an entry. The lerobot and codebase versions keep the two
    variants' entries (v2.1 and v3.0) under different hashes.
    """
    try:
        from lerobot.datasets.dataset_metadata import CODEBASE_VERSION
    except ImportError:  # lerobot 0.3.x, which writes v2.1
        from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION

    return _sha256_json({
        "contract": yaml.safe_load(contract_path.read_text(encoding="utf-8")),
        "source": _source_digest(),
        "lerobot": _pkg_version("lerobot"),
        "codebase_version": CODEBASE_VERSION,
        "video_encoding": dict(video_encoding),
        "streaming_encoding": streaming_encoding,
        "features": features,
    })


def episode_key(
    *,
    event_id: str | None,
    src_ds_id: str,
    start_time_ns: int,
    end_time_ns: int,
    task_label: str | None,
    buffer_ns: int,
    action_lead_ns: int,
    contract: Contract,
    topics: dict[str, list[Any]],
) -> str:
    """Hash one event's inputs, or one session's.

    For a session, ``event_id`` and ``src_ds_id`` are both the session id
    and ``topics`` are those of the session's files. The events on a session
    are not hashed: they become sub-tasks, which the export writes after it
    merges the entries.

    ``contract`` is the dataset's role-resolved contract, so a change in
    which topic a ``role:`` binds to changes the key. Each topic chunk that
    intersects the fetch window contributes its id, file id, file version,
    time range, message count and schema checksum, so a re-uploaded or
    re-ingested source file changes the key.
    """
    window_start = start_time_ns - buffer_ns
    window_end = end_time_ns + buffer_ns + action_lead_ns
    fingerprint = sorted(
        [
            name,
            t.topic_id,
            t.file_id,
            t.record.association.association_version,
            t.start_time,
            t.end_time,
            t.message_count,
            t.record.schema_checksum,
        ]
        for name, topic_list in topics.items()
        for t in topic_list
        if _topic_intersects_window(t, window_start, window_end)
    )
    return _sha256_json({
        "event_id": event_id,
        "src_ds_id": src_ds_id,
        "start_time_ns": int(start_time_ns),
        "end_time_ns": int(end_time_ns),
        "task_label": task_label,
        "buffer_ns": int(buffer_ns),
        "action_lead_ns": int(action_lead_ns),
        "bindings": {
            kind: [spec.topic for spec in getattr(contract, kind)]
            for kind in ("observations", "videos", "actions", "tasks")
        },
        "topics": fingerprint,
    })


def entry_path(recipe: str, key: str) -> str:
    """Directory of one entry, relative to the root of the cache's files."""
    return f"{CACHE_PREFIX}/{recipe}/{key}"


def episode_summary(entry: Path) -> dict:
    """The episode summary in the marker of the local entry ``entry``, without its file list or locations."""
    marker = json.loads((entry / MARKER_NAME).read_text())
    marker.pop(MARKER_FILES_KEY, None)
    marker.pop(MARKER_OBJECTS_KEY, None)
    return marker


def object_ref(file: Any) -> dict:
    """Where the bytes of the ``roboto.File`` ``file`` are stored, as JSON :func:`download_refs` takes."""
    return {
        "bucket": file.record.bucket,
        "uri": file.record.uri,
        "association_id": file.association.association_id,
        "org_id": file.org_id,
    }


def download_refs(pairs: list[tuple[dict, Path]]) -> None:
    """Download ``(object_ref, dest)`` pairs, one ``FileService`` batch per association.

    A batch fetches read credentials once and transfers its files
    concurrently; ``File.download`` sets up both for every file. Raises
    when any file cannot be downloaded.
    """
    from roboto import RobotoClient
    from roboto.association import Association
    from roboto.storage.file_service import FileService

    batches: dict[tuple[str, str], list[dict]] = {}
    for ref, dest in pairs:
        dest.parent.mkdir(parents=True, exist_ok=True)
        batches.setdefault((ref["association_id"], ref["org_id"]), []).append(
            {"bucket_name": ref["bucket"], "source_uri": ref["uri"], "destination_path": dest}
        )
    service = FileService(RobotoClient.defaulted())
    for (association_id, org_id), batch in batches.items():
        service.download(files=batch, association=Association.from_id(association_id), caller_org_id=org_id)


class EpisodeCache:
    """Cache entries stored as files in a ``roboto.FileSystem``, the org's assets.

    Uses only ``list_files`` (to check access), ``get_file_by_path``,
    :func:`download_refs` and ``upload_files``. Fetched entries land at
    ``<local_root>/<entry_path(...)>``, the same layout :meth:`upload`
    reads from.
    """

    def __init__(self, files: Any) -> None:
        self._files = files

    def probe(self) -> None:
        """Make one listing call that matches no file, to check the files can be reached.

        Raises whatever the listing raises, such as
        ``RobotoFeatureNotAvailableException`` when the org's asset storage
        is not enabled.
        """
        next(iter(self._files.list_files(include_patterns=[f"{CACHE_PREFIX}/{MARKER_NAME}"])), None)

    def fetch(self, recipe: str, keys: list[str], local_root: Path) -> set[str]:
        """Download every complete entry among ``keys`` and return their keys.

        Fetches each key's marker by path; a missing marker, or a marker
        without a file list, is a miss. Then downloads the files of every
        entry whose marker records where they are stored, all at once, and
        looks up the files of the other entries by path, as it does for all
        entries when that download fails. An entry missing any of its files
        is a miss too, and its partial local copy is removed.
        """

        def remote(key: str, name: str) -> str:
            return f"{entry_path(recipe, key)}/{name}"

        def local(key: str) -> Path:
            return local_root / entry_path(recipe, key)

        with ThreadPoolExecutor(_FETCH_THREADS) as pool:
            found = self._download_all(pool, [(remote(k, MARKER_NAME), local(k) / MARKER_NAME) for k in keys])
            markers = {
                k: json.loads((local(k) / MARKER_NAME).read_text())
                for k, ok in zip(keys, found, strict=True)
                if ok
            }
            files = {k: m.get(MARKER_FILES_KEY) for k, m in markers.items()}
            located = {
                k: m[MARKER_OBJECTS_KEY]
                for k, m in markers.items()
                if files[k] is not None and set(m.get(MARKER_OBJECTS_KEY) or {}) == set(files[k])
            }
            try:
                download_refs([(refs[rel], local(k) / rel) for k, refs in located.items() for rel in files[k]])
            except Exception as e:  # any failure falls back to lookups by path
                logger.warning("Episode cache: downloading by stored location failed (%s); looking files up by path.", e)
                located = {}
            jobs = [(k, rel) for k, rels in files.items() if k not in located for rel in rels or []]
            fetched = self._download_all(pool, [(remote(k, rel), local(k) / rel) for k, rel in jobs])
            incomplete = {k for (k, _), ok in zip(jobs, fetched, strict=True) if not ok}
        misses = incomplete | {k for k, rels in files.items() if rels is None}
        for k in misses:
            shutil.rmtree(local(k), ignore_errors=True)
        return set(files) - misses

    def _download_all(self, pool: ThreadPoolExecutor, items: list[tuple[str, Path]]) -> list[bool]:
        """Download each ``(relative_path, dest)``; ``False`` where no file has that path.

        Looks the files up by path on ``pool``, then downloads the ones found
        in one batch.
        """
        found = list(pool.map(self._lookup, [rel for rel, _ in items]))
        download_refs([(object_ref(f), dest) for f, (_, dest) in zip(found, items, strict=True) if f is not None])
        return [f is not None for f in found]

    def _lookup(self, relative_path: str) -> Any:
        try:
            return self._files.get_file_by_path(relative_path)
        except RobotoNotFoundException:
            return None

    def upload(self, recipe: str, local_root: Path, keys: list[str]) -> None:
        """Upload entries from ``<local_root>/<entry_path(...)>``, markers last.

        Adds the entry's other files to each marker, under
        ``MARKER_FILES_KEY``, and where each was stored, under
        ``MARKER_OBJECTS_KEY``, before uploading it. An entry whose files
        cannot all be looked up after their upload gets a marker without
        locations, which :meth:`fetch` handles by looking the files up.
        """
        bodies: dict[Path, str] = {}
        entries: dict[Path, list[Path]] = {}
        for key in keys:
            entry = local_root / entry_path(recipe, key)
            entries[entry] = sorted(p for p in entry.rglob("*") if p.is_file() and p.name != MARKER_NAME)
            for path in entries[entry]:
                bodies[path] = path.relative_to(local_root).as_posix()
        found: dict[Path, Any] = {}
        if bodies:
            self._files.upload_files(list(bodies), bodies, print_progress=False)
            with ThreadPoolExecutor(_FETCH_THREADS) as pool:
                found = dict(zip(bodies, pool.map(self._lookup, bodies.values()), strict=True))
        markers: dict[Path, str] = {}
        for entry, files in entries.items():
            marker = entry / MARKER_NAME
            content = json.loads(marker.read_text())
            content[MARKER_FILES_KEY] = [p.relative_to(entry).as_posix() for p in files]
            if all(found[p] is not None for p in files):
                content[MARKER_OBJECTS_KEY] = {p.relative_to(entry).as_posix(): object_ref(found[p]) for p in files}
            marker.write_text(json.dumps(content, default=str))
            markers[marker] = marker.relative_to(local_root).as_posix()
        if markers:
            self._files.upload_files(list(markers), markers, print_progress=False)
