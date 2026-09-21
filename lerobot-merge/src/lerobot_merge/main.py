"""Hosted merge of N sharded LeRobot datasets into one.

Designed to be invoked after a fan-out of ``roboto-to-lerobot`` invocations
have each written their own LeRobot output to a shared Roboto dataset under
a per-invocation prefix. The Roboto runtime pre-downloads every shard tree
(plus its manifest and contract) into ``context.input_dir`` via the
invocation's ``input_data`` patterns; this action then calls
``lerobot.datasets.aggregate.aggregate_datasets`` (which stream-copies
videos and reindexes parquet columns in a single pass) and writes the
merged dataset to ``context.output_dir`` for the runtime to auto-upload.

The merged manifest is **self-contained**: it inlines every per-shard
conversion manifest's episode_to_event (renumbered to the merged dataset's
global episode indices), dedup records, and skipped events. Once a merge
completes successfully, the per-shard collections, the intermediate shards
dataset, and the conversion invocations can all be deleted without losing
provenance — every fact needed to trace a merged episode back to its source
event lives in the merged manifest itself.

Compared to running the merge in a notebook, the data movement here is
AWS-internal (same region as the source dataset's S3 bucket): no WAN
round-trip for video bytes.
"""

from __future__ import annotations

import datetime as _dt
import json
import pathlib
import shutil
import time
from collections import Counter
from importlib.metadata import version as _pkg_version

import roboto
from roboto.updates import MetadataChangeset

from .logger import logger

# Name used by the conversion action for the archived contract; mirroring
# it here means a merged dataset's <iv>/contract.yaml sits at the same
# relative path as a single-shot conversion's <iv>/contract.yaml.
_ARCHIVED_CONTRACT_FILENAME = "contract.yaml"


def main(context: roboto.InvocationContext) -> None:
    logger.setLevel(context.log_level)
    logger.info("Starting lerobot-merge (invocation_id=%s)", context.invocation_id)

    shard_invocation_ids = _parse_shard_invocation_ids(
        context.get_parameter("shard_invocation_ids")
    )

    parent_collection_id = context.get_parameter("parent_collection_id").strip()
    if not parent_collection_id:
        raise ValueError("parent_collection_id must be a non-empty string")
    parent_collection_version_raw = context.get_optional_parameter("parent_collection_version")
    parent_collection_version = (
        int(parent_collection_version_raw)
        if parent_collection_version_raw is not None
        else None
    )

    aggregate_kwargs = _parse_aggregate_overrides(context)

    # Source dataset comes from the invocation's data_source. The runtime
    # has already pre-downloaded the patterns we asked for via input_data
    # into context.input_dir; we just point at them.
    source_dataset_id = context.dataset.dataset_id

    # Conversion convention: each shard lives at <iv>/combined/ with a
    # sibling <iv>/manifest.json and <iv>/contract.yaml.
    source_prefixes = [f"{iv}/combined" for iv in shard_invocation_ids]

    logger.info(
        "Merging %d shard(s) from dataset %s: %s",
        len(source_prefixes), source_dataset_id, source_prefixes,
    )

    shard_roots, shard_manifests, shard_contract_paths = (
        _load_shards_from_input_dir(context.input_dir, shard_invocation_ids)
    )
    per_shard_summary = _summarize_shards(shard_roots, shard_manifests)

    # Mirror the conversion action's layout: everything for this run lives
    # under <output_dir>/<invocation_id>/. After auto-upload the merged
    # dataset will hold <merge_iv>/{combined/, manifest.json, contract.yaml},
    # which is the same shape a single-shot conversion produces.
    subfolder = context.output_dir / context.invocation_id
    subfolder.mkdir(parents=True, exist_ok=True)
    merged_root = subfolder / "combined"
    # Do NOT pre-create merged_root: lerobot's LeRobotDatasetMetadata.create()
    # — called inside aggregate_datasets — does ``root.mkdir(exist_ok=False)``
    # and would raise FileExistsError if the directory already exists.

    merged_info = _run_aggregate(shard_roots, merged_root, aggregate_kwargs)

    # Archive the contract bytes next to the merged dataset, same as the
    # conversion action does for its <iv>/contract.yaml. All shards must
    # share the same contract.sha256 (enforced in _assemble_manifest), so
    # picking the first shard that has a contract file is sufficient.
    archived_contract_path = subfolder / _ARCHIVED_CONTRACT_FILENAME
    source_contract = next(
        (p for p in shard_contract_paths if p is not None and p.is_file()),
        None,
    )
    if source_contract is None:
        archived_contract_filename: str | None = None
        logger.warning(
            "No per-shard contract.yaml found in any shard prefix; the "
            "merged dataset will not carry an archived contract. The "
            "merged manifest's contract metadata is still populated from "
            "the per-shard conversion manifests."
        )
    else:
        shutil.copyfile(source_contract, archived_contract_path)
        archived_contract_filename = _ARCHIVED_CONTRACT_FILENAME
        logger.info("Contract archived: %s", archived_contract_path)

    manifest = _assemble_manifest(
        context=context,
        source_dataset_id=source_dataset_id,
        source_prefixes=source_prefixes,
        parent_collection_id=parent_collection_id,
        parent_collection_version=parent_collection_version,
        shard_manifests=shard_manifests,
        per_shard_summary=per_shard_summary,
        merged_info=merged_info,
        archived_contract_filename=archived_contract_filename,
    )
    manifest_path = subfolder / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    logger.info("Wrote merged manifest: %s", manifest_path)

    # Provenance stamping — same shape the conversion action writes, with
    # the merge invocation_id substituted in. After this, both per-file
    # metadata queries (e.g. metadata.contract_name == 'X') and
    # dataset-level invocation lookups (metadata.invocations.<iv>) work
    # against the merged dataset the same way they would against a
    # single-shot conversion output.
    contract_meta = manifest["contract"]
    lerobot_version = _pkg_version("lerobot")
    per_file_stamp = {
        "invocation_id": context.invocation_id,
        "contract_name": contract_meta.get("name"),
        "contract_version": contract_meta.get("version"),
        "contract_sha256": contract_meta.get("sha256"),
        "contract_path": contract_meta.get("path"),
        "collection_id": parent_collection_id,
        "collection_version": parent_collection_version,
        "lerobot_version": lerobot_version,
    }
    _stamp_output_files(context, subfolder, per_file_stamp)

    # Dataset-level metadata.invocations.<merge_iv> goes on the upload
    # destination — the merged dataset itself. context.dataset resolves
    # to the invocation's data_source (the shards dataset), which the
    # caller typically deletes after the merge, so stamping there would
    # be lost. Look the destination up off context.invocation and write
    # there; fall back to a warning when no dataset destination is
    # configured (e.g. invoke-local without an upload target).
    upload_destination = context.invocation.upload_destination
    if upload_destination is not None and upload_destination.is_dataset:
        merged_dataset = roboto.Dataset.from_id(
            upload_destination.destination_id,
            roboto_client=context.roboto_client,
        )
        merged_dataset.update(
            metadata_changeset=MetadataChangeset.Builder()
            .put_field(f"invocations.{context.invocation_id}", {
                "contract_name": contract_meta.get("name"),
                "contract_version": contract_meta.get("version"),
                "contract_sha256": contract_meta.get("sha256"),
                "contract_path": contract_meta.get("path"),
                "collection_id": parent_collection_id,
                "collection_version": parent_collection_version,
                "total_episodes": manifest["total_episodes"],
                "total_frames": manifest["total_frames"],
                "lerobot_version": lerobot_version,
                "had_dedup": bool(manifest.get("dedup")),
                "dedup_groups": len(manifest.get("dedup") or []),
            })
            .build()
        )
    else:
        logger.warning(
            "Merge invocation has no dataset upload destination "
            "(destination=%s); skipping dataset-level metadata.invocations write.",
            upload_destination,
        )

    logger.info(
        "lerobot-merge complete: %d episodes, %d frames merged from %d shard(s).",
        merged_info["total_episodes"], merged_info["total_frames"], len(source_prefixes),
    )


def _stamp_output_files(
    context: roboto.InvocationContext,
    subfolder: pathlib.Path,
    stamp: dict,
) -> None:
    """Annotate every file under ``subfolder`` with provenance metadata.

    Port of the conversion action's stamp loop. The Roboto hosted runtime
    populates ``file_changeset_manager``; ``invoke-local`` does not, so we
    log and skip in that case rather than failing the merge.
    """
    try:
        changeset_manager = context.file_changeset_manager
    except ValueError as exc:
        logger.warning(
            "Skipping per-file metadata stamping (local invocation): %s", exc,
        )
        return

    for path in subfolder.rglob("*"):
        if not path.is_file():
            continue
        relative_path = str(path.relative_to(context.output_dir))
        changeset_manager.put_fields(relative_path, stamp)


def _parse_shard_invocation_ids(raw: str) -> list[str]:
    try:
        ids = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"shard_invocation_ids must be a JSON array; got: {raw!r}"
        ) from exc
    if not isinstance(ids, list) or not ids:
        raise ValueError("shard_invocation_ids must be a non-empty JSON array")
    cleaned: list[str] = []
    for entry in ids:
        if not isinstance(entry, str) or not entry.strip():
            raise ValueError(
                f"Each entry in shard_invocation_ids must be a non-empty string; got: {entry!r}"
            )
        # Guard against accidental path-like values — these are bare invocation IDs.
        normalized = entry.strip().strip("/")
        if "/" in normalized:
            raise ValueError(
                f"shard_invocation_ids entries must be bare invocation IDs (no '/'); got: {entry!r}"
            )
        cleaned.append(normalized)
    return cleaned


def _parse_aggregate_overrides(context: roboto.InvocationContext) -> dict:
    kwargs: dict = {}
    data_size = context.get_optional_parameter("data_files_size_in_mb")
    if data_size is not None:
        kwargs["data_files_size_in_mb"] = float(data_size)
    video_size = context.get_optional_parameter("video_files_size_in_mb")
    if video_size is not None:
        kwargs["video_files_size_in_mb"] = float(video_size)
    chunk_size = context.get_optional_parameter("chunk_size")
    if chunk_size is not None:
        kwargs["chunk_size"] = int(chunk_size)
    if kwargs:
        logger.info("aggregate_datasets overrides: %s", kwargs)
    return kwargs


def _load_shards_from_input_dir(
    input_dir: pathlib.Path,
    shard_invocation_ids: list[str],
) -> tuple[list[pathlib.Path], list[dict | None], list[pathlib.Path | None]]:
    """Locate each shard's pre-downloaded LeRobot root, manifest, and contract.

    The Roboto runtime has already downloaded everything matched by the
    invocation's ``input_data`` patterns into ``input_dir``, preserving the
    source dataset's relative paths. The conversion action writes each
    shard as ``<iv>/{combined/, manifest.json, contract.yaml}``, so this
    function just resolves those paths under ``input_dir / <iv>``.

    Returns (shard_roots, shard_manifests, shard_contract_paths). For shard i:
    - shard_roots[i] is required (raises if missing — the merge cannot proceed
      without the LeRobot tree).
    - shard_manifests[i] is None if the conversion manifest is missing or
      unparseable (the merge still proceeds; diagnostics for that shard degrade).
    - shard_contract_paths[i] is None if no contract.yaml was found at the
      expected location. The caller picks any non-None entry as the source
      bytes for the merged dataset's archived contract.
    """
    shard_roots: list[pathlib.Path] = []
    shard_manifests: list[dict | None] = []
    shard_contract_paths: list[pathlib.Path | None] = []

    for idx, iv in enumerate(shard_invocation_ids):
        shard_dir = input_dir / iv
        lerobot_root = shard_dir / "combined"
        if not lerobot_root.is_dir():
            raise FileNotFoundError(
                f"Expected pre-downloaded LeRobot root at {lerobot_root} for shard "
                f"{idx} (invocation {iv!r}). Check that the invocation's input_data "
                f"includes a pattern covering {iv}/combined/** in the data_source dataset."
            )

        manifest_local = shard_dir / "manifest.json"
        manifest_obj: dict | None = None
        if manifest_local.is_file():
            try:
                manifest_obj = json.loads(manifest_local.read_text())
            except json.JSONDecodeError as exc:
                logger.warning(
                    "Shard %d (%s): manifest at %s is not valid JSON (%s); "
                    "merged manifest will lack this shard's per-event provenance.",
                    idx, iv, manifest_local, exc,
                )
        else:
            logger.warning(
                "Shard %d (%s): no conversion manifest at %s; "
                "merged manifest will lack this shard's per-event provenance.",
                idx, iv, manifest_local,
            )

        contract_local = shard_dir / "contract.yaml"
        contract_path = contract_local if contract_local.is_file() else None

        logger.info(
            "Shard %d (%s): root=%s manifest=%s contract=%s",
            idx, iv, lerobot_root,
            "loaded" if manifest_obj else "missing/invalid",
            "found" if contract_path else "missing",
        )

        shard_roots.append(lerobot_root)
        shard_manifests.append(manifest_obj)
        shard_contract_paths.append(contract_path)

    return shard_roots, shard_manifests, shard_contract_paths


def _summarize_shards(
    shard_roots: list[pathlib.Path],
    shard_manifests: list[dict | None],
) -> list[dict]:
    """Read each shard's ``meta/info.json`` and report basic stats."""
    summary: list[dict] = []
    for idx, root in enumerate(shard_roots):
        info_path = root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(
                f"Shard {idx} at {root} is missing meta/info.json — "
                f"the prefix probably does not point at a LeRobot dataset root."
            )
        info = json.loads(info_path.read_text())
        m = shard_manifests[idx] or {}
        entry = {
            "shard_idx": idx,
            "root": str(root),
            "total_episodes": int(info.get("total_episodes", 0)),
            "total_frames": int(info.get("total_frames", 0)),
            "fps": info.get("fps"),
            "codebase_version": info.get("codebase_version"),
            "conversion_invocation_id": m.get("invocation_id"),
            "conversion_collection_id": m.get("collection_id"),
            "conversion_collection_version": m.get("collection_version"),
        }
        summary.append(entry)
        logger.info(
            "Shard %d: episodes=%d frames=%d fps=%s codebase=%s conv_iv=%s",
            entry["shard_idx"], entry["total_episodes"], entry["total_frames"],
            entry["fps"], entry["codebase_version"], entry["conversion_invocation_id"],
        )
    return summary


def _run_aggregate(
    shard_roots: list[pathlib.Path],
    merged_root: pathlib.Path,
    aggregate_kwargs: dict,
) -> dict:
    """Call ``lerobot.datasets.aggregate.aggregate_datasets`` and read merged info."""
    # Imported lazily so the action's module imports stay cheap during cold start.
    from lerobot.datasets.aggregate import aggregate_datasets

    repo_ids = [f"shard-{i:03d}" for i in range(len(shard_roots))]
    logger.info(
        "Running lerobot.aggregate_datasets: %d shard(s) -> %s",
        len(shard_roots), merged_root,
    )
    t0 = time.monotonic()
    aggregate_datasets(
        repo_ids=repo_ids,
        aggr_repo_id=merged_root.name,
        roots=shard_roots,
        aggr_root=merged_root,
        **aggregate_kwargs,
    )
    elapsed = time.monotonic() - t0
    logger.info("aggregate_datasets finished in %.1fs", elapsed)

    # Read meta/info.json directly. Re-opening via LeRobotDatasetMetadata(...)
    # would trigger a HuggingFace Hub lookup for the synthetic repo_id and 401
    # in the action's no-token environment.
    info = json.loads((merged_root / "meta" / "info.json").read_text())
    return {
        "total_episodes": int(info.get("total_episodes", 0)),
        "total_frames": int(info.get("total_frames", 0)),
        "total_tasks": int(info.get("total_tasks", 0)),
        "fps": info.get("fps"),
        "codebase_version": info.get("codebase_version"),
        "aggregate_elapsed_s": elapsed,
    }


def _assemble_manifest(
    *,
    context: roboto.InvocationContext,
    source_dataset_id: str,
    source_prefixes: list[str],
    parent_collection_id: str,
    parent_collection_version: int | None,
    shard_manifests: list[dict | None],
    per_shard_summary: list[dict],
    merged_info: dict,
    archived_contract_filename: str | None,
) -> dict:
    """Build a single-shot-equivalent manifest from the per-shard conversion manifests.

    The merged dataset's episode_to_event is renumbered using the same offset
    rule that aggregate_datasets applies internally: shards are stacked in
    input order, shard i contributes episodes [offset_i, offset_i + n_i),
    where offset_i = sum(n_j for j < i).
    """
    if any(m is None for m in shard_manifests):
        missing = [i for i, m in enumerate(shard_manifests) if m is None]
        logger.warning(
            "Per-shard conversion manifests missing for shard(s) %s; merged "
            "manifest's episode_to_event / dedup / skipped_events will be incomplete.",
            missing,
        )
    present = [m for m in shard_manifests if m is not None]

    contract = _verify_or_warn(
        present, key_path=("contract", "sha256"), label="contract.sha256",
    )
    contract_meta = _coalesce_contract(present)
    # Re-add archived_filename for parity with single-shot manifests. We
    # intentionally rebuild this field rather than carrying it across from
    # the per-shard manifests because the archived file's location is
    # specific to the merge invocation, not any one conversion.
    if archived_contract_filename is not None:
        contract_meta["archived_filename"] = archived_contract_filename

    lerobot_versions = _collect(present, ("codebase", "lerobot"))
    if len({v for v in lerobot_versions if v is not None}) > 1:
        logger.warning(
            "Per-shard codebase.lerobot versions differ: %s. The merged dataset "
            "was produced by lerobot.aggregate_datasets at this action's install "
            "version; downstream tools should treat the merged dataset as the "
            "newest of the shard versions.",
            sorted({str(v) for v in lerobot_versions if v is not None}),
        )

    # Renumber episode_to_event with offsets matching aggregate_datasets.
    merged_episode_to_event: list[dict] = []
    offset = 0
    for shard_idx, m in enumerate(shard_manifests):
        if m is None:
            offset += per_shard_summary[shard_idx]["total_episodes"]
            continue
        local_entries = m.get("episode_to_event") or []
        for entry in local_entries:
            new_entry = dict(entry)
            new_entry["episode_index"] = offset + int(entry["episode_index"])
            # Diagnostic only — load-bearing data is the renumbered episode_index.
            new_entry["source_shard_idx"] = shard_idx
            merged_episode_to_event.append(new_entry)
        offset += int(m.get("total_episodes", per_shard_summary[shard_idx]["total_episodes"]))

    # Union of dedup records and skipped_events, with rollup summaries
    # rebuilt in the same shape the conversion action produces.
    dedup_records: list[dict] = []
    skipped_events: list[dict] = []
    source_datasets: set[str] = set()
    episodes_per_dataset: Counter = Counter()
    for m in present:
        dedup_records.extend(m.get("dedup") or [])
        skipped_events.extend(m.get("skipped_events") or [])
        for ds in m.get("source_datasets") or []:
            source_datasets.add(ds)
        for ds, n in (m.get("episodes_per_dataset") or {}).items():
            episodes_per_dataset[ds] += int(n)

    dedup_summary = _build_dedup_summary(dedup_records)
    skipped_events_summary = _build_skipped_summary(skipped_events)

    return {
        "invocation_id": context.invocation_id,
        "contract": contract_meta,
        "collection_id": parent_collection_id,
        "collection_version": parent_collection_version,
        "source_datasets": sorted(source_datasets),
        "episodes_per_dataset": dict(episodes_per_dataset),
        "episode_to_event": merged_episode_to_event,
        "total_episodes": merged_info["total_episodes"],
        "total_frames": merged_info["total_frames"],
        "total_tasks": merged_info["total_tasks"],
        "fps": merged_info["fps"],
        "dedup": dedup_records,
        "dedup_summary": dedup_summary,
        "skipped_events": skipped_events,
        "skipped_events_summary": skipped_events_summary,
        "codebase": {
            "lerobot": merged_info.get("codebase_version"),
            "shards_lerobot": sorted({str(v) for v in lerobot_versions if v is not None}),
        },
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
        # Diagnostic block. NOT load-bearing for traceability: every fact
        # needed to map a merged episode back to its source event lives
        # in the fields above. Safe to ignore once the merge has been
        # audited; safe to lose if downstream tooling strips unknown keys.
        "parallel_provenance": {
            "merge_invocation_id": context.invocation_id,
            "source_dataset_id": source_dataset_id,
            "source_prefixes": source_prefixes,
            "shards": per_shard_summary,
            "aggregate_elapsed_s": merged_info["aggregate_elapsed_s"],
            "contract_sha256_consistent": contract,
        },
    }


def _verify_or_warn(manifests: list[dict], *, key_path: tuple[str, ...], label: str) -> bool:
    """Return True iff every manifest has the same value at ``key_path``.

    For contract.sha256 mismatch we raise — different contracts producing
    different schemas yields a merged dataset that's silently miscomposed.
    """
    values = _collect(manifests, key_path)
    distinct = {v for v in values if v is not None}
    if len(distinct) > 1:
        if label == "contract.sha256":
            raise ValueError(
                f"Per-shard {label} values differ across shards: {sorted(distinct)}. "
                f"All shards must have been produced from the same contract; refusing "
                f"to merge incompatible datasets."
            )
        logger.warning("Per-shard %s values differ: %s", label, sorted(distinct))
        return False
    return True


def _collect(manifests: list[dict], key_path: tuple[str, ...]) -> list:
    out = []
    for m in manifests:
        cur = m
        for k in key_path:
            if not isinstance(cur, dict):
                cur = None
                break
            cur = cur.get(k)
        out.append(cur)
    return out


def _coalesce_contract(manifests: list[dict]) -> dict:
    """Pick the contract block from the first shard that has one.

    All shards must share contract.sha256 (verified by _verify_or_warn);
    name/version/path should therefore also match. ``archived_filename``
    is omitted here — the per-shard value points at each conversion's own
    archive; the caller injects a merge-action-owned value after copying
    the contract bytes into ``<merge_iv>/contract.yaml``.
    """
    for m in manifests:
        c = m.get("contract")
        if isinstance(c, dict):
            return {
                "name": c.get("name"),
                "version": c.get("version"),
                "path": c.get("path"),
                "sha256": c.get("sha256"),
            }
    return {}


def _build_skipped_summary(skipped_events: list[dict]) -> dict:
    """Rollup of per-event skip records — mirrors the conversion action's shape."""
    return {
        "count": len(skipped_events),
        "by_stage": dict(Counter(s.get("stage") for s in skipped_events)),
        "by_error_class": dict(Counter(s.get("error_class") for s in skipped_events)),
        "by_source_dataset": dict(Counter(
            s.get("source_dataset_id") for s in skipped_events
        )),
        "event_ids": sorted({
            str(s["event_id"]) for s in skipped_events
            if s.get("event_id") is not None
        }),
    }


def _build_dedup_summary(dedup_records: list[dict]) -> dict:
    """Rollup of dedup records — mirrors the conversion action's shape."""
    dropped_file_ids: set[str] = set()
    topics_dropped = 0
    for r in dedup_records:
        for d in r.get("dropped", []) or []:
            fid = d.get("file_id")
            if fid is not None:
                dropped_file_ids.add(fid)
        topics_dropped += len(r.get("dropped", []) or [])
    return {
        "duplicate_groups": len(dedup_records),
        "topics_dropped": topics_dropped,
        "dropped_file_ids": sorted(dropped_file_ids),
    }
