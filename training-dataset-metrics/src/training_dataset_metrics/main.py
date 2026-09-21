from __future__ import annotations

import datetime as _dt
import hashlib
import json
import pathlib
import re
import shutil
from typing import Any

import numpy as np
import roboto

from .core.aggregation import safe_float, to_json_serializable
from .core.pairing import pair_state_action
from .core.types import (
    ContractRef,
    DatasetSummary,
    FlaggedEpisode,
    MetricResult,
    QualityReport,
    SourceDescriptor,
)
from .logger import logger
from .metrics import METRICS
from .metrics.filtering import build_cleanup_command
from .report import render_all_plots, render_html_report
from .sources import load_post_conversion_episodes, load_pre_conversion_episodes


def main(context: roboto.InvocationContext) -> None:
    logger.setLevel(context.log_level)

    collection_id = _get_param(context, "collection_id")
    contract_param = _get_param(context, "contract")
    write_audit_tags = _coerce_bool(_get_param(context, "write_audit_tags"))

    contract_path: pathlib.Path | None = None
    if collection_id:
        episodes, source, metadata = _run_mode_a(
            context=context,
            collection_id=collection_id,
            contract_name=contract_param,
        )
        contract_path = metadata.get("contract_path")
        mode = "pre_conversion"
    else:
        episodes, source, metadata = _run_mode_b(context)
        mode = "post_conversion"
        if write_audit_tags:
            logger.warning(
                "write_audit_tags=true is ignored in post-conversion mode — "
                "only Mode A (pre-conversion) has events to tag."
            )

    if not episodes:
        raise RuntimeError("No episodes loaded from source; cannot proceed.")

    logger.info("Loaded %d episodes in %s mode", len(episodes), mode)

    state_spec = metadata.get("state_spec")
    action_spec = metadata.get("action_spec")
    pairing = pair_state_action(state_spec, action_spec)
    logger.info("Feature pairing: %s", pairing.method)

    metric_results: dict[str, MetricResult] = {}
    for name, module in METRICS.items():
        logger.info("Computing metric: %s", name)
        try:
            if name == "state_action_alignment":
                result = module.compute(episodes, pairing=pairing)
            else:
                result = module.compute(episodes)
        except Exception as exc:
            logger.exception("Metric %s failed", name)
            result = MetricResult(name=name, errors=[repr(exc)])
        metric_results[name] = result

    flagged_episodes, cli_cleanup = _aggregate_flags(
        episodes=episodes,
        metric_results=metric_results,
        source=source,
    )

    if write_audit_tags and mode == "pre_conversion":
        _write_audit_tags_to_events(
            events=metadata.get("events") or [],
            episodes=episodes,
            flagged_episodes=flagged_episodes,
        )
        _write_audit_tag_to_collection(
            collection_id=source.detail.get("collection_id"),
            invocation_id=getattr(context, "invocation_id", None),
            roboto_client=getattr(context, "roboto_client", None),
        )

    summary = _build_summary(
        episodes=episodes,
        mode=mode,
        state_spec=state_spec,
        action_spec=action_spec,
        metric_results=metric_results,
        n_flagged=len(flagged_episodes),
        codebase_version=metadata.get("codebase_version"),
    )

    run_dir = _resolve_run_output_dir(context)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Writing audit artifacts under %s", run_dir)

    contract_ref = _archive_contract(contract_path, run_dir) if contract_path else None

    report = QualityReport(
        source=source,
        dataset_summary=summary,
        feature_pairing=pairing,
        metrics=metric_results,
        flagged_episodes=flagged_episodes,
        cli_cleanup_command=cli_cleanup,
        contract=contract_ref,
    )

    json_path = run_dir / "report.json"
    json_path.write_text(
        json.dumps(
            to_json_serializable(report.model_dump(mode="json")),
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote %s", json_path)

    plots_dir = run_dir / "plots"
    plots = render_all_plots(report, plots_dir)
    logger.info("Rendered %d plots", len(plots))

    html_path = run_dir / "audit_report.html"
    render_html_report(report, plots, html_path, run_dir=run_dir)
    logger.info("Wrote %s", html_path)


def _get_param(context: roboto.InvocationContext, name: str) -> Any:
    return context.get_optional_parameter(name)


# Runs land in a unique subdir of `context.output_dir` so the Roboto harness'
# implicit upload doesn't overwrite prior reports at the dataset level. The
# invocation_id is preferred (stable, unique, already Roboto's own handle for
# this run); when it isn't set (local `invoke-local` without a recorded
# invocation), we fall back to a UTC timestamp.
_AUDIT_REPORTS_DIR = "audit_reports"
_STAMP_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _resolve_run_output_dir(context: roboto.InvocationContext) -> pathlib.Path:
    base = pathlib.Path(context.output_dir)
    invocation_id = getattr(context, "invocation_id", None)
    if invocation_id:
        stamp = _STAMP_SAFE_RE.sub("-", str(invocation_id))
    else:
        stamp = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    return base / _AUDIT_REPORTS_DIR / stamp


_CONTRACT_ARCHIVE_NAME = "contract.yaml"


def _archive_contract(
    contract_path: pathlib.Path, run_dir: pathlib.Path
) -> ContractRef:
    """Copy the contract YAML into the audit run folder and hash its bytes.

    Keeping a sidecar file next to report.json preserves grep-ability when
    someone browses the output folder; the sha256 gives report.json a
    machine-checkable reference so a moved/edited YAML can be detected.
    """
    data = contract_path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    dest = run_dir / _CONTRACT_ARCHIVE_NAME
    shutil.copyfile(contract_path, dest)
    logger.info("Archived contract → %s (sha256=%s)", dest, sha[:12])
    return ContractRef(
        filename=_CONTRACT_ARCHIVE_NAME,
        sha256=sha,
        source_relative_path=str(contract_path.name),
    )


def _coerce_bool(value: Any) -> bool:
    """Accept bool, or string literals 'true'/'false' from the CLI harness."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value) if value is not None else False


# Audit tags are namespaced under a shared prefix so they are visibly grouped
# on the event and cleanup on re-runs can safely target `audit:*` without
# touching unrelated tags (e.g. `to_lerobot`).
_AUDIT_TAG_PREFIX = "audit:"
_AUDIT_CLEAN_TAG = f"{_AUDIT_TAG_PREFIX}clean"


def _write_audit_tags_to_events(
    events: list,
    episodes: list,
    flagged_episodes: list,
) -> tuple[int, int]:
    """Mode A only: write per-event audit tags based on metric flags.

    - Each raised flag on an episode becomes `audit:<flag_name>` on its event.
    - Episodes with no flags receive `audit:clean`.
    - Tags within the `audit:` namespace from prior runs that aren't part of
      the new verdict are removed; tags outside that namespace are never
      touched.
    - The new verdict is added BEFORE stale audit tags are removed, so a
      partial failure can never leave the event with no audit tags at all
      (worst case is verdict-plus-leftover-stale, which the next run repairs).

    Returns ``(n_succeeded, n_failed)`` — events whose verdict was already
    current count as succeeded.
    """
    flags_by_episode = {fe.episode_index: list(fe.flags) for fe in flagged_episodes}
    audited_indices = sorted({int(ep.episode_index) for ep in episodes})

    n_succeeded = 0
    n_failed = 0
    for idx in audited_indices:
        if idx < 0 or idx >= len(events):
            logger.warning("Episode %d has no matching event — skipping tag write", idx)
            continue
        event = events[idx]
        flag_names = flags_by_episode.get(idx, [])
        new_tags = (
            [f"{_AUDIT_TAG_PREFIX}{name}" for name in flag_names]
            if flag_names
            else [_AUDIT_CLEAN_TAG]
        )
        existing_audit = {
            t for t in (event.tags or []) if t.startswith(_AUDIT_TAG_PREFIX)
        }
        to_add = [t for t in new_tags if t not in existing_audit]
        to_remove = sorted(existing_audit.difference(new_tags))

        if not to_add and not to_remove:
            n_succeeded += 1
            continue

        try:
            if to_add:
                event.put_tags(to_add)
            if to_remove:
                event.remove_tags(to_remove)
            logger.info("Event %d tagged: %s", idx, new_tags)
            n_succeeded += 1
        except Exception:
            logger.exception("Failed to write audit tags for event %d", idx)
            n_failed += 1

    total = n_succeeded + n_failed
    if n_failed:
        logger.warning(
            "Audit tag write: %d/%d events failed; re-run the audit to repair",
            n_failed, total,
        )
    else:
        logger.info("Audit tag write: %d events tagged", n_succeeded)
    return n_succeeded, n_failed


def _write_audit_tag_to_collection(
    collection_id: str | None,
    invocation_id: str | None,
    roboto_client: Any | None,
) -> None:
    """Mode A only: stamp the source Collection with a pointer to this run.

    The tag value is ``audit:<invocation_id>``, which is enough to walk
    back to the audit run that produced it: the invocation's output
    dataset holds the ``audit_reports/<invocation_id>/audit_report.html``
    file. Per-episode verdicts live on the events themselves (see
    ``_write_audit_tags_to_events``); this collection-level tag exists
    purely so anyone who finds the collection can locate the most
    recent audit report.

    Idempotency: add the new pointer first, then remove any other
    ``audit:*`` tag left behind by a prior run, so the collection
    always points at the latest audit. A failure here is logged but
    never raised — the audit report has already been written to disk,
    and the tag can be repaired by re-running.
    """
    if not collection_id:
        # Defensive — Mode A always sets this in source.detail, but if a
        # future caller wires this helper without a collection (e.g. a
        # local invocation that skipped the collection load), don't NPE.
        logger.debug("No collection_id available; skipping collection-tag write")
        return
    if not invocation_id:
        # Local ``invoke-local`` runs without a recorded invocation have
        # no stable handle to point at, so writing a tag would create
        # something un-followable. Skip rather than tag with a junk
        # value.
        logger.info(
            "No invocation_id on context; skipping collection-tag write. "
            "The audit report is on disk under output_dir/audit_reports/."
        )
        return

    new_tag = f"{_AUDIT_TAG_PREFIX}{invocation_id}"
    try:
        collection = roboto.Collection.from_id(
            collection_id, roboto_client=roboto_client
        )
        existing_audit = {
            t for t in (collection.record.tags or [])
            if t.startswith(_AUDIT_TAG_PREFIX)
        }
        to_add = [new_tag] if new_tag not in existing_audit else []
        to_remove = sorted(existing_audit - {new_tag})

        if not to_add and not to_remove:
            logger.info("Collection %s already tagged %s", collection_id, new_tag)
            return

        update_kwargs: dict[str, Any] = {}
        if to_add:
            update_kwargs["add_tags"] = to_add
        if to_remove:
            update_kwargs["remove_tags"] = to_remove
        collection.update(**update_kwargs)
        logger.info(
            "Collection %s tagged %s (replaced: %s)",
            collection_id, new_tag, to_remove or "<none>",
        )
    except Exception:
        logger.exception(
            "Failed to write audit tag to collection %s; "
            "report on disk is unaffected, re-run the audit to repair",
            collection_id,
        )


def _run_mode_a(
    context: roboto.InvocationContext,
    collection_id: str,
    contract_name: str | None,
) -> tuple[list, SourceDescriptor, dict]:
    dataset = getattr(context, "dataset", None)
    if dataset is None:
        raise RuntimeError(
            "Mode A (pre-conversion) requires context.dataset; "
            "invoke the action on a dataset."
        )
    dataset_id = getattr(dataset, "dataset_id", None) or getattr(dataset, "id", None)
    if dataset_id is None:
        raise RuntimeError("Could not resolve dataset_id from context.dataset")

    contract_path = _resolve_contract_path(context, contract_name)
    logger.info("Using contract: %s", contract_path)

    roboto_client = getattr(context, "roboto_client", None)
    return load_pre_conversion_episodes(
        dataset=dataset,
        dataset_id=dataset_id,
        contract_path=contract_path,
        collection_id=collection_id,
        roboto_client=roboto_client,
    )


def _resolve_contract_path(
    context: roboto.InvocationContext, contract_name: str | None
) -> pathlib.Path:
    """Fetch the contract YAML from the Roboto dataset and download it locally.

    Mirrors ``roboto-to-lerobot``'s resolution logic: an explicit ``contract``
    parameter is looked up by path in the dataset; otherwise the dataset is
    scanned for a ``contract.yaml``. The file is then downloaded into
    ``context.input_dir`` so the rest of the pipeline can open it as a path.
    """
    dataset = context.dataset
    if contract_name:
        logger.info("Contract file specified: %s", contract_name)
        contract_file = dataset.get_file_by_path(contract_name)
    else:
        yamls = dataset.list_files(include_patterns=["contract.yaml"])
        try:
            found_yaml = next(yamls)
        except StopIteration as exc:
            raise FileNotFoundError(
                "No contract file found in dataset"
            ) from exc
        # If the iterator still has another match, warn and keep the first.
        try:
            next(yamls)
            logger.warning(
                "Multiple contract files found in dataset. Using %s",
                found_yaml.relative_path,
            )
        except StopIteration:
            pass
        contract_file = dataset.get_file_by_path(found_yaml.relative_path)

    contract_path = pathlib.Path(context.input_dir) / contract_file.relative_path
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_file.download(contract_path)
    return contract_path


def _run_mode_b(
    context: roboto.InvocationContext,
) -> tuple[list, SourceDescriptor, dict]:
    if not context.input_dir:
        raise RuntimeError("Mode B (post-conversion) requires downloaded inputs.")
    return load_post_conversion_episodes(pathlib.Path(context.input_dir))


_FLAG_REASONS: dict[str, str] = {
    "stuck_sensor": (
        "state signal is nearly constant — sensor appears frozen (ACF≈1 at "
        "short lags or std≈0 on a dim)"
    ),
    "alignment_fail": (
        "state↔action peak cross-correlation is ≥2 frames off zero — likely "
        "timestamp misalignment between observations and commands"
    ),
    "low_movement": (
        "at least one action dim is still for >90% of the episode — robot "
        "barely moves"
    ),
    "jerky_motion": (
        "p95 jerk on the action stream is >3× the dataset median — unusually "
        "rough motion (teleop glitch or unsmoothed replay)"
    ),
    "high_action_velocity": (
        "mean |Δaction| for this episode is a z>3 outlier on the high side vs "
        "the dataset's median/MAD — unusually fast or abrupt commanded motion"
    ),
    "variance_outlier": (
        "per-dim std differs from the dataset distribution by |z|>3 — episode "
        "explores a very different range than peers"
    ),
    "outlier_length": (
        "episode length is >3·MAD away from the dataset median — unusually "
        "short or long take"
    ),
    "zero_variance_dim": (
        "at least one state or action dim has std≈0 across the whole episode "
        "— dead channel"
    ),
}


def _reason_for(flag_names: list[str]) -> str:
    parts = [_FLAG_REASONS.get(f, f) for f in flag_names]
    return "; ".join(parts)


def _aggregate_flags(
    episodes: list,
    metric_results: dict[str, MetricResult],
    source: SourceDescriptor,
) -> tuple[list[FlaggedEpisode], str | None]:
    flags_by_episode: dict[int, set[str]] = {ep.episode_index: set() for ep in episodes}
    for metric in metric_results.values():
        for row in metric.flags:
            idx = row.get("episode_index")
            if idx is None:
                continue
            for key, val in row.items():
                if key == "episode_index":
                    continue
                if isinstance(val, bool) and val:
                    flags_by_episode.setdefault(int(idx), set()).add(key)

    flagged = []
    flagged_indices: list[int] = []
    for idx, flag_names in flags_by_episode.items():
        if not flag_names:
            continue
        flagged_indices.append(idx)
        names_sorted = sorted(flag_names)
        flagged.append(
            FlaggedEpisode(
                episode_index=idx,
                flags=names_sorted,
                reason=_reason_for(names_sorted),
            )
        )
    flagged.sort(key=lambda x: x.episode_index)

    repo_id = _guess_repo_id(source)
    cli = build_cleanup_command(repo_id, flagged_indices) if repo_id else None
    return flagged, cli


def _guess_repo_id(source: SourceDescriptor) -> str | None:
    if source.kind == "lerobot_dataset":
        return source.identifier
    return None


def _build_summary(
    episodes: list,
    mode: str,
    state_spec,
    action_spec,
    metric_results: dict[str, MetricResult],
    n_flagged: int,
    codebase_version: str | None,
) -> DatasetSummary:
    ess_total = None
    ess_res = metric_results.get("effective_sample_size")
    if ess_res is not None:
        ess_total = safe_float(
            ess_res.per_dataset.get("total_ess_action_approx", float("nan"))
        )
        if not np.isfinite(ess_total):
            ess_total = None

    # `low_effective_dim` is a dataset-level PCA verdict (see
    # effective_dim.py) — surfaced here rather than as a per-episode flag so
    # it doesn't broadcast onto every episode's audit tags.
    low_effective_dim = None
    dim_res = metric_results.get("effective_dimensionality")
    if dim_res is not None:
        low_effective_dim = dim_res.per_dataset.get("low_effective_dim_flag")

    total_frames = int(sum(int(ep.n_frames) for ep in episodes))
    fps_values = [ep.fps for ep in episodes if ep.fps and np.isfinite(ep.fps)]
    fps = float(np.median(fps_values)) if fps_values else 0.0

    return DatasetSummary(
        mode=mode,  # type: ignore[arg-type]
        n_episodes=len(episodes),
        n_frames=total_frames,
        fps=fps,
        state_dim=state_spec.dim if state_spec else None,
        action_dim=action_spec.dim if action_spec else None,
        ess_total=ess_total,
        n_flagged_episodes=n_flagged,
        low_effective_dim=low_effective_dim,
        codebase_version=codebase_version,
    )
