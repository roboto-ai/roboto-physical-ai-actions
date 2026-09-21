#!/usr/bin/env python3
"""Episode-ordering-aware equivalence check for two roboto-to-lerobot outputs.

Companion to ``verify_byte_equivalence.py``. The byte-equivalence verifier
gates the conversion-perf PR stack against a same-action baseline; it
assumes both sides processed events in the same order and so episode indices
line up file-by-file.

This script relaxes that assumption: it matches episodes by
``(start_time_ns, end_time_ns, n_frames)`` from each side's
``manifest.json#episode_to_event`` and then compares the matched pairs.
Useful when comparing a single-shot ``roboto-to-lerobot`` run against the
``roboto-to-lerobot`` (sharded) + ``lerobot-merge`` pipeline, where the
merge stacks shards in an order unrelated to the single-shot's per-event
traversal order.

Inputs (same shape as ``verify_byte_equivalence.py``)::

    <dir>/manifest.json
    <dir>/combined/meta/info.json
    <dir>/combined/meta/stats.json
    <dir>/combined/meta/tasks.parquet
    <dir>/combined/meta/episodes/chunk-*/file-*.parquet
    <dir>/combined/data/chunk-*/file-*.parquet
    <dir>/combined/videos/<key>/chunk-*/file-*.mp4

Checks:

* ``info.json``: parsed-JSON equality (fps, features, totals, paths).
* Per-episode data: groupby ``episode_index`` on each side, then for each
  matched pair compare the frame columns modulo
  ``{episode_index, index, task_index}`` (renumbered by concatenation order).
* ``meta/episodes/`` parquet: reorder candidate rows to reference order via
  the episode mapping, then compare modulo concatenation-position fields
  (``episode_index``, ``dataset_{from,to}_index``,
  ``data/{chunk,file}_index``, ``videos/.../{chunk,file}_index``).
* ``meta/stats.json``: parsed-JSON equality (dataset-level aggregate, so
  reordering shouldn't matter).
* ``meta/tasks.parquet``: structural shape (same task count). Task strings
  themselves are reported as info, not failures — the two runs can have
  been driven by event tags whose task descriptions legitimately differ.
* mp4s under ``combined/videos/``: per-episode frame-range PSNR. Each
  episode's frame slice is derived from ``meta/episodes/`` parquet's
  ``videos/<key>/{from,to}_timestamp`` × ``fps``; matched pairs are PSNR-
  compared at the same per-episode offset.
* ``manifest.json``: contract.sha256 must match across runs. Other fields
  legitimately differ (single-shot vs merge action produce different
  manifest shapes).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq

# Reuse the same PSNR floor as verify_byte_equivalence.py. Calibration
# rationale (same-host SVT-AV1 noise floor) carries over: a real perceptual
# regression drops well below 30 dB; a clean re-encode of the same source
# frames stays well above 50 dB.
MP4_PSNR_FLOOR_DB: float = 50.0

# Frame columns whose values are concatenation-position artifacts, not
# per-frame payload. ``episode_index`` and ``index`` are renumbered by
# stacking order; ``task_index`` points into ``tasks.parquet`` whose entry
# strings can legitimately differ between runs (we surface that as info,
# not a failure).
_DATA_IGNORE_COLS: frozenset[str] = frozenset({"episode_index", "index", "task_index"})

# Episode-row columns whose values are concatenation-position artifacts.
# ``dataset_{from,to}_index`` is the cumulative offset into the
# concatenated data parquet, set by the order episodes are written; the
# chunk/file indices are pure addressing in the same parquet/mp4 file.
# Everything else (length, tasks, stats/*) is per-episode payload.
_EPISODE_POSITION_COL_SUFFIXES: tuple[str, ...] = ("chunk_index", "file_index")
_EPISODE_POSITION_COLS: frozenset[str] = frozenset({
    "episode_index",
    "dataset_from_index",
    "dataset_to_index",
})

# Float tolerance for stats fields (per-episode min/max/mean/std/quantiles).
# Aggregate reductions over the same float32 inputs can drift in the last
# few mantissa bits depending on summation order; per-episode stats sum
# over identical frame slices so should be bit-identical, but the comparison
# is still tolerance-based to insulate against any future re-ordering.
_FLOAT_ATOL: float = 1e-6
_FLOAT_RTOL: float = 1e-6

# Per-episode frame sample for PSNR. Matches verify_byte_equivalence's
# head/middle/tail × N pattern — same rationale: catch regressions that
# surface after the opening GOP.
_VIDEO_FRAMES_PSNR_SAMPLE: int = 5


@dataclasses.dataclass(slots=True)
class _Failure:
    path: str
    detail: str

    def __str__(self) -> str:
        return f"  - {self.path}: {self.detail}"


def _episode_key(entry: dict) -> tuple[int, int, int]:
    """Time-window identity of an episode_to_event entry.

    ``event_id`` would be more direct, but it can drift between runs if
    upstream events were regenerated (different IDs for the same time
    window). The (start_ns, end_ns, n_frames) tuple is what actually
    identifies the slice of the source dataset.
    """
    return (
        int(entry["start_time_ns"]),
        int(entry["end_time_ns"]),
        int(entry["n_frames"]),
    )


def _build_episode_mapping(
    cand_manifest: dict, ref_manifest: dict,
) -> tuple[dict[int, int], list[_Failure]]:
    """Map candidate episode_index → reference episode_index via time-key.

    Returns ``(mapping, failures)``. Any unmatched episodes on either side
    show up as failures; the mapping covers only matched pairs.
    """
    failures: list[_Failure] = []
    cand_e2e = cand_manifest.get("episode_to_event") or []
    ref_e2e = ref_manifest.get("episode_to_event") or []
    if not cand_e2e:
        failures.append(_Failure(
            path="manifest.episode_to_event",
            detail="candidate manifest has no episode_to_event entries",
        ))
    if not ref_e2e:
        failures.append(_Failure(
            path="manifest.episode_to_event",
            detail="reference manifest has no episode_to_event entries",
        ))

    ref_by_key: dict[tuple[int, int, int], int] = {}
    for e in ref_e2e:
        ref_by_key[_episode_key(e)] = int(e["episode_index"])

    mapping: dict[int, int] = {}
    used_ref_indices: set[int] = set()
    for e in cand_e2e:
        k = _episode_key(e)
        if k not in ref_by_key:
            failures.append(_Failure(
                path=f"episode cand[{e['episode_index']}]",
                detail=f"no reference episode with key {k}",
            ))
            continue
        ref_idx = ref_by_key[k]
        if ref_idx in used_ref_indices:
            failures.append(_Failure(
                path=f"episode cand[{e['episode_index']}]",
                detail=f"reference episode {ref_idx} already mapped (duplicate time-key)",
            ))
            continue
        mapping[int(e["episode_index"])] = ref_idx
        used_ref_indices.add(ref_idx)

    unmatched_ref = sorted(
        int(e["episode_index"]) for e in ref_e2e
        if int(e["episode_index"]) not in used_ref_indices
    )
    for idx in unmatched_ref:
        failures.append(_Failure(
            path=f"episode ref[{idx}]",
            detail="no candidate episode matched this reference episode",
        ))
    return mapping, failures


def _read_single_parquet(combined: Path, relglob: str) -> Path:
    """Return the single parquet file matched by ``relglob`` under ``combined``."""
    matches = sorted((combined).glob(relglob))
    if len(matches) != 1:
        raise SystemExit(
            f"Expected exactly one parquet matching {relglob} under {combined}; "
            f"found {len(matches)}: {[str(m.relative_to(combined)) for m in matches]}"
        )
    return matches[0]


def _compare_info_json(cand_combined: Path, ref_combined: Path) -> list[_Failure]:
    cand = json.loads((cand_combined / "meta" / "info.json").read_text())
    ref = json.loads((ref_combined / "meta" / "info.json").read_text())
    if cand == ref:
        return []
    diffs: list[str] = []
    for key in sorted(set(cand) | set(ref)):
        if cand.get(key) != ref.get(key):
            diffs.append(key)
    return [_Failure(
        path="combined/meta/info.json",
        detail=f"fields differ: {', '.join(diffs)}",
    )]


def _compare_stats_json(cand_combined: Path, ref_combined: Path) -> list[_Failure]:
    """Dataset-level aggregate stats — invariant under per-episode reordering.

    Strips the same bookkeeping-column stats (``episode_index``, ``index``)
    that the episodes-parquet comparator skips: their values reflect
    concatenation order, not data content, so flagging them here would
    surface as a false positive on every reorder.

    Float values are compared with ``_FLOAT_ATOL``/``_FLOAT_RTOL`` tolerance:
    aggregate reductions (mean/std/quantiles) over float32 inputs drift in
    the last few ULPs depending on summation order, which differs between
    a single-shot reduce and a per-shard reduce-then-merge. The drift is
    far below any meaningful regression signal.
    """
    cand = json.loads((cand_combined / "meta" / "stats.json").read_text())
    ref = json.loads((ref_combined / "meta" / "stats.json").read_text())
    for d in (cand, ref):
        d.pop("episode_index", None)
        d.pop("index", None)
    diffs: list[str] = []
    for key in sorted(set(cand) | set(ref)):
        sub_diffs = _diff_json_with_tolerance(cand.get(key), ref.get(key), prefix=key)
        diffs.extend(sub_diffs)
    if not diffs:
        return []
    return [_Failure(
        path="combined/meta/stats.json",
        detail=f"{len(diffs)} field(s) differ beyond float tolerance: {diffs[:8]}"
               + (" ..." if len(diffs) > 8 else ""),
    )]


def _diff_json_with_tolerance(a, b, *, prefix: str) -> list[str]:
    """Recurse into nested JSON; return path strings for values that differ.

    Floats (including lists of floats) use ``np.allclose`` with the same
    tolerance the parquet comparator uses. Everything else falls back to
    Python equality.
    """
    if isinstance(a, dict) or isinstance(b, dict):
        a_d = a if isinstance(a, dict) else {}
        b_d = b if isinstance(b, dict) else {}
        out: list[str] = []
        for k in sorted(set(a_d) | set(b_d)):
            out.extend(_diff_json_with_tolerance(a_d.get(k), b_d.get(k), prefix=f"{prefix}.{k}"))
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{prefix} (list length {len(a)} vs {len(b)})"]
        if all(isinstance(x, (int, float)) for x in a + b):
            aa = np.asarray(a, dtype=float)
            bb = np.asarray(b, dtype=float)
            if np.allclose(aa, bb, atol=_FLOAT_ATOL, rtol=_FLOAT_RTOL, equal_nan=True):
                return []
            return [f"{prefix} (max |Δ| = {float(np.nanmax(np.abs(aa - bb))):.3g})"]
        out = []
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            out.extend(_diff_json_with_tolerance(x, y, prefix=f"{prefix}[{i}]"))
        return out
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            return [f"{prefix} (one side missing)"]
        if math.isclose(float(a), float(b), abs_tol=_FLOAT_ATOL, rel_tol=_FLOAT_RTOL):
            return []
        return [f"{prefix} ({a!r} vs {b!r})"]
    if a == b:
        return []
    return [f"{prefix} ({a!r} vs {b!r})"]


def _compare_data_parquet(
    cand_combined: Path, ref_combined: Path, mapping: dict[int, int],
) -> list[_Failure]:
    """Per-episode frame-payload equivalence, modulo position columns.

    Slices each side's concatenated data parquet by ``episode_index``, then
    compares the matched pair column-by-column ignoring renumbered
    bookkeeping (``episode_index``, ``index``, ``task_index``).
    """
    cand_path = _read_single_parquet(cand_combined, "data/chunk-*/file-*.parquet")
    ref_path = _read_single_parquet(ref_combined, "data/chunk-*/file-*.parquet")

    cand_tbl = pq.read_table(cand_path).to_pandas()
    ref_tbl = pq.read_table(ref_path).to_pandas()

    failures: list[_Failure] = []

    cand_groups = cand_tbl.groupby("episode_index", sort=True)
    ref_groups = ref_tbl.groupby("episode_index", sort=True)

    common_cols = sorted(
        (set(cand_tbl.columns) & set(ref_tbl.columns)) - _DATA_IGNORE_COLS
    )
    extra_cand = sorted(set(cand_tbl.columns) - set(ref_tbl.columns))
    extra_ref = sorted(set(ref_tbl.columns) - set(cand_tbl.columns))
    if extra_cand:
        failures.append(_Failure(
            path="combined/data/*.parquet",
            detail=f"columns present only in candidate: {extra_cand}",
        ))
    if extra_ref:
        failures.append(_Failure(
            path="combined/data/*.parquet",
            detail=f"columns present only in reference: {extra_ref}",
        ))

    for cand_idx, ref_idx in sorted(mapping.items()):
        try:
            cand_block = cand_groups.get_group(cand_idx).reset_index(drop=True)
        except KeyError:
            failures.append(_Failure(
                path=f"combined/data/*.parquet episode cand[{cand_idx}]",
                detail="no rows in candidate data parquet for this episode_index",
            ))
            continue
        try:
            ref_block = ref_groups.get_group(ref_idx).reset_index(drop=True)
        except KeyError:
            failures.append(_Failure(
                path=f"combined/data/*.parquet episode ref[{ref_idx}]",
                detail="no rows in reference data parquet for this episode_index",
            ))
            continue
        if len(cand_block) != len(ref_block):
            failures.append(_Failure(
                path=f"combined/data/*.parquet episode cand[{cand_idx}]↔ref[{ref_idx}]",
                detail=(
                    f"frame count differs (cand={len(cand_block)}, "
                    f"ref={len(ref_block)})"
                ),
            ))
            continue
        for col in common_cols:
            mismatch = _compare_pandas_column(cand_block[col], ref_block[col])
            if mismatch is not None:
                failures.append(_Failure(
                    path=f"combined/data/*.parquet episode cand[{cand_idx}]↔ref[{ref_idx}] col={col}",
                    detail=mismatch,
                ))
    return failures


def _compare_pandas_column(a, b) -> str | None:
    """Return ``None`` on equivalence, else a short diff description."""
    if len(a) != len(b):
        return f"length differs ({len(a)} vs {len(b)})"
    # Tolerate list-typed columns (observation.state, action): pyarrow loads
    # fixed_size_list and list-of-float identically through to_pandas() as
    # numpy arrays nested in object dtype, so we coerce both sides through
    # np.asarray and compare elementwise with float tolerance.
    try:
        a_arr = np.stack([np.asarray(v) for v in a.to_numpy()])
        b_arr = np.stack([np.asarray(v) for v in b.to_numpy()])
    except (TypeError, ValueError):
        # Fallback for scalar columns or anything that can't stack cleanly.
        if (a.to_numpy() == b.to_numpy()).all():
            return None
        return "values differ (non-numeric/scalar comparison)"
    if a_arr.shape != b_arr.shape:
        return f"shape differs ({a_arr.shape} vs {b_arr.shape})"
    if np.issubdtype(a_arr.dtype, np.floating) or np.issubdtype(b_arr.dtype, np.floating):
        if np.allclose(a_arr, b_arr, atol=_FLOAT_ATOL, rtol=_FLOAT_RTOL, equal_nan=True):
            return None
        max_abs = float(np.nanmax(np.abs(a_arr - b_arr)))
        return f"float values differ (max |Δ| = {max_abs:.3g})"
    if np.array_equal(a_arr, b_arr):
        return None
    return "values differ"


def _compare_episodes_parquet(
    cand_combined: Path, ref_combined: Path, mapping: dict[int, int],
) -> list[_Failure]:
    """Per-episode metadata (length, tasks, stats/*) modulo concat position."""
    cand_path = _read_single_parquet(cand_combined, "meta/episodes/chunk-*/file-*.parquet")
    ref_path = _read_single_parquet(ref_combined, "meta/episodes/chunk-*/file-*.parquet")
    cand_tbl = pq.read_table(cand_path).to_pandas().set_index("episode_index", drop=False)
    ref_tbl = pq.read_table(ref_path).to_pandas().set_index("episode_index", drop=False)

    failures: list[_Failure] = []
    common_cols = sorted(set(cand_tbl.columns) & set(ref_tbl.columns))
    extra_cand = sorted(set(cand_tbl.columns) - set(ref_tbl.columns))
    extra_ref = sorted(set(ref_tbl.columns) - set(cand_tbl.columns))
    if extra_cand:
        failures.append(_Failure(
            path="combined/meta/episodes/*.parquet",
            detail=f"columns present only in candidate: {extra_cand}",
        ))
    if extra_ref:
        failures.append(_Failure(
            path="combined/meta/episodes/*.parquet",
            detail=f"columns present only in reference: {extra_ref}",
        ))

    # Skip columns whose values are concatenation-position artifacts rather
    # than per-episode payload:
    #   - position bookkeeping (episode_index, dataset_{from,to}_index,
    #     */chunk_index, */file_index)
    #   - video frame-range timestamps (from/to_timestamp into the concatenated
    #     mp4) — consumed separately by the video comparator
    #   - stats over the renumbered bookkeeping columns themselves:
    #     stats/episode_index/* and stats/index/* are min/max/mean/quantiles
    #     of the per-frame ``episode_index`` and ``index`` columns within each
    #     episode. Those underlying columns are already in
    #     ``_DATA_IGNORE_COLS`` because they're renumbered by stacking order;
    #     stats over them inherit that same position-dependence.
    def is_position(col: str) -> bool:
        if col in _EPISODE_POSITION_COLS:
            return True
        if col.startswith("stats/episode_index/") or col.startswith("stats/index/"):
            return True
        return col.endswith(_EPISODE_POSITION_COL_SUFFIXES) or (
            col.startswith("videos/") and (
                col.endswith("/from_timestamp") or col.endswith("/to_timestamp")
            )
        )

    payload_cols = [c for c in common_cols if not is_position(c)]

    # The per-episode ``tasks`` column repeats whatever string lives in
    # ``meta/tasks.parquet``. When the upstream events have legitimately
    # different task descriptions, every single episode flags a diff here;
    # we already surface that once globally from the tasks.parquet comparator,
    # so squash per-episode duplicates of the same diff.
    cand_tasks_global: set[str] = set()
    ref_tasks_global: set[str] = set()
    if "tasks" in payload_cols:
        for cand_idx, ref_idx in mapping.items():
            if cand_idx in cand_tbl.index:
                cand_tasks_global.update(str(t) for t in np.atleast_1d(cand_tbl.loc[cand_idx, "tasks"]))
            if ref_idx in ref_tbl.index:
                ref_tasks_global.update(str(t) for t in np.atleast_1d(ref_tbl.loc[ref_idx, "tasks"]))
    global_task_diff = bool(cand_tasks_global and ref_tasks_global and cand_tasks_global != ref_tasks_global)

    for cand_idx, ref_idx in sorted(mapping.items()):
        if cand_idx not in cand_tbl.index or ref_idx not in ref_tbl.index:
            failures.append(_Failure(
                path=f"combined/meta/episodes/*.parquet cand[{cand_idx}]↔ref[{ref_idx}]",
                detail="episode_index missing in one side's episodes parquet",
            ))
            continue
        cand_row = cand_tbl.loc[cand_idx]
        ref_row = ref_tbl.loc[ref_idx]
        for col in payload_cols:
            cv = cand_row[col]
            rv = ref_row[col]
            mismatch = _compare_episode_cell(cv, rv)
            if mismatch is None:
                continue
            if col == "tasks" and global_task_diff:
                continue
            failures.append(_Failure(
                path=f"combined/meta/episodes/*.parquet cand[{cand_idx}]↔ref[{ref_idx}] col={col}",
                detail=mismatch,
            ))
    return failures


def _compare_episode_cell(cv, rv) -> str | None:
    """Compare a single cell from the episodes parquet."""
    try:
        cv_arr = np.asarray(cv)
        rv_arr = np.asarray(rv)
    except Exception:
        if cv == rv:
            return None
        return f"values differ (cand={cv!r} ref={rv!r})"
    if cv_arr.shape != rv_arr.shape:
        return f"shape differs ({cv_arr.shape} vs {rv_arr.shape})"
    if np.issubdtype(cv_arr.dtype, np.floating) or np.issubdtype(rv_arr.dtype, np.floating):
        if np.allclose(cv_arr, rv_arr, atol=_FLOAT_ATOL, rtol=_FLOAT_RTOL, equal_nan=True):
            return None
        max_abs = float(np.nanmax(np.abs(cv_arr.astype(float) - rv_arr.astype(float))))
        return f"float values differ (max |Δ| = {max_abs:.3g})"
    if np.array_equal(cv_arr, rv_arr):
        return None
    return f"values differ (cand={cv!r} ref={rv!r})"


def _compare_tasks_parquet(cand_combined: Path, ref_combined: Path) -> tuple[list[_Failure], list[str]]:
    """tasks.parquet: count must match. Per-row task strings reported as info."""
    cand_tbl = pq.read_table(cand_combined / "meta" / "tasks.parquet").to_pandas()
    ref_tbl = pq.read_table(ref_combined / "meta" / "tasks.parquet").to_pandas()
    notes: list[str] = []
    failures: list[_Failure] = []
    if len(cand_tbl) != len(ref_tbl):
        failures.append(_Failure(
            path="combined/meta/tasks.parquet",
            detail=f"row count differs (cand={len(cand_tbl)}, ref={len(ref_tbl)})",
        ))
        return failures, notes
    if not cand_tbl.equals(ref_tbl):
        try:
            cand_tasks = sorted(cand_tbl["task"].tolist()) if "task" in cand_tbl.columns else (
                sorted(cand_tbl.index.tolist())
            )
            ref_tasks = sorted(ref_tbl["task"].tolist()) if "task" in ref_tbl.columns else (
                sorted(ref_tbl.index.tolist())
            )
        except KeyError:
            cand_tasks = []
            ref_tasks = []
        notes.append(
            f"task strings differ: cand={cand_tasks} ref={ref_tasks} "
            "(not a failure; treated as upstream-event metadata diff)"
        )
    return failures, notes


def _decode_selected_frames(
    path: Path, wanted: set[int],
) -> dict[int, np.ndarray]:
    """Return ``{frame_index: bgr24 array}`` for only the requested indices.

    Concatenated lerobot mp4s hold every episode's frames in one stream —
    ~12k frames × 480×854×3 ≈ 14 GB if you naively keep all decoded frames
    in memory, and we have four cameras per side. We sample a few frames per
    matched episode pair, so a single sequential decode that drops anything
    not in ``wanted`` stays bounded at len(wanted) frames in RAM.
    """
    out: dict[int, np.ndarray] = {}
    if not wanted:
        return out
    last = max(wanted)
    container = av.open(str(path))
    try:
        for idx, frame in enumerate(container.decode(video=0)):
            if idx in wanted:
                out[idx] = frame.to_ndarray(format="bgr24")
            if idx >= last:
                break
    finally:
        container.close()
    return out


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("nan")
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10(255.0 * 255.0 / mse)


def _episode_frame_ranges(
    episodes_parquet: Path, video_key: str, fps: int,
) -> dict[int, tuple[int, int]]:
    """Per-episode (start_frame, end_frame_exclusive) for one video stream."""
    tbl = pq.read_table(episodes_parquet).to_pandas()
    from_col = f"videos/{video_key}/from_timestamp"
    to_col = f"videos/{video_key}/to_timestamp"
    if from_col not in tbl.columns or to_col not in tbl.columns:
        raise SystemExit(
            f"episodes parquet {episodes_parquet} is missing {from_col} / {to_col}; "
            f"available columns: {sorted(tbl.columns)}"
        )
    ranges: dict[int, tuple[int, int]] = {}
    for _, row in tbl.iterrows():
        start = round(float(row[from_col]) * fps)
        end = round(float(row[to_col]) * fps)
        ranges[int(row["episode_index"])] = (start, end)
    return ranges


def _sample_indices(start: int, end: int, sample: int) -> list[int]:
    """Pick up to ``3 × sample`` indices from [start, end) — head/middle/tail.

    Mirrors verify_byte_equivalence's PSNR sampling shape: the same regression
    surfaces (rate-control drift, lookahead changes) we worry about within a
    full-episode mp4 also apply to a per-episode frame slice of a longer mp4.
    """
    n = end - start
    if n <= 0:
        return []
    head = range(start, start + min(sample, n))
    tail = range(max(start, end - sample), end)
    mid_anchor = start + n // 2
    mid_start = max(start, mid_anchor - sample // 2)
    mid = range(mid_start, min(mid_start + sample, end))
    return sorted(set(head) | set(mid) | set(tail))


def _compare_videos(
    cand_combined: Path,
    ref_combined: Path,
    mapping: dict[int, int],
    fps: int,
    psnr_floor_db: float,
    sample: int,
) -> tuple[list[_Failure], list[str]]:
    """Per-episode PSNR comparison of every shared mp4 stream."""
    cand_videos = cand_combined / "videos"
    ref_videos = ref_combined / "videos"
    if not cand_videos.is_dir() or not ref_videos.is_dir():
        return [], ["(no combined/videos/ on one side; skipping video comparison)"]

    cand_relpaths = {p.relative_to(cand_videos) for p in cand_videos.rglob("*.mp4")}
    ref_relpaths = {p.relative_to(ref_videos) for p in ref_videos.rglob("*.mp4")}

    failures: list[_Failure] = []
    notes: list[str] = []

    for rel in sorted(cand_relpaths - ref_relpaths):
        failures.append(_Failure(
            path=f"combined/videos/{rel}",
            detail="present in candidate, missing in reference",
        ))
    for rel in sorted(ref_relpaths - cand_relpaths):
        failures.append(_Failure(
            path=f"combined/videos/{rel}",
            detail="present in reference, missing in candidate",
        ))

    shared = sorted(cand_relpaths & ref_relpaths)
    if not shared:
        return failures, notes

    cand_eps_pq = _read_single_parquet(cand_combined, "meta/episodes/chunk-*/file-*.parquet")
    ref_eps_pq = _read_single_parquet(ref_combined, "meta/episodes/chunk-*/file-*.parquet")

    for rel in shared:
        print(f"  comparing video {rel} ...", flush=True)
        # rel looks like 'observation.images.left/chunk-000/file-000.mp4';
        # the video key is the first path component.
        video_key = rel.parts[0]
        cand_ranges = _episode_frame_ranges(cand_eps_pq, video_key, fps)
        ref_ranges = _episode_frame_ranges(ref_eps_pq, video_key, fps)

        # First pass over the mapping: figure out which absolute frame
        # indices we want from each side, and remember the (cand_abs, ref_abs)
        # pairs we'll PSNR-compare. We never hold the full decoded video in
        # memory — at this dataset's size that's ~14 GB per video × 8 streams.
        pair_list: list[tuple[int, int, int, int]] = []  # (cand_idx, ref_idx, ci, ri)
        cand_wanted: set[int] = set()
        ref_wanted: set[int] = set()
        range_failures: list[_Failure] = []
        for cand_idx, ref_idx in sorted(mapping.items()):
            c_start, c_end = cand_ranges.get(cand_idx, (0, 0))
            r_start, r_end = ref_ranges.get(ref_idx, (0, 0))
            c_len = c_end - c_start
            r_len = r_end - r_start
            if c_len != r_len:
                range_failures.append(_Failure(
                    path=f"combined/videos/{rel} cand[{cand_idx}]↔ref[{ref_idx}]",
                    detail=f"frame-range length differs (cand={c_len}, ref={r_len})",
                ))
                continue
            if c_len == 0:
                continue
            for offset in _sample_indices(0, c_len, sample):
                ci = c_start + offset
                ri = r_start + offset
                pair_list.append((cand_idx, ref_idx, ci, ri))
                cand_wanted.add(ci)
                ref_wanted.add(ri)
        failures.extend(range_failures)

        cand_decoded = _decode_selected_frames(cand_videos / rel, cand_wanted)
        ref_decoded = _decode_selected_frames(ref_videos / rel, ref_wanted)

        worst_psnr = float("inf")
        episode_broken: set[tuple[int, int]] = set()
        for cand_idx, ref_idx, ci, ri in pair_list:
            if (cand_idx, ref_idx) in episode_broken:
                continue
            cf = cand_decoded.get(ci)
            rf = ref_decoded.get(ri)
            if cf is None or rf is None:
                failures.append(_Failure(
                    path=f"combined/videos/{rel} cand[{cand_idx}]↔ref[{ref_idx}]#cand_frame{ci}/ref_frame{ri}",
                    detail=(
                        "frame index out of bounds during decode "
                        f"(cand_has={cf is not None}, ref_has={rf is not None})"
                    ),
                ))
                episode_broken.add((cand_idx, ref_idx))
                continue
            psnr = _psnr(cf, rf)
            if math.isnan(psnr):
                failures.append(_Failure(
                    path=f"combined/videos/{rel} cand[{cand_idx}]↔ref[{ref_idx}]#cand_frame{ci}/ref_frame{ri}",
                    detail=f"shape mismatch (cand={cf.shape}, ref={rf.shape})",
                ))
                episode_broken.add((cand_idx, ref_idx))
                continue
            if math.isfinite(psnr):
                worst_psnr = min(worst_psnr, psnr)
            if psnr < psnr_floor_db:
                failures.append(_Failure(
                    path=f"combined/videos/{rel} cand[{cand_idx}]↔ref[{ref_idx}]#cand_frame{ci}/ref_frame{ri}",
                    detail=f"PSNR {psnr:.2f} dB < floor {psnr_floor_db:.2f} dB",
                ))
                episode_broken.add((cand_idx, ref_idx))
        if math.isfinite(worst_psnr):
            notes.append(f"video {rel}: worst per-episode PSNR = {worst_psnr:.2f} dB")
        else:
            notes.append(f"video {rel}: all sampled frames byte-identical (PSNR = inf)")
    return failures, notes


def _compare_manifest_contract(cand: Path, ref: Path) -> list[_Failure]:
    cand_m = json.loads((cand / "manifest.json").read_text())
    ref_m = json.loads((ref / "manifest.json").read_text())
    cand_sha = (cand_m.get("contract") or {}).get("sha256")
    ref_sha = (ref_m.get("contract") or {}).get("sha256")
    if cand_sha != ref_sha:
        return [_Failure(
            path="manifest.json:contract.sha256",
            detail=f"differs (cand={cand_sha}, ref={ref_sha}) — datasets were "
                   "produced from different contracts and are not directly comparable",
        )]
    return []


def verify(cand: Path, ref: Path, *, psnr_floor_db: float, psnr_sample: int) -> tuple[list[_Failure], list[str]]:
    cand_combined = cand / "combined"
    ref_combined = ref / "combined"
    if not cand_combined.is_dir():
        raise SystemExit(f"candidate missing combined/: {cand}")
    if not ref_combined.is_dir():
        raise SystemExit(f"reference missing combined/: {ref}")

    failures: list[_Failure] = []
    notes: list[str] = []

    failures.extend(_compare_manifest_contract(cand, ref))

    cand_manifest = json.loads((cand / "manifest.json").read_text())
    ref_manifest = json.loads((ref / "manifest.json").read_text())
    mapping, map_failures = _build_episode_mapping(cand_manifest, ref_manifest)
    failures.extend(map_failures)
    notes.append(
        f"matched {len(mapping)} episode(s) by (start_ns, end_ns, n_frames); "
        f"identity={'yes' if all(c == r for c, r in mapping.items()) else 'no'}"
    )

    failures.extend(_compare_info_json(cand_combined, ref_combined))
    failures.extend(_compare_stats_json(cand_combined, ref_combined))
    failures.extend(_compare_data_parquet(cand_combined, ref_combined, mapping))
    failures.extend(_compare_episodes_parquet(cand_combined, ref_combined, mapping))
    task_failures, task_notes = _compare_tasks_parquet(cand_combined, ref_combined)
    failures.extend(task_failures)
    notes.extend(task_notes)

    info = json.loads((cand_combined / "meta" / "info.json").read_text())
    fps = int(info.get("fps", 30))

    video_failures, video_notes = _compare_videos(
        cand_combined, ref_combined, mapping, fps,
        psnr_floor_db=psnr_floor_db, sample=psnr_sample,
    )
    failures.extend(video_failures)
    notes.extend(video_notes)

    return failures, notes


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--psnr-floor-db", type=float, default=MP4_PSNR_FLOOR_DB)
    parser.add_argument("--psnr-sample", type=int, default=_VIDEO_FRAMES_PSNR_SAMPLE)
    args = parser.parse_args(argv)

    failures, notes = verify(
        args.candidate.resolve(), args.reference.resolve(),
        psnr_floor_db=args.psnr_floor_db, psnr_sample=args.psnr_sample,
    )
    for note in notes:
        print(f"  note: {note}")
    if failures:
        print(f"FAIL: {len(failures)} difference(s) between {args.candidate} and {args.reference}")
        for f in failures:
            print(f)
        return 1
    print(
        f"PASS: {args.candidate} ≡ {args.reference} "
        f"(episode-mapping reordered; per-episode parquet exact, "
        f"mp4 PSNR ≥ {args.psnr_floor_db:.2f} dB)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
