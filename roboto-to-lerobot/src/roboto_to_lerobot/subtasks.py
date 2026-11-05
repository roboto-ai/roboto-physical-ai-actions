"""Turn the events on a session into LeRobot sub-tasks.

Each event on a session in the input collection labels the frames of that
session's episode, using the event's name as the sub-task. The labels are
written the way lerobot >= 0.6 reads them: ``style="subtask"`` rows in the
``language_persistent`` column. The ``subtask_index`` /
``meta/subtasks.parquet`` layout some LeRobot docs still describe is the legacy
one, and lerobot's own annotation writer drops it.

A persistent row stays active until the next row supersedes it, so it has a
start but no end, while an event has both. :func:`plan_segments` flattens an
episode's events into one label per frame and marks the frames no event covers
with :data:`GAP_SUBTASK`; without that row, a sub-task would appear to run on
through the gap after its event ended.

The columns are added after the dataset is finalized and merged, so sharded
writing and video encoding are unaffected. A dataset whose sessions have no
events is left exactly as it was written.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import roboto
from roboto.exceptions import RobotoFeatureNotAvailableException

from .contract_utils import fps_to_time_step_ns
from .logger import logger

GAP_SUBTASK = "no_subtask"
"""The sub-task of frames that no event on the session covers."""


@dataclass(frozen=True)
class SubtaskWindow:
    """One event, as the span of time it labels."""

    event_id: str
    name: str
    start_time: int
    end_time: int


@dataclass(frozen=True)
class SubtaskSegment:
    """A run of frames sharing a sub-task, from ``start_frame`` to the next segment."""

    content: str
    start_frame: int
    event_id: str | None


def session_subtask_windows(
    session: Any,
    session_files: Iterable[Any],
) -> tuple[SubtaskWindow, ...]:
    """Return the events that label ``session``, ordered by start time.

    Uses the events on the session itself. An org that has no session events,
    or a session with none, falls back to the legacy events on the session's
    files, which are the events a migration would move onto the session.
    """
    try:
        events = list(session.list_events())
    except RobotoFeatureNotAvailableException:
        events = []

    if not events:
        by_id = {
            event.event_id: event
            for session_file in session_files
            for event in roboto.Event.get_by_file(session_file.file_id)
        }
        events = list(by_id.values())

    windows = (
        SubtaskWindow(e.event_id, e.name, int(e.start_time), int(e.end_time))
        for e in events
    )
    return tuple(sorted(windows, key=lambda w: (w.start_time, w.event_id)))


def plan_segments(
    windows: Iterable[SubtaskWindow],
    *,
    episode_start_ns: int,
    n_frames: int,
    time_step_ns: int,
) -> tuple[list[SubtaskSegment], list[SubtaskWindow]]:
    """Flatten overlapping windows into one sub-task per frame.

    Where windows overlap, the one that started last is active, and the earlier
    one resumes if it outlasts it. Windows are clipped to the episode.

    Returns:
        The segments, empty if no window covers a frame, and the windows that
        cover no frame: those outside the episode, and instantaneous events
        shorter than a frame.
    """

    def frame_of(time_ns: int) -> int:
        return round((time_ns - episode_start_ns) / time_step_ns)

    active: list[SubtaskWindow | None] = [None] * n_frames
    dropped: list[SubtaskWindow] = []
    # Longest first among windows that start together, so the shorter one stays visible.
    for window in sorted(windows, key=lambda w: (w.start_time, -w.end_time, w.event_id)):
        first = max(frame_of(window.start_time), 0)
        end = min(frame_of(window.end_time), n_frames)
        if first >= end:
            dropped.append(window)
            continue
        active[first:end] = [window] * (end - first)

    if all(window is None for window in active):
        return [], dropped

    segments: list[SubtaskSegment] = []
    previous: SubtaskWindow | None = None
    for frame, window in enumerate(active):
        if frame > 0 and window is previous:
            continue
        previous = window
        segments.append(
            SubtaskSegment(
                content=window.name if window else GAP_SUBTASK,
                start_frame=frame,
                event_id=window.event_id if window else None,
            )
        )
    return segments, dropped


def write_subtasks(root: Path, segments_by_episode: Mapping[int, Sequence[SubtaskSegment]]) -> None:
    """Add the language columns holding ``segments_by_episode`` to the dataset at ``root``."""
    # Imported here because lerobot 0.3.x, behind the v2_1 variant, has no language columns.
    from lerobot.datasets.io_utils import load_info, write_info, write_table_one_row_group_per_episode
    from lerobot.datasets.language import LANGUAGE_EVENTS, LANGUAGE_PERSISTENT, language_feature_info

    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(path)
        episode_indices = table.column("episode_index").to_pylist()
        frame_indices = table.column("frame_index").to_pylist()
        timestamps = table.column("timestamp").to_pylist()

        # Rows take the frame's stored timestamp, so a resolver asking about that
        # frame compares equal rather than off by float32 rounding.
        timestamp_of = dict(zip(zip(episode_indices, frame_indices, strict=True), timestamps, strict=True))
        rows_by_episode = {
            episode: [
                {
                    "role": "assistant",
                    "content": segment.content,
                    "style": "subtask",
                    "timestamp": timestamp_of[(episode, segment.start_frame)],
                    "camera": None,
                    "tool_calls": None,
                }
                for segment in segments_by_episode.get(episode, ())
            ]
            for episode in set(episode_indices)
        }

        # A persistent row list is repeated on every frame of its episode. The Arrow
        # type is inferred, as in lerobot's writer: the canonical one has a JSON field
        # that pa.array cannot build from Python values.
        table = table.append_column(
            LANGUAGE_PERSISTENT, pa.array([rows_by_episode[e] for e in episode_indices])
        ).append_column(LANGUAGE_EVENTS, pa.array([[] for _ in episode_indices]))

        tmp_path = path.with_suffix(path.suffix + ".tmp")
        write_table_one_row_group_per_episode(table, tmp_path)
        tmp_path.replace(path)

    info = load_info(root)
    info.features = {**info.features, **language_feature_info()}
    write_info(info, root)


def _supports_language_columns() -> bool:
    try:
        import lerobot.datasets.language  # noqa: F401
    except ImportError:
        return False
    return True


def apply_session_subtasks(
    dataset_root: Path,
    sources: Iterable[Any],
    episode_to_event: Iterable[Mapping[str, Any]],
    fps: float,
) -> dict[str, Any] | None:
    """Write the sub-tasks of every session episode into the finished dataset.

    Args:
        dataset_root: The finalized, merged LeRobot dataset.
        sources: The episode sources of the run.
        episode_to_event: The run's final manifest entries, one per written episode.
        fps: The dataset's frame rate.

    Returns:
        The manifest's record of what was written, or ``None`` if no sub-tasks
        were: the sessions have no events, or lerobot cannot store them.
    """
    windows_by_session = {s.source_id: s.subtask_windows for s in sources if s.subtask_windows}
    if not windows_by_session:
        return None
    if not _supports_language_columns():
        logger.warning(
            "%d session(s) have events, but this lerobot version has no language columns to "
            "store sub-tasks in; use roboto-to-lerobot-v3_0 to include them.",
            len(windows_by_session),
        )
        return None

    time_step_ns = fps_to_time_step_ns(fps)
    segments_by_episode: dict[int, list[SubtaskSegment]] = {}
    episodes: list[dict[str, Any]] = []
    dropped_events: list[dict[str, Any]] = []
    for entry in episode_to_event:
        session_id = entry.get("session_id")
        if session_id not in windows_by_session:
            continue
        n_frames = int(entry["n_frames"])
        start_ns = int(entry["start_time_ns"])
        expected_frames = (int(entry["end_time_ns"]) - start_ns) // time_step_ns + 1
        if n_frames != expected_frames:
            # Frames are numbered after NaN rows are dropped, so labels after a drop run early.
            logger.warning(
                "Session %s kept %d of %d frames; its sub-task boundaries may be off by the difference.",
                session_id, n_frames, expected_frames,
            )
        segments, dropped = plan_segments(
            windows_by_session[session_id],
            episode_start_ns=start_ns,
            n_frames=n_frames,
            time_step_ns=time_step_ns,
        )
        for window in dropped:
            logger.warning(
                "Event %s (%r) on session %s covers no frame at %g fps; it has no sub-task.",
                window.event_id, window.name, session_id, fps,
            )
            dropped_events.append({"session_id": session_id, **asdict(window)})
        if segments:
            segments_by_episode[int(entry["episode_index"])] = segments
            episodes.append({
                "episode_index": int(entry["episode_index"]),
                "session_id": session_id,
                "segments": [asdict(segment) for segment in segments],
            })

    if segments_by_episode:
        write_subtasks(dataset_root, segments_by_episode)
        logger.info("Wrote sub-tasks for %d episode(s).", len(segments_by_episode))
    return {"gap_subtask": GAP_SUBTASK, "episodes": episodes, "dropped_events": dropped_events}
