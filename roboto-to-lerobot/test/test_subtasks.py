"""Session events reach a LeRobot dataset as sub-tasks that lerobot itself can read."""
import sys

import numpy as np
import pytest

pytest.importorskip("lerobot.datasets.language", reason="needs lerobot >= 0.6")

from lerobot.datasets.language_render import active_at
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from roboto_to_lerobot.episode_sources import EpisodeSource
from roboto_to_lerobot.subtasks import (
    GAP_SUBTASK,
    SubtaskWindow,
    apply_session_subtasks,
    parse_event_patterns,
)

FPS = 10
STEP_NS = 100_000_000
EPISODE_START_NS = 1_000_000_000_000
REPO_ID = "test/subtasks"


def _ns(seconds: float) -> int:
    return EPISODE_START_NS + round(seconds * 1e9)


def _window(event_id: str, name: str, start_s: float, end_s: float, tags: tuple[str, ...] = ()) -> SubtaskWindow:
    return SubtaskWindow(event_id, name, _ns(start_s), _ns(end_s), tags)


def _source(session_id: str, *windows: SubtaskWindow) -> EpisodeSource:
    return EpisodeSource(
        source_type="session",
        source_id=session_id,
        scope_key=session_id,
        dataset_ids=("ds_a",),
        start_time=EPISODE_START_NS,
        end_time=_ns(1),
        subtask_windows=windows,
    )


def _entry(episode_index: int, session_id: str, dropped_frames: tuple[int, ...] = ()) -> dict:
    """The manifest entry of a 20-frame episode that left ``dropped_frames`` out of its reference timeline."""
    return {
        "episode_index": episode_index,
        "session_id": session_id,
        "n_frames": 20,
        "dropped_frames": list(dropped_frames),
        "start_time_ns": EPISODE_START_NS,
        "end_time_ns": EPISODE_START_NS + (20 + len(dropped_frames) - 1) * STEP_NS,
    }


@pytest.fixture
def dataset_root(tmp_path):
    """A real two-episode dataset of 20 frames each, as the writers finish it."""
    root = tmp_path / "combined"
    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        root=root,
        use_videos=False,
        features={"observation.state": {"dtype": "float32", "shape": (1,), "names": ["a"]}},
    )
    for _ in range(2):
        for _ in range(20):
            dataset.add_frame({"observation.state": np.zeros(1, dtype=np.float32), "task": "pick"})
        dataset.save_episode()
    dataset.finalize()
    return root


def _active_subtask(dataset: LeRobotDataset, frame: int) -> str | None:
    sample = dataset[frame]
    row = active_at(float(sample["timestamp"]), persistent=sample["language_persistent"], style="subtask")
    return row["content"] if row else None


def test_events_label_the_frames_they_cover(dataset_root):
    sources = [
        _source(
            "se_a",
            _window("ev_grasp", "grasp", 0.5, 1.0),
            # Overlaps grasp: active from its start, then grasp resumes.
            _window("ev_slip", "slip", 0.7, 0.8),
            # Runs past the end of the episode: clipped to it.
            _window("ev_lift", "lift", 1.5, 99.0),
        ),
        _source("se_without_events"),
    ]

    apply_session_subtasks(dataset_root, sources, [_entry(0, "se_a"), _entry(1, "se_without_events")], FPS)

    dataset = LeRobotDataset(REPO_ID, root=dataset_root)
    labels = {frame: _active_subtask(dataset, frame) for frame in (0, 4, 5, 6, 7, 8, 9, 10, 14, 15, 19)}
    assert labels == {
        0: GAP_SUBTASK,
        4: GAP_SUBTASK,
        5: "grasp",
        6: "grasp",
        7: "slip",
        8: "grasp",
        9: "grasp",
        10: GAP_SUBTASK,
        14: GAP_SUBTASK,
        15: "lift",
        19: "lift",
    }
    # The next episode starts at frame 20: its session had no events.
    assert _active_subtask(dataset, 20) is None
    assert _active_subtask(dataset, 39) is None


def test_dataset_is_untouched_when_no_session_has_events(dataset_root):
    before = {p: p.read_bytes() for p in dataset_root.rglob("*") if p.is_file()}

    summary = apply_session_subtasks(dataset_root, [_source("se_a")], [_entry(0, "se_a")], FPS)

    assert summary is None
    assert {p: p.read_bytes() for p in dataset_root.rglob("*") if p.is_file()} == before


def test_event_shorter_than_a_frame_is_reported_rather_than_silently_lost(dataset_root):
    sources = [
        _source(
            "se_a",
            _window("ev_grasp", "grasp", 0.5, 1.0),
            _window("ev_instant", "bump", 1.2, 1.2),
        )
    ]

    summary = apply_session_subtasks(dataset_root, sources, [_entry(0, "se_a")], FPS)

    assert [d["event_id"] for d in summary["dropped_events"]] == ["ev_instant"]
    dataset = LeRobotDataset(REPO_ID, root=dataset_root)
    assert _active_subtask(dataset, 12) == GAP_SUBTASK


def test_dropped_frames_do_not_move_a_subtask_off_the_frames_its_event_covers(dataset_root):
    # Arrange
    # grasp covers reference frames 5 to 9. With reference frames 0, 1 and 7 left out of the
    # episode, those are the episode's frames 3 to 6.
    sources = [_source("se_a", _window("ev_grasp", "grasp", 0.5, 1.0))]

    # Act
    apply_session_subtasks(dataset_root, sources, [_entry(0, "se_a", dropped_frames=(0, 1, 7))], FPS)

    # Assert
    dataset = LeRobotDataset(REPO_ID, root=dataset_root)
    assert [_active_subtask(dataset, frame) for frame in (2, 3, 6, 7)] == [GAP_SUBTASK, "grasp", "grasp", GAP_SUBTASK]


@pytest.mark.parametrize(
    ("subtask_events", "at_frame_7", "at_frame_14"),
    [
        ("!qa/*", "grasp", GAP_SUBTASK),
        ("grasp, lift", "grasp", GAP_SUBTASK),
        ("qa/*", GAP_SUBTASK, "qa/blur"),
        (None, "grasp", "qa/blur"),
    ],
    ids=["exclude", "include-list", "include-glob", "unset-selects-all"],
)
def test_subtask_events_chooses_by_name_which_events_become_subtasks(
    dataset_root, subtask_events, at_frame_7, at_frame_14,
):
    # Arrange
    sources = [
        _source("se_a", _window("ev_grasp", "grasp", 0.5, 1.0), _window("ev_qa", "qa/blur", 1.2, 1.6)),
    ]

    # Act
    apply_session_subtasks(
        dataset_root, sources, [_entry(0, "se_a")], FPS, event_patterns=parse_event_patterns(subtask_events),
    )

    # Assert
    dataset = LeRobotDataset(REPO_ID, root=dataset_root)
    assert (_active_subtask(dataset, 7), _active_subtask(dataset, 14)) == (at_frame_7, at_frame_14)


@pytest.mark.parametrize(
    ("subtask_events", "expected"),
    [
        ("tag:subtask", ("grasp", GAP_SUBTASK, GAP_SUBTASK, GAP_SUBTASK)),
        ("!tag:qa", ("grasp", GAP_SUBTASK, GAP_SUBTASK, "lift")),
        ("grasp*,!tag:qa", ("grasp", GAP_SUBTASK, GAP_SUBTASK, GAP_SUBTASK)),
        ("lift,tag:rev*", (GAP_SUBTASK, GAP_SUBTASK, "grasp_check", "lift")),
    ],
    ids=["include-tag", "exclude-tag", "exclude-tag-from-name-glob", "name-or-tag-glob"],
)
def test_subtask_events_chooses_by_tag_and_name_which_events_become_subtasks(dataset_root, subtask_events, expected):
    # Arrange
    sources = [
        _source(
            "se_a",
            _window("ev_grasp", "grasp", 0.5, 1.0, tags=("subtask",)),
            _window("ev_blur", "blur", 1.2, 1.4, tags=("qa",)),
            _window("ev_check", "grasp_check", 1.5, 1.7, tags=("qa", "review")),
            _window("ev_lift", "lift", 1.8, 1.9),
        ),
    ]

    # Act
    apply_session_subtasks(
        dataset_root, sources, [_entry(0, "se_a")], FPS, event_patterns=parse_event_patterns(subtask_events),
    )

    # Assert
    dataset = LeRobotDataset(REPO_ID, root=dataset_root)
    assert tuple(_active_subtask(dataset, frame) for frame in (7, 12, 15, 18)) == expected


def test_lerobot_without_language_columns_skips_subtasks_and_leaves_the_dataset_alone(dataset_root, monkeypatch):
    # lerobot 0.3.x, behind the v2_1 variant, has no such module.
    monkeypatch.setitem(sys.modules, "lerobot.datasets.language", None)
    before = {p: p.read_bytes() for p in dataset_root.rglob("*") if p.is_file()}

    summary = apply_session_subtasks(
        dataset_root, [_source("se_a", _window("ev_grasp", "grasp", 0.5, 1.0))], [_entry(0, "se_a")], FPS,
    )

    assert summary is None
    assert {p: p.read_bytes() for p in dataset_root.rglob("*") if p.is_file()} == before
