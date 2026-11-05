"""Session events reach a LeRobot dataset as sub-tasks that lerobot itself can read."""
import numpy as np
import pytest

pytest.importorskip("lerobot.datasets.language", reason="needs lerobot >= 0.6")

from lerobot.datasets.language_render import active_at
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from roboto_to_lerobot.episode_sources import EpisodeSource
from roboto_to_lerobot.subtasks import GAP_SUBTASK, SubtaskWindow, apply_session_subtasks

FPS = 10
STEP_NS = 100_000_000
EPISODE_START_NS = 1_000_000_000_000
REPO_ID = "test/subtasks"


def _ns(seconds: float) -> int:
    return EPISODE_START_NS + round(seconds * 1e9)


def _window(event_id: str, name: str, start_s: float, end_s: float) -> SubtaskWindow:
    return SubtaskWindow(event_id, name, _ns(start_s), _ns(end_s))


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


def _entry(episode_index: int, session_id: str, n_frames: int = 20) -> dict:
    return {
        "episode_index": episode_index,
        "session_id": session_id,
        "n_frames": n_frames,
        "start_time_ns": EPISODE_START_NS,
        "end_time_ns": EPISODE_START_NS + (n_frames - 1) * STEP_NS,
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
