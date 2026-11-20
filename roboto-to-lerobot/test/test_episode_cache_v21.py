"""Tests for the episode cache on the v2_1 variant (lerobot 0.3.x, LeRobot v2.1).

Reuses the harness of ``test_episode_cache.py``: the org's assets are a
local directory, topics are in-memory, and shards run in-process. Entries
are one-episode v2.1 datasets, merged by ``merge_v21_entries``; outputs are
compared file by file, byte for byte, with a single-writer conversion of
the same events.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from roboto_to_lerobot.main import _shard_path_supported
from test_episode_cache import _SECOND, _events, _export_manifest, _Harness

pytestmark = pytest.mark.skipif(_shard_path_supported(), reason="v2.1 output needs lerobot 0.3.x")


@pytest.fixture
def harness(tmp_path, monkeypatch) -> _Harness:
    return _Harness(tmp_path, monkeypatch)


def _files(root: Path) -> dict[str, str]:
    """sha256 of every file of the dataset, by relative path.

    Leaves out the lineage file, which the reference conversion does not write.
    """
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != "roboto_lineage.json"
    }


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_warm_export_converts_nothing_and_equals_full_conversion(harness):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # Arrange
    events = _events(0, 1, 2, 3)
    harness.export(events)

    # Act
    warm = harness.export(events)

    # Assert
    assert harness.converted == [["ev_0", "ev_1", "ev_2", "ev_3"], []]
    assert _export_manifest(warm) == [(f"ev_{i}", True) for i in range(4)]
    full = harness.full_conversion(events)
    assert _files(warm) == _files(full)
    # The byte comparison leaves out the lineage file; LeRobot must still load the dataset with it present.
    merged, reference = LeRobotDataset(warm.name, root=warm), LeRobotDataset("full", root=full)
    for i in (0, len(reference) - 1):
        assert merged[i]["observation.images.cam"].equal(reference[i]["observation.images.cam"])


def test_changed_event_converts_only_that_one_and_equals_full_conversion(harness):
    # Arrange
    harness.export(_events(0, 1, 2))
    events = _events(0, 1, 2, shift_end_ns={1: _SECOND // 10})

    # Act
    out = harness.export(events)

    # Assert
    assert harness.converted[1] == ["ev_1"]
    assert _files(out) == _files(harness.full_conversion(events))


def test_reordered_subset_gets_new_episode_and_task_indices(harness):
    # Arrange
    harness.export(_events(0, 1, 3, 4))
    # ev_0 and ev_3 dropped, ev_2 added: cached ev_1 and ev_4 move from
    # episodes 1 and 3 to 0 and 2. Each entry numbers its one task 0, so the
    # merge must renumber tasks in output order.
    events = _events(1, 2, 4)

    # Act
    out = harness.export(events)

    # Assert
    assert harness.converted[1] == ["ev_2"]
    assert sorted(p.relative_to(out).as_posix() for p in (out / "data").rglob("*.parquet")) == [
        f"data/chunk-000/episode_00000{i}.parquet" for i in range(3)
    ]
    assert _jsonl(out / "meta" / "tasks.jsonl") == [
        {"task_index": 0, "task": "pick"}, {"task_index": 1, "task": "place"}, {"task_index": 2, "task": "fold"},
    ]
    assert _files(out) == _files(harness.full_conversion(events))


def test_merge_numbers_an_episodes_tasks_in_the_order_its_frames_use_them(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from roboto_to_lerobot.merge_v21 import merge_v21_entries

    # Arrange
    # The harness gives each event one task, so this test builds multi-task
    # entries itself. lerobot 0.3.x numbers an entry's tasks in a set's
    # iteration order, so the entries' own task indices are arbitrary.
    features = {"observation.state": {"dtype": "float32", "shape": (2,), "names": None}}
    episodes = [["fold", "fold"], ["wipe", "pick", "wipe", "fold"]]
    parts = []
    for i, tasks in enumerate(episodes):
        entry = LeRobotDataset.create(repo_id=f"e{i}", fps=10, features=features, root=tmp_path / f"e{i}")
        for task in tasks:
            entry.add_frame({"observation.state": np.zeros(2, dtype=np.float32)}, task=task)
        entry.save_episode()
        parts.append({"root": str(tmp_path / f"e{i}")})

    # Act
    merge_v21_entries(parts, tmp_path / "merged")

    # Assert
    merged = LeRobotDataset("merged", root=tmp_path / "merged")
    assert [merged[i]["task"] for i in range(len(merged))] == [t for ep in episodes for t in ep]
    assert _jsonl(tmp_path / "merged" / "meta" / "tasks.jsonl") == [
        {"task_index": 0, "task": "fold"}, {"task_index": 1, "task": "wipe"}, {"task_index": 2, "task": "pick"},
    ]
    assert [e["tasks"] for e in _jsonl(tmp_path / "merged" / "meta" / "episodes.jsonl")] == [
        ["fold"], ["wipe", "pick", "fold"],
    ]
