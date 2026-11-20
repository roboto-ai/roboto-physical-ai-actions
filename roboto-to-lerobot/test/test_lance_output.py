"""Tests for the ``output_format`` parameter: the Lance layout of an export.

Reuses the harness of ``test_episode_cache.py`` to build small LeRobot
datasets. Converting needs ``lerobot-lancedb``, which only the v3_0 variant
installs.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor

import pytest
import roboto_to_lerobot.main as main_mod
from test_episode_cache import _events, _Harness


@pytest.fixture
def harness(tmp_path, monkeypatch) -> _Harness:
    return _Harness(tmp_path, monkeypatch)


@pytest.mark.skipif(importlib.util.find_spec("lerobot_lancedb") is None, reason="needs lerobot-lancedb")
def test_lance_only_export_replaces_the_dataset_with_one_lerobot_reads_the_same(harness, monkeypatch):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # Arrange
    exported = harness.export(_events(0, 1, 2))
    # The harness swaps in an inline pool; ``_write_lance`` needs the real one,
    # which runs the converter in a spawned process.
    monkeypatch.setattr(main_mod, "ProcessPoolExecutor", ProcessPoolExecutor)
    plain = harness.tmp_path / "plain"
    shutil.copytree(exported, plain)
    lance = exported.parent / "lance"

    # Act
    main_mod._write_lance(exported, lance, keep_lerobot=False)

    # Assert
    assert not exported.exists()
    read = LeRobotDataset("lance", root=lance)
    reference = LeRobotDataset("plain", root=plain)
    assert read.meta.storage_format == "lance"
    assert len(read) == len(reference) == 33
    assert read[0]["observation.images.cam"].equal(reference[0]["observation.images.cam"])
    assert read[0]["observation.state"].equal(reference[0]["observation.state"])
    # The last episode is not first in its video file, so reading its last frame checks
    # that the Lance dataset decodes from the episode's offset into that file.
    last = len(reference) - 1
    assert read[last]["observation.images.cam"].equal(reference[last]["observation.images.cam"])
    assert read[last]["observation.state"].equal(reference[last]["observation.state"])
    lineage = json.loads((lance / "meta" / "roboto_lineage.json").read_text())
    assert [e["event_id"] for e in lineage["episodes"]] == ["ev_0", "ev_1", "ev_2"]
    assert list(read.meta.episodes["roboto/event_id"]) == ["ev_0", "ev_1", "ev_2"]


def test_output_format_rejects_unknown_values():
    with pytest.raises(ValueError, match="output_format must be one of"):
        main_mod._resolve_output_format("LeRobot,lance")


def test_output_format_lance_fails_where_the_converter_is_missing(monkeypatch):
    # Arrange
    # A ``None`` entry in ``sys.modules`` makes ``find_spec`` return ``None``, as for a missing package.
    monkeypatch.setitem(sys.modules, "lerobot_lancedb", None)

    # Act / Assert
    assert main_mod._resolve_output_format(None) == "lerobot"
    with pytest.raises(ValueError, match="only roboto-to-lerobot-v3_0"):
        main_mod._resolve_output_format("lance")
