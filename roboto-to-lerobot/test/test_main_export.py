"""Tests that run ``main`` end to end, for a collection of sessions and one of events.

Reuses the harness of ``test_episode_cache.py`` (in-memory topics, the org's
assets in a local directory, shards in-process) and adds the two seams ``main``
itself needs: the invocation context (``_Context``) and the collection, whose
resolved sources each test builds directly. ``test_episode_sources.py`` covers
how a collection resolves into those sources.
"""

from __future__ import annotations

import json
import shutil
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import roboto
from roboto_to_lerobot.contract_utils import load_contract
from roboto_to_lerobot.episode_sources import (
    EpisodeSource,
    PreparedScope,
    ResolvedSources,
)
from roboto_to_lerobot.subtasks import SubtaskWindow
from test_episode_cache import _DS, _SECOND, _T0, _events, _Harness, main_mod

pytestmark = pytest.mark.skipif(
    not main_mod._shard_path_supported(), reason="v3.0 output needs lerobot 0.6 or later",
)


class _Context:
    """``roboto.InvocationContext`` stand-in for a local run: no file changesets, no invocation record."""

    log_level = 30
    dataset_id = "ds_invocation"
    roboto_client = None

    def __init__(self, harness: _Harness, invocation_id: str, parameters: dict[str, str]) -> None:
        self.invocation_id = invocation_id
        self.input_dir = harness.tmp_path / invocation_id / "input"
        self.output_dir = harness.tmp_path / invocation_id / "output"
        self.output_dir.mkdir(parents=True)
        self.org = SimpleNamespace(assets=harness.assets)
        self._parameters = {"collection_id": "cl_test", **parameters}

        def download(dest: Path) -> None:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(harness.contract_path, dest)

        contract = SimpleNamespace(relative_path="contract.yaml", download=download)
        self.dataset = SimpleNamespace(
            list_files=lambda include_patterns: iter([contract]),
            get_file_by_path=lambda path: contract,
            update=lambda metadata_changeset: None,
        )

    def get_parameter(self, name: str) -> str:
        return self._parameters[name]

    def get_optional_parameter(self, name: str) -> str | None:
        return self._parameters.get(name)

    @property
    def file_changeset_manager(self):
        raise ValueError("no file changesets in a local invocation")

    @property
    def invocation(self):
        raise RuntimeError("no invocation record in a local invocation")


@pytest.fixture
def export(tmp_path, monkeypatch):
    """Return ``export(resource_type, sources, **parameters)``, which runs ``main`` and returns its output folder."""
    harness = _Harness(tmp_path, monkeypatch)
    contract = load_contract(harness.contract_path)
    # ``main`` would redirect the test process's stderr and serialize the fake topics' records.
    monkeypatch.setattr(main_mod, "install_svt_stderr_filter", lambda: None)
    monkeypatch.setattr(main_mod, "topic_descriptors_from_topics", lambda topics: {})

    def run(resource_type: roboto.CollectionResourceType, sources: list[EpisodeSource], **parameters: str) -> Path:
        collection = SimpleNamespace(record=SimpleNamespace(version=1, resource_type=resource_type))
        prepared = {s.scope_key: PreparedScope(contract, harness.topics, []) for s in sources}
        monkeypatch.setattr(roboto.Collection, "from_id", lambda collection_id: collection)
        monkeypatch.setattr(main_mod, "resolve_episode_sources", lambda *_: ResolvedSources(sources, prepared))
        harness.converted.append([])
        context = _Context(harness, f"iv_{len(harness.converted)}", parameters)
        main_mod.main(context)
        return context.output_dir / context.invocation_id

    run.harness = harness
    return run


def _lineage(out: Path) -> list[tuple]:
    episodes = json.loads((out / "combined" / "meta" / "roboto_lineage.json").read_text())["episodes"]
    return [(e["session_id"], e["event_id"], e["source_dataset_id"], e["cache_hit"]) for e in episodes]


def test_session_collection_exports_through_the_cache_with_subtasks_and_session_lineage(export):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # Arrange
    (_, start, end, task), (_, later_start, later_end, later_task) = _events(0, 1)
    grasp = SubtaskWindow("ev_grasp", "grasp", start + 2 * _SECOND // 10, start + 6 * _SECOND // 10)
    sessions = [
        EpisodeSource("session", "se_0", "se_0", (_DS,), start, end, {"task": task}, (grasp,)),
        EpisodeSource("session", "se_1", "se_1", (_DS,), later_start, later_end, {"task": later_task}),
    ]

    # Act
    export(roboto.CollectionResourceType.Session, sessions, episode_cache="true")
    out = export(roboto.CollectionResourceType.Session, sessions, episode_cache="true")

    # Assert
    assert export.harness.converted == [["se_0", "se_1"], []]
    assert _lineage(out) == [("se_0", None, _DS, True), ("se_1", None, _DS, True)]
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["collection_resource_type"] == "session"
    assert manifest["source_datasets"] == [_DS]
    assert [(e["session_id"], e["event_id"]) for e in manifest["episode_to_event"]] == [("se_0", None), ("se_1", None)]
    # The cached episodes hold no sub-tasks; the export labels them after the merge.
    assert manifest["subtasks"]["episodes"] == [{
        "episode_index": 0,
        "session_id": "se_0",
        "segments": [
            {"content": "no_subtask", "start_frame": 0, "event_id": None},
            {"content": "grasp", "start_frame": 2, "event_id": "ev_grasp"},
            {"content": "no_subtask", "start_frame": 6, "event_id": None},
        ],
    }]
    root = out / "combined"
    second_session = LeRobotDataset(root.name, root=root, episode_filter=lambda ep: ep["roboto/session_id"] == "se_1")
    assert second_session.episodes == [1]
    assert "language_persistent" in second_session.meta.features


def test_subtasks_stay_on_their_frames_when_an_episode_drops_frames_at_its_start(export):
    # Arrange
    # The session starts 0.3 s before the recording has data. The joint state and action hold their
    # last value, and have none to hold on the first three reference frames, so the episode drops them.
    start = _T0 - 3 * _SECOND // 10
    grasp = SubtaskWindow("ev_grasp", "grasp", _T0 + 2 * _SECOND // 10, _T0 + 6 * _SECOND // 10)
    session = EpisodeSource("session", "se_0", "se_0", (_DS,), start, _T0 + _SECOND, {}, (grasp,))

    # Act
    out = export(roboto.CollectionResourceType.Session, [session], episode_cache="true")

    # Assert
    manifest = json.loads((out / "manifest.json").read_text())
    (episode,) = manifest["episode_to_event"]
    # 11 of the 14 reference frames from -0.3 s to 1.0 s are kept.
    assert (episode["dropped_frames"], episode["n_frames"]) == ([0, 1, 2], 11)
    # grasp covers reference frames 5 to 8, which are the episode's frames 2 to 5.
    assert manifest["subtasks"]["episodes"][0]["segments"] == [
        {"content": "no_subtask", "start_frame": 0, "event_id": None},
        {"content": "grasp", "start_frame": 2, "event_id": "ev_grasp"},
        {"content": "no_subtask", "start_frame": 6, "event_id": None},
    ]


def test_event_collection_exports_with_event_ids_and_no_session_fields(export):
    # Arrange
    events = [
        EpisodeSource("event", event_id, _DS, (_DS,), start, end, {"task": task})
        for event_id, start, end, task in _events(0, 1)
    ]

    # Act
    out = export(roboto.CollectionResourceType.Event, events)

    # Assert
    assert _lineage(out) == [(None, "ev_0", _DS, False), (None, "ev_1", _DS, False)]
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["collection_resource_type"] == "event"
    assert manifest["episodes_per_dataset"] == {_DS: 2}
    assert [e["event_id"] for e in manifest["episode_to_event"]] == ["ev_0", "ev_1"]
    assert all("session_id" not in e for e in manifest["episode_to_event"])
    assert manifest["subtasks"] is None


def _fail_to_label_subtasks(*_args, **_kwargs):
    raise RuntimeError("labelling sub-tasks failed")


@pytest.mark.parametrize(
    ("label_subtasks", "outcome"),
    [
        pytest.param(main_mod.apply_session_subtasks, nullcontext(), id="succeeds"),
        pytest.param(
            _fail_to_label_subtasks, pytest.raises(RuntimeError, match="labelling sub-tasks failed"),
            id="fails-after-writing",
        ),
    ],
)
def test_an_export_leaves_no_spool_directory_behind(export, monkeypatch, label_subtasks, outcome):
    """The episodes' spool files are scratch space: the runtime must not find them in the output to upload."""
    # Arrange
    events = [
        EpisodeSource("event", event_id, _DS, (_DS,), start, end, {"task": task})
        for event_id, start, end, task in _events(0, 1)
    ]
    # Sub-tasks are labelled after every episode is written, so a failure there leaves a written dataset.
    monkeypatch.setattr(main_mod, "apply_session_subtasks", label_subtasks)

    # Act
    with outcome:
        export(roboto.CollectionResourceType.Event, events)

    # Assert
    (out,) = export.harness.tmp_path.glob("iv_*/output/iv_*")
    assert (out / "combined").is_dir()
    assert not (out / "_spool").exists()
