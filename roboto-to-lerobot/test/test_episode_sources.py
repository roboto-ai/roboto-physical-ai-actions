"""Tests for resolving a collection of sessions into episode sources."""
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import roboto
from roboto.exceptions import RobotoFeatureNotAvailableException
from roboto_to_lerobot import episode_sources, subtasks
from roboto_to_lerobot.contract_utils import ActionSpec, Contract, ObservationSpec
from roboto_to_lerobot.episode_sources import (
    EpisodeSource,
    annotate_manifest_entry,
    resolve_episode_sources,
)

TOPIC = "/state"


@dataclass
class FakeTopic:
    name: str
    file_id: str
    topic_id: str
    start_time: int
    end_time: int
    message_count: int


@dataclass
class FakeFile:
    file_id: str
    dataset_id: str
    relative_path: str
    topics: list[FakeTopic] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def get_topics(self):
        return list(self.topics)


@dataclass
class FakeSession:
    session_id: str
    files: list
    min_timestamp_ns: int | None = None
    max_timestamp_ns: int | None = None
    metadata: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    has_session_events: bool = True

    def list_files(self):
        return list(self.files)

    def list_events(self):
        if not self.has_session_events:
            raise RobotoFeatureNotAvailableException("no session events in this org")
        return list(self.events)


def _session_file(file_id, dataset_id="ds_a", lo=None, hi=None, offset=0):
    return SimpleNamespace(
        file_id=file_id,
        dataset_id=dataset_id,
        min_wall_clock_timestamp_ns=lo,
        max_wall_clock_timestamp_ns=hi,
        unix_epoch_offset_ns=offset,
    )


def _contract() -> Contract:
    return Contract(
        name="test", version=1, fps=10, action_lead_steps=0,
        observations=[ObservationSpec(key="observation.state", topic=TOPIC, type="x")],
        actions=[ActionSpec(key="action", topic=TOPIC, type="x")],
        videos=[], tasks=[], robot_type=None,
    )


def _collection(sessions):
    return SimpleNamespace(
        collection_id="cl_test",
        record=SimpleNamespace(
            resource_type=roboto.CollectionResourceType.Session,
            version=3,
            resources={roboto.CollectionResourceType.Session: sessions},
            missing={},
        ),
    )


@pytest.fixture
def platform(monkeypatch):
    """Stand in for ``roboto.Session`` / ``roboto.File`` lookups.

    ``resolve_episode_sources`` builds sessions from the collection's
    hydrated records and looks files up by id; both are routed to fakes
    registered on the returned namespace.
    """
    state = SimpleNamespace(files={}, legacy_events={})
    monkeypatch.setattr(
        subtasks.roboto.Event, "get_by_file",
        lambda file_id: iter(state.legacy_events.get(file_id, [])),
    )
    fake_roboto = SimpleNamespace(
        CollectionResourceType=roboto.CollectionResourceType,
        Session=lambda record: record,
        SessionRecord=SimpleNamespace(model_validate=lambda r: r),
        File=SimpleNamespace(from_id=lambda fid: state.files[fid]),
    )
    monkeypatch.setattr(episode_sources, "roboto", fake_roboto)

    def add_file(file_id, dataset_id, start, end):
        f = FakeFile(
            file_id, dataset_id, f"{file_id}.bag",
            [FakeTopic(TOPIC, file_id, f"tp_{file_id}", start, end, 10)],
        )
        state.files[file_id] = f
        return f

    state.add_file = add_file
    return state


def test_each_session_becomes_one_source_scoped_to_its_own_files(platform):
    platform.add_file("fl_1", "ds_a", 100, 200)
    platform.add_file("fl_2", "ds_a", 300, 400)
    platform.add_file("fl_3", "ds_b", 400, 500)
    later = FakeSession(
        "se_late", [_session_file("fl_2"), _session_file("fl_3", "ds_b")],
        min_timestamp_ns=300, max_timestamp_ns=500, metadata={"task": "walk"},
    )
    earlier = FakeSession(
        "se_early", [_session_file("fl_1")], min_timestamp_ns=100, max_timestamp_ns=200,
    )

    sources, prepared = resolve_episode_sources(_collection([later, earlier]), _contract())

    # Chronological, keyed by session id, windowed by the session's bounds.
    assert [s.source_id for s in sources] == ["se_early", "se_late"]
    assert [s.scope_key for s in sources] == ["se_early", "se_late"]
    assert [(s.start_time, s.end_time) for s in sources] == [(100, 200), (300, 500)]
    assert sources[1].dataset_ids == ("ds_a", "ds_b")
    assert sources[1].metadata == {"task": "walk"}
    # Topics come from the session's files only, never the rest of a dataset.
    assert [t.file_id for t in prepared["se_early"].topics[TOPIC]] == ["fl_1"]
    assert {t.file_id for t in prepared["se_late"].topics[TOPIC]} == {"fl_2", "fl_3"}


def test_session_without_bounds_falls_back_to_file_windows_and_topic_spans(platform):
    platform.add_file("fl_1", "ds_a", 100, 200)
    platform.add_file("fl_2", "ds_a", 150, 900)
    session = FakeSession(
        "se_x", [_session_file("fl_1"), _session_file("fl_2", lo=500, hi=600)],
    )

    sources, _ = resolve_episode_sources(_collection([session]), _contract())

    # fl_1 contributes its topic span; fl_2 the narrower window the session
    # holds of it rather than its whole topic span.
    assert (sources[0].start_time, sources[0].end_time) == (100, 600)


def test_session_with_re_anchored_file_is_refused(platform):
    platform.add_file("fl_1", "ds_a", 100, 200)
    session = FakeSession(
        "se_x", [_session_file("fl_1", offset=5)], min_timestamp_ns=105, max_timestamp_ns=205,
    )

    with pytest.raises(ValueError, match="unix_epoch_offset_ns"):
        resolve_episode_sources(_collection([session]), _contract())


def test_sessions_without_files_are_skipped(platform):
    platform.add_file("fl_1", "ds_a", 100, 200)
    empty = FakeSession("se_empty", [])
    full = FakeSession(
        "se_full", [_session_file("fl_1")], min_timestamp_ns=100, max_timestamp_ns=200,
    )

    sources, prepared = resolve_episode_sources(_collection([empty, full]), _contract())

    assert [s.source_id for s in sources] == ["se_full"]
    assert set(prepared) == {"se_full"}


def test_missing_contract_topic_names_the_session(platform):
    f = platform.add_file("fl_1", "ds_a", 100, 200)
    f.topics = []
    session = FakeSession(
        "se_x", [_session_file("fl_1")], min_timestamp_ns=100, max_timestamp_ns=200,
    )

    with pytest.raises(ValueError, match="Session se_x"):
        resolve_episode_sources(_collection([session]), _contract())


def test_unsupported_resource_type_is_rejected():
    collection = SimpleNamespace(
        collection_id="cl_files",
        record=SimpleNamespace(resource_type=roboto.CollectionResourceType.File),
    )
    with pytest.raises(ValueError, match=r"events .* or sessions"):
        resolve_episode_sources(collection, _contract())


def test_annotate_relabels_session_entries_and_leaves_event_entries():
    session = EpisodeSource(
        source_type="session", source_id="se_1", scope_key="se_1",
        dataset_ids=("ds_a", "ds_b"), start_time=0, end_time=1,
    )
    event = EpisodeSource(
        source_type="event", source_id="ev_1", scope_key="ds_c",
        dataset_ids=("ds_c",), start_time=0, end_time=1,
    )
    by_key = {s.scope_key: s for s in (session, event)}

    session_entry = {"episode_index": 0, "event_id": "se_1", "source_dataset_id": "se_1"}
    event_entry = {"episode_index": 1, "event_id": "ev_1", "source_dataset_id": "ds_c"}

    assert annotate_manifest_entry(session_entry, by_key) == {
        "episode_index": 0,
        "event_id": None,
        "session_id": "se_1",
        "source_dataset_id": "ds_a",
        "source_dataset_ids": ["ds_a", "ds_b"],
    }
    assert annotate_manifest_entry(event_entry, by_key) is event_entry


def _event(event_id, name, start, end):
    return SimpleNamespace(event_id=event_id, name=name, start_time=start, end_time=end)


def _subtask_names(platform, session):
    sources, _ = resolve_episode_sources(_collection([session]), _contract())
    return [w.name for w in sources[0].subtask_windows]


def test_subtasks_come_from_the_events_on_the_session(platform):
    platform.add_file("fl_1", "ds_a", 100, 200)
    platform.legacy_events["fl_1"] = [_event("ev_old", "legacy", 110, 120)]
    session = FakeSession(
        "se_a", [_session_file("fl_1")], min_timestamp_ns=100, max_timestamp_ns=200,
        events=[_event("evt_2", "lift", 150, 160), _event("evt_1", "grasp", 120, 130)],
    )

    assert _subtask_names(platform, session) == ["grasp", "lift"]


def test_org_without_session_events_uses_the_events_on_the_sessions_files(platform):
    platform.add_file("fl_1", "ds_a", 100, 200)
    platform.add_file("fl_2", "ds_a", 200, 300)
    platform.add_file("fl_other", "ds_a", 100, 300)
    shared = _event("ev_shared", "span", 190, 210)
    platform.legacy_events.update({
        "fl_1": [_event("ev_1", "grasp", 110, 120), shared],
        "fl_2": [shared],
        "fl_other": [_event("ev_other", "not_this_session", 110, 120)],
    })
    session = FakeSession(
        "se_a", [_session_file("fl_1"), _session_file("fl_2")],
        min_timestamp_ns=100, max_timestamp_ns=300, has_session_events=False,
    )

    assert _subtask_names(platform, session) == ["grasp", "span"]
