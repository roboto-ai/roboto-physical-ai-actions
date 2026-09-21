"""Tests for the duplicate-topic dedup in collect_topics_from_dataset."""
from dataclasses import dataclass, field

from roboto_to_lerobot.contract_utils import (
    ActionSpec,
    Contract,
    ObservationSpec,
    collect_topics_from_dataset,
)


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
    relative_path: str
    topics: list[FakeTopic] = field(default_factory=list)

    def get_topics(self):
        return list(self.topics)


@dataclass
class FakeDataset:
    files: list[FakeFile]

    def list_files(self):
        return list(self.files)


def _contract_with_topic(name: str) -> Contract:
    return Contract(
        name="test", version=1, fps=30, action_lead_steps=0,
        observations=[ObservationSpec(key="observation.state", topic=name, type="x")],
        actions=[ActionSpec(key="action", topic=name, type="x")],
        videos=[], tasks=[], robot_type=None,
    )


def test_chunked_topics_preserved():
    """Same topic, disjoint time ranges → all kept (chunked recording)."""
    name = "/foo"
    contract = _contract_with_topic(name)
    files = [
        FakeFile("chunk_a.mcap", [FakeTopic(name, "fl_a", "tp_a", 0, 1000, 50)]),
        FakeFile("chunk_b.mcap", [FakeTopic(name, "fl_b", "tp_b", 1000, 2000, 50)]),
        FakeFile("chunk_c.mcap", [FakeTopic(name, "fl_c", "tp_c", 2000, 3000, 50)]),
    ]
    topics, dedup = collect_topics_from_dataset(FakeDataset(files), contract)

    assert len(topics[name]) == 3
    assert {t.file_id for t in topics[name]} == {"fl_a", "fl_b", "fl_c"}
    assert dedup == []


def test_exact_duplicates_collapsed():
    """Same topic, identical (start, end, count) → keep one, drop rest."""
    name = "/foo"
    contract = _contract_with_topic(name)
    files = [
        FakeFile("e303/ep_000.mcap", [FakeTopic(name, "fl_z", "tp_z", 100, 200, 660)]),
        FakeFile("e304/ep_000.mcap", [FakeTopic(name, "fl_a", "tp_a", 100, 200, 660)]),
        FakeFile("e305/ep_305.mcap", [FakeTopic(name, "fl_m", "tp_m", 100, 200, 660)]),
    ]
    topics, dedup = collect_topics_from_dataset(FakeDataset(files), contract)

    # Winner is lowest file_id ("fl_a"); the other two are dropped.
    assert len(topics[name]) == 1
    assert topics[name][0].file_id == "fl_a"

    assert len(dedup) == 1
    record = dedup[0]
    assert record["topic_name"] == name
    assert record["signature"] == {
        "start_time_ns": 100, "end_time_ns": 200, "message_count": 660,
    }
    assert record["kept"]["file_id"] == "fl_a"
    assert {d["file_id"] for d in record["dropped"]} == {"fl_z", "fl_m"}


def test_chunks_and_duplicates_coexist():
    """Two chunks where one chunk is itself duplicated."""
    name = "/foo"
    contract = _contract_with_topic(name)
    files = [
        FakeFile("a.mcap", [FakeTopic(name, "fl_a", "tp_a", 0, 1000, 50)]),
        FakeFile("b1.mcap", [FakeTopic(name, "fl_c", "tp_c", 1000, 2000, 50)]),
        FakeFile("b2.mcap", [FakeTopic(name, "fl_b", "tp_b", 1000, 2000, 50)]),  # dup of b1
    ]
    topics, dedup = collect_topics_from_dataset(FakeDataset(files), contract)

    # First chunk + one winner of the duplicated second chunk.
    assert sorted(t.file_id for t in topics[name]) == ["fl_a", "fl_b"]
    assert len(dedup) == 1
    assert dedup[0]["kept"]["file_id"] == "fl_b"
    assert [d["file_id"] for d in dedup[0]["dropped"]] == ["fl_c"]


def test_count_differs_not_a_duplicate():
    """Same window but different message_count → not a duplicate."""
    name = "/foo"
    contract = _contract_with_topic(name)
    files = [
        FakeFile("a.mcap", [FakeTopic(name, "fl_a", "tp_a", 100, 200, 660)]),
        FakeFile("b.mcap", [FakeTopic(name, "fl_b", "tp_b", 100, 200, 661)]),
    ]
    topics, dedup = collect_topics_from_dataset(FakeDataset(files), contract)

    assert len(topics[name]) == 2
    assert dedup == []
