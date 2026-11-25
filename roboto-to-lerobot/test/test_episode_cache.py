"""Tests for the per-episode conversion cache (``episode_cache``).

Each test drives ``_convert_with_episode_cache`` end to end with the real
writer, real SVT-AV1 encoding and the real ``aggregate_datasets`` merge, on a
small synthetic recording. Three seams are replaced:

* the org's assets are a local directory (``_DirFiles``);
* topics are in-memory DataFrames (``_FakeTopic``) installed by a fake
  ``_worker_init``, and the event pool runs inline;
* ``_spawn_shards`` runs each shard in-process, because spawned children
  would re-import the module and lose the fakes. It records which events
  each export converted.

Outputs are compared with a single-writer conversion of the same events:
equal ``info.json`` and tasks, equal data parquet, identical video packets.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import av
import cv2
import numpy as np
import pandas as pd
import pathspec
import pytest
from roboto.exceptions import (
    RobotoFeatureNotAvailableException,
    RobotoNotFoundException,
)
from roboto_to_lerobot import episode_cache, event_worker
from roboto_to_lerobot.contract_utils import (
    DataCollection,
    _compute_buffer_ns,
    fps_to_time_step_ns,
    load_contract,
)
from roboto_to_lerobot.episode_cache import (
    MARKER_NAME,
    EpisodeCache,
    recipe_hash,
)
from roboto_to_lerobot.main import _STREAMING_ENCODING, _DrainAccum, _open_episode_cache
from roboto_to_lerobot.writers import LeRobotWriter
from test_sharded import _SyncPool

main_mod = sys.modules["roboto_to_lerobot.main"]

# The v2.1 cache tests (test_episode_cache_v21.py) reuse this harness, so
# importing this module must not skip.
pytestmark = pytest.mark.skipif(
    not main_mod._shard_path_supported(), reason="v3.0 output needs lerobot 0.6 or later",
)

_DS = "ds_src"
_T0 = 1_700_000_000_000_000_000
_SECOND = 1_000_000_000

_CONTRACT = """\
name: cache_test
version: 1
robot_type: test_arm
fps: 10
observations:
  - key: observation.images.cam
    topic: /cam
    type: sensor_msgs/msg/CompressedImage
    image:
      resize: [64, 64]
    align: {method: nearest, tolerance_ms: 100}
  - key: observation.state
    topic: /joints
    type: sensor_msgs/msg/JointState
    selector:
      names: [j1, j2]
    align: {method: hold, tolerance_ms: 200}
actions:
  - key: action
    topic: /action
    type: sensor_msgs/msg/JointState
    selector:
      names: [j1, j2]
"""


class _FakeTopic:
    """``roboto.Topic`` stand-in serving a DataFrame indexed by datetime."""

    def __init__(self, name: str, df: pd.DataFrame, file_version: int = 1) -> None:
        self.name = name
        self.topic_id = "tp_" + name.strip("/")
        self.file_id = "fl_recording"
        self.schema_name = None
        self.df = df
        self.start_time = int(df.index[0].value)
        self.end_time = int(df.index[-1].value)
        self.message_count = len(df)
        self.record = SimpleNamespace(
            association=SimpleNamespace(association_version=file_version),
            schema_checksum="sha_" + name.strip("/"),
        )

    def get_data_as_df(self, *, start_time, end_time, message_paths_include=None):
        df = self.df.loc[pd.Timestamp(start_time):pd.Timestamp(end_time)]
        if message_paths_include:
            df = df[[c for c in df.columns if any(
                c == p or c.startswith(p + ".") for p in message_paths_include
            )]]
        return df.copy()


def _recording(file_version: int = 1) -> dict[str, list[_FakeTopic]]:
    """A 6 s recording: 10 Hz 64x64 JPEG camera, 50 Hz joint state and action."""
    rng = np.random.default_rng(0)

    def frame(ts: np.ndarray, **cols: Any) -> pd.DataFrame:
        return pd.DataFrame(
            {"header.stamp.sec": ts // _SECOND, "header.stamp.nanosec": ts % _SECOND, **cols},
            index=pd.to_datetime(ts),
        )

    cam_ts = _T0 + np.arange(60) * (_SECOND // 10)
    jpegs = [
        cv2.imencode(".jpg", rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))[1].tobytes()
        for _ in cam_ts
    ]
    joint_ts = _T0 + np.arange(300) * (_SECOND // 50)
    names = [["j1", "j2"]] * len(joint_ts)
    topics = {
        "/cam": frame(cam_ts, format=["jpeg"] * len(cam_ts), data=jpegs),
        "/joints": frame(joint_ts, name=names, position=list(rng.normal(size=(len(joint_ts), 2)))),
        "/action": frame(joint_ts, name=names, position=list(rng.normal(size=(len(joint_ts), 2)))),
    }
    return {n: [_FakeTopic(n, df, file_version)] for n, df in topics.items()}


class _DirFiles:
    """``roboto.FileSystem`` stand-in (the org's assets) backed by a local directory.

    Like the platform, its listing leaves out every path that starts with
    ``.``, which is every cache file; only fetching a file by path finds one.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.lookups = 0

    def list_files(self, include_patterns=None):
        spec = pathspec.PathSpec.from_lines("gitignore", include_patterns or ["*"])
        for path in sorted(self.root.rglob("*")):
            rel = path.relative_to(self.root).as_posix()
            if path.is_file() and not rel.startswith(".") and spec.match_file(rel):
                yield SimpleNamespace(relative_path=rel)

    def get_file_by_path(self, relative_path):
        self.lookups += 1
        source = self.root / relative_path
        if not source.is_file():
            raise RobotoNotFoundException(f"no file at {relative_path}")
        return SimpleNamespace(source=source)

    def upload_files(self, files, file_destination_paths, print_progress=True):
        for path in files:
            dest = self.root / file_destination_paths[path]
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)


def _ref(file) -> dict:
    """``object_ref`` stand-in: where ``_DirFiles`` keeps the file's bytes."""
    return {"source": str(file.source)}


def _copy_refs(pairs) -> None:
    """``download_refs`` stand-in; raises, as the real one does, when a stored file is gone."""
    for ref, dest in pairs:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ref["source"], dest)


class _Harness:
    """Runs exports against one org's assets and records conversions."""

    def __init__(self, tmp_path: Path, monkeypatch) -> None:
        self.tmp_path = tmp_path
        self.assets = _DirFiles(tmp_path / "assets")
        self.assets.root.mkdir()
        self.contract_path = tmp_path / "contract.yaml"
        self.contract_path.write_text(_CONTRACT)
        self.topics = _recording()
        self.converted: list[list[str]] = []
        self._runs = 0
        monkeypatch.setattr(main_mod, "ProcessPoolExecutor", _SyncPool)
        monkeypatch.setattr(main_mod, "_worker_init", self._worker_init)
        monkeypatch.setattr(main_mod, "_spawn_shards", self._spawn_inline)
        monkeypatch.setattr(episode_cache, "object_ref", _ref)
        monkeypatch.setattr(episode_cache, "download_refs", _copy_refs)

    def _worker_init(self, descriptors_by_ds, contracts_by_ds, spool_dir=None) -> None:
        event_worker._WORKER_STATE["topics"] = dict.fromkeys(contracts_by_ds, self.topics)
        event_worker._WORKER_STATE["contracts"] = contracts_by_ds
        event_worker._WORKER_STATE["spool_dir"] = spool_dir

    def _spawn_inline(self, *, chunks_drain, chunks_tasks, shard_workdir, on_episodes, **kwargs):
        self.converted[-1].extend(sorted(m["event_id"] for drain in chunks_drain for m in drain))
        episode_conn = SimpleNamespace(send=lambda report: on_episodes([report]))
        results = []
        for i, (drain, tasks) in enumerate(zip(chunks_drain, chunks_tasks, strict=True)):
            root = shard_workdir / f"shard-{i:03d}"
            result_path = shard_workdir / f"shard-{i:03d}.result.json"
            shard_workdir.mkdir(parents=True, exist_ok=True)
            main_mod._run_shard_subprocess(
                shard_idx=i, shard_root=str(root), repo_id=root.name,
                pool_tasks=tasks, drain_order=drain, result_path=str(result_path),
                episode_conn=episode_conn, **kwargs,
            )
            results.append({**json.loads(result_path.read_text()), "root": str(root)})
        return results

    def _features(self, contract, events, buffer_ns):
        # Mirrors ``main``: features come from the first event's data.
        start, end = events[0][1], events[0][2]
        first = DataCollection(
            contract=contract, topics=self.topics,
            start_time_ns=start - buffer_ns, end_time_ns=end + buffer_ns,
        )
        return contract.get_lerobot_features(resolved_features=first.resolved_features)

    def _matched(self, events, time_step_ns):
        matched, matched_events = [], []
        for idx, (event_id, start, end, task) in enumerate(events):
            matched.append((idx, _DS, task, int((end - start) / time_step_ns) + 1))
            matched_events.append(SimpleNamespace(event_id=event_id, start_time=start, end_time=end))
        return matched, matched_events

    def export(self, events: list[tuple], max_parallel: int = 2, video_overrides: dict | None = None) -> Path:
        """Export ``events`` (chronological) through the cache; return the dataset root.

        ``video_overrides`` are the video parameters' encoder settings, as
        ``main`` passes them to the writer.
        """
        video_encoding = LeRobotWriter.resolve_video_encoding(video_overrides or {})
        self._runs += 1
        self.converted.append([])
        contract = load_contract(self.contract_path)
        buffer_ns = _compute_buffer_ns(contract)
        features = self._features(contract, events, buffer_ns)
        matched, matched_events = self._matched(events, fps_to_time_step_ns(contract.fps))
        out = self.tmp_path / f"run-{self._runs}"
        recipe = recipe_hash(
            contract_path=self.contract_path, features=features, streaming_encoding=_STREAMING_ENCODING,
            video_encoding=video_encoding,
        )
        accumulator = _DrainAccum()
        main_mod._convert_with_episode_cache(
            cache=EpisodeCache(self.assets),
            recipe=recipe,
            matched=matched,
            matched_events=matched_events,
            topics_by_ds={_DS: self.topics},
            contracts_by_ds={_DS: contract},
            buffer_ns=buffer_ns,
            action_lead_ns=0,
            dataset_root=out / "combined",
            workdir=out / "_episode_cache",
            max_parallel=max_parallel,
            accumulator=accumulator,
            fps=int(contract.fps),
            features=features,
            robot_type=contract.robot_type,
            encoder_threads=2,
            video_encoding=video_encoding,
            pool_size=1,
            video_specs_by_key={v.key: v for v in contract.videos},
            descriptors_by_ds={_DS: {}},
            log_level=30,
        )
        # Mirrors ``main``, which records lineage after every conversion path.
        main_mod._write_lineage(out / "combined", {"recipe_hash": recipe}, accumulator.episode_to_event)
        return out / "combined"

    def full_conversion(self, events: list[tuple]) -> Path:
        """Single-writer conversion of ``events``, the reference output."""
        contract = load_contract(self.contract_path)
        buffer_ns = _compute_buffer_ns(contract)
        features = self._features(contract, events, buffer_ns)
        matched, matched_events = self._matched(events, fps_to_time_step_ns(contract.fps))
        (drain,), (tasks,) = main_mod._partition_shards(
            matched, matched_events, shard_count=1, buffer_ns=buffer_ns, action_lead_ns=0,
        )
        drain.sort(key=lambda d: d["event_idx"])
        tasks.sort(key=lambda t: t.event_idx)
        root = self.tmp_path / f"full-{len(list(self.tmp_path.glob('full-*')))}"
        main_mod._run_shard_subprocess(
            shard_idx=0, log_level=30, shard_root=str(root), repo_id=root.name,
            fps=int(contract.fps), features=features, robot_type=contract.robot_type,
            encoder_threads=2, pool_size=1,
            video_specs_by_key={v.key: v for v in contract.videos},
            descriptors_by_ds={_DS: {}}, contracts_by_ds={_DS: contract},
            pool_tasks=tasks, drain_order=drain, result_path=str(root) + ".result.json",
        )
        return root


def _events(*indices: int, shift_end_ns: dict[int, int] | None = None) -> list[tuple]:
    """1 s events starting every 1.2 s; each task covers two consecutive events."""
    tasks = ["pick", "place", "fold"]
    out = []
    for i in indices:
        start = _T0 + int((0.5 + 1.2 * i) * _SECOND)
        end = start + _SECOND + (shift_end_ns or {}).get(i, 0)
        out.append((f"ev_{i}", start, end, tasks[(i // 2) % 3]))
    return out


def _content(root: Path) -> dict[str, Any]:
    """What must match between two datasets: info, data rows, video packets."""
    data = pd.concat(
        [pd.read_parquet(f) for f in sorted((root / "data").rglob("*.parquet"))],
        ignore_index=True,
    )
    packets = []
    for mp4 in sorted((root / "videos").rglob("*.mp4")):
        with av.open(str(mp4)) as container:
            stream = container.streams.video[0]
            packets += [(bytes(p), p.pts) for p in container.demux(stream) if p.size]
    return {
        "info": json.loads((root / "meta" / "info.json").read_text()),
        "tasks": pd.read_parquet(root / "meta" / "tasks.parquet").to_dict(),
        "data": {c: np.stack(data[c].to_numpy()).tolist() for c in data.columns},
        "packets": packets,
    }


def _export_manifest(root: Path) -> list[tuple[str, bool]]:
    """Each episode's event and whether it came from the cache, from the dataset's lineage."""
    lineage = json.loads((root / "meta" / "roboto_lineage.json").read_text())
    return [(e["event_id"], e["cache_hit"]) for e in lineage["episodes"]]


def _layout(root: Path) -> dict[str, list]:
    """Where each episode's rows and frames are: the file columns of the episode metadata."""
    episodes = pd.concat(
        [pd.read_parquet(f) for f in sorted((root / "meta" / "episodes").rglob("*.parquet"))],
        ignore_index=True,
    )
    columns = [c for c in episodes.columns if c.startswith(("data/", "videos/"))]
    return {c: episodes[c].tolist() for c in columns}


@pytest.fixture
def harness(tmp_path, monkeypatch) -> _Harness:
    return _Harness(tmp_path, monkeypatch)


def test_second_export_of_same_events_converts_nothing_and_matches_first(harness):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # Arrange
    events = _events(0, 1, 2)

    # Act
    first = harness.export(events)
    second = harness.export(events)

    # Assert
    assert harness.converted == [["ev_0", "ev_1", "ev_2"], []]
    assert _content(second) == _content(first)
    assert _export_manifest(second) == [("ev_0", True), ("ev_1", True), ("ev_2", True)]
    # The lineage in meta/ must not stop LeRobot loading it.
    assert len(LeRobotDataset(second.name, root=second)) == 33


def test_lineage_columns_let_lerobot_select_episodes_by_source(harness):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # Arrange
    harness.export(_events(0, 2))
    out = harness.export(_events(0, 1, 2))

    # Act
    converted = LeRobotDataset(out.name, root=out, episode_filter=lambda ep: not ep["roboto/cache_hit"])
    second_event = LeRobotDataset(out.name, root=out, episode_filter=lambda ep: ep["roboto/event_id"] == "ev_1")

    # Assert
    assert converted.episodes == second_event.episodes == [1]
    assert len(second_event) == 11


def test_edited_event_or_reuploaded_source_converts_again(harness):
    # Arrange
    harness.export(_events(0, 1, 2))

    # Act
    harness.export(_events(0, 1, 2, shift_end_ns={1: _SECOND // 10}))
    harness.topics = _recording(file_version=2)
    harness.export(_events(0))

    # Assert
    assert harness.converted[1:] == [["ev_1"], ["ev_0"]]


def test_contract_change_converts_everything_but_reformatting_converts_nothing(harness):
    # Arrange
    events = _events(0, 1)
    harness.export(events)

    # Act
    harness.contract_path.write_text("# same contract, reformatted\n" + _CONTRACT.replace(
        "    selector:\n      names: [j1, j2]\n", "    selector: {names: [j1, j2]}\n",
    ))
    harness.export(events)
    harness.contract_path.write_text(_CONTRACT.replace("robot_type: test_arm", "robot_type: other_arm"))
    harness.export(events)

    # Assert
    assert harness.converted[1:] == [[], ["ev_0", "ev_1"]]


def test_video_setting_change_converts_everything_but_restating_a_default_converts_nothing(harness):
    # Arrange
    events = _events(0, 1)
    harness.export(events)

    # Act
    harness.export(events, video_overrides={"crf": 30})
    harness.export(events, video_overrides={"crf": 40})

    # Assert
    assert harness.converted[1:] == [[], ["ev_0", "ev_1"]]


def test_second_export_looks_up_only_the_markers(harness):
    # Arrange
    events = _events(0, 1, 2)
    harness.export(events)
    harness.assets.lookups = 0

    # Act
    harness.export(events)

    # Assert
    assert harness.converted[1] == []
    # The count bounds the export's latency: each lookup is one API call, so a fully cached
    # export looks up one marker per event, not every cached file.
    assert harness.assets.lookups == len(events)


def test_entry_whose_stored_bytes_moved_is_fetched_by_path(harness):
    # Arrange
    events = _events(0, 1)
    first = harness.export(events)
    # Point every marker's stored locations at files that no longer exist.
    for m in harness.assets.root.rglob(MARKER_NAME):
        marker = json.loads(m.read_text())
        marker["objects"] = {rel: {"source": str(harness.tmp_path / "gone" / rel)} for rel in marker["objects"]}
        m.write_text(json.dumps(marker))

    # Act
    out = harness.export(events)

    # Assert
    assert harness.converted[1] == []
    assert _content(out) == _content(first)


def test_entry_without_completion_marker_is_not_a_hit(harness):
    # Arrange
    events = _events(0, 1)
    harness.export(events)
    # Simulate an upload interrupted before ev_1's marker was written.
    for m in harness.assets.root.rglob(MARKER_NAME):
        if json.loads(m.read_text())["event_id"] == "ev_1":
            m.unlink()

    # Act
    out = harness.export(events)

    # Assert
    assert harness.converted[1] == ["ev_1"]
    assert _export_manifest(out) == [("ev_0", True), ("ev_1", False)]


def test_export_mixing_hits_and_misses_matches_full_conversion(harness):
    # Arrange
    harness.export(_events(0, 1, 4))
    # Drop cached ev_0 and convert ev_2 and ev_3, which fall between cached ev_1
    # and ev_4, on two shards. Each cached entry numbers its one task 0, so the
    # merge must renumber tasks in output order, as a full conversion does.
    events = _events(1, 2, 3, 4)

    # Act
    out = harness.export(events)

    # Assert
    assert harness.converted[1] == ["ev_2", "ev_3"]
    assert _export_manifest(out) == [("ev_1", True), ("ev_2", False), ("ev_3", False), ("ev_4", True)]
    full = harness.full_conversion(events)
    assert _layout(out) == _layout(full)
    assert sorted(p.relative_to(out) for p in out.rglob("*.mp4")) == sorted(
        p.relative_to(full) for p in full.rglob("*.mp4")
    )
    assert _content(out) == _content(full)


def test_export_that_fails_keeps_the_entries_it_finished(harness, monkeypatch):
    # Arrange
    events = _events(0, 1, 2)
    # One shard converts ev_0, ev_1, ev_2 in turn; finishing ev_2 fails.
    finish = main_mod._DatasetPerEpisodeWriter.finish_episode
    finished = []

    def finish_twice_then_fail(writer):
        if len(finished) == 2:
            raise RuntimeError("disk full")
        finished.append(writer)
        return finish(writer)

    monkeypatch.setattr(main_mod._DatasetPerEpisodeWriter, "finish_episode", finish_twice_then_fail)
    with pytest.raises(RuntimeError, match="disk full"):
        harness.export(events, max_parallel=1)
    monkeypatch.setattr(main_mod._DatasetPerEpisodeWriter, "finish_episode", finish)

    # Act
    out = harness.export(events)

    # Assert
    assert harness.converted[1] == ["ev_2"]
    assert _export_manifest(out) == [("ev_0", True), ("ev_1", True), ("ev_2", False)]


def _stats(root: Path) -> dict[str, dict[str, np.ndarray]]:
    """The dataset's stats.json."""
    stats = json.loads((root / "meta" / "stats.json").read_text())
    return {
        feature: {stat: np.asarray(value, dtype=float) for stat, value in by_stat.items()}
        for feature, by_stat in stats.items()
    }


def test_merged_stats_match_full_conversion_for_every_feature_and_quantile(harness):
    """Both merges, from the cache and of shards, including the episode, row and task numbering they renumber."""
    # Arrange
    harness.export(_events(0, 2))
    events = _events(0, 1, 2, 3)
    full = _stats(harness.full_conversion(events))
    shards = [harness.full_conversion(events[:2]), harness.full_conversion(events[2:])]

    # Act
    from_cache = _stats(harness.export(events))
    main_mod._aggregate_shards([{"repo_id": r.name, "root": str(r)} for r in shards], harness.tmp_path / "sharded")
    sharded = _stats(harness.tmp_path / "sharded")

    # Assert
    for merge, merged in (("cache", from_cache), ("shards", sharded)):
        assert sorted(merged) == sorted(full), merge
        for feature, by_stat in full.items():
            assert sorted(merged[feature]) == sorted(by_stat), f"{merge}: {feature}"
            for stat, value in by_stat.items():
                np.testing.assert_allclose(
                    merged[feature][stat], value, rtol=1e-9, atol=1e-12, err_msg=f"{merge}: {feature}/{stat}",
                )


def test_episodes_packed_after_others_in_a_file_decode_their_own_frames(harness, monkeypatch):
    from lerobot.datasets import aggregate
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # Arrange
    # A size limit of about two episodes' video splits the export over several files.
    monkeypatch.setattr(aggregate, "DEFAULT_VIDEO_FILE_SIZE_IN_MB", 0.05)
    events = _events(0, 1, 2, 3, 4, 5)

    # Act
    out = harness.export(events)

    # Assert
    layout = _layout(out)
    files = list(zip(layout["videos/observation.images.cam/chunk_index"],
                     layout["videos/observation.images.cam/file_index"], strict=True))
    assert 1 < len(set(files)) < len(files)
    packed, reference = LeRobotDataset(out.name, root=out), LeRobotDataset("full", root=harness.full_conversion(events))
    episodes = reference.meta.episodes
    for first, end in zip(episodes["dataset_from_index"], episodes["dataset_to_index"], strict=True):
        for i in (first, end - 1):
            assert packed[i]["observation.images.cam"].equal(reference[i]["observation.images.cam"])


def test_switch_is_ignored_with_a_warning_when_org_asset_storage_is_disabled(caplog):
    # Arrange
    class _DisabledAssets:
        def list_files(self, include_patterns=None):
            raise RobotoFeatureNotAvailableException("not available for this organization")

    context = SimpleNamespace(
        get_optional_parameter={"episode_cache": "true"}.get,
        org=SimpleNamespace(assets=_DisabledAssets()),
    )

    # Act
    with caplog.at_level("WARNING", logger="roboto_to_lerobot"):
        cache = _open_episode_cache(context)

    # Assert
    # None selects the same conversion path as leaving the switch unset.
    assert cache is None
    assert [r.getMessage() for r in caplog.records] == [
        "episode_cache ignored: org asset storage is not enabled for this org. Converting every event."
    ]
