"""End-to-end and flow tests for the v3.0 -> v2.1 downgrade.

The heavy tests synthesise a tiny real LeRobot **v3.0** dataset with lerobot 0.5.x
(which creates v3.0 by default), run the action's downgrade, and assert the output
is a well-formed **v2.1** tree whose per-episode parquet/video content matches the
source. They ``importorskip`` the heavy deps (``lerobot``/``av``/ffmpeg) and skip
cleanly where those are absent. ``main()``'s branching (passthrough / rejections)
is exercised with a stub context and fake metadata trees.
"""

import json
import logging
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lerobot")
av = pytest.importorskip("av")
import pyarrow.parquet as pq  # noqa: E402  (after importorskip on purpose)

from lerobot_v3_to_v2_1.downgrade import downgrade_v30_to_v21  # noqa: E402
from lerobot_v3_to_v2_1.main import main  # noqa: E402

CAM_KEY = "observation.images.cam"
CAM2_KEY = "observation.images.cam2"


def _make_v3_dataset(root: Path, *, episode_lengths=(3, 5), cameras=(CAM_KEY,)) -> None:
    """Create a tiny real v3.0 LeRobot dataset at ``root``."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "observation.state": {"dtype": "float32", "shape": (2,), "names": ["x", "y"]},
        "action": {"dtype": "float32", "shape": (2,), "names": ["dx", "dy"]},
    }
    for cam in cameras:
        features[cam] = {
            "dtype": "video",
            "shape": (16, 16, 3),
            "names": ["height", "width", "channel"],
        }

    dataset = LeRobotDataset.create(
        repo_id=root.name,
        fps=10,
        features=features,
        root=root,
        robot_type="test",
        use_videos=True,
        vcodec="h264",  # widely available; the downgrade stream-copies regardless
    )
    rng = np.random.default_rng(0)
    for episode_index, length in enumerate(episode_lengths):
        for frame_index in range(length):
            frame = {
                "observation.state": np.array([episode_index, frame_index], np.float32),
                "action": np.array([frame_index, episode_index], np.float32),
                "task": f"task {episode_index}",
            }
            for cam in cameras:
                frame[cam] = (rng.random((16, 16, 3)) * 255).astype(np.uint8)
            dataset.add_frame(frame)
        dataset.save_episode()
    dataset.finalize()


def _count_decoded_frames(path: Path) -> int:
    with av.open(str(path)) as container:
        return sum(1 for _ in container.decode(container.streams.video[0]))


def _decode_frames(path: Path):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]


def _stub_context(input_dir: Path, output_dir: Path, *, is_dry_run: bool = True):
    """A minimal stand-in for roboto.InvocationContext (the attrs main() touches).

    Defaults to dry-run so these branch tests don't reach the live SDK publish
    path (covered separately, with the SDK mocked, in test_roboto_io.py).
    """
    return types.SimpleNamespace(
        input_dir=Path(input_dir),
        output_dir=Path(output_dir),
        log_level=logging.INFO,
        is_dry_run=is_dry_run,
        dataset=types.SimpleNamespace(name="src", dataset_id="ds_src"),
        dataset_id="ds_src",
        org_id="og_1",
        invocation_id="iv_1",
        get_optional_parameter=lambda name: None,
    )


def _write_v3_meta(dataset_root: Path, features: dict, codebase_version: str = "v3.0") -> None:
    meta = dataset_root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "info.json").write_text(
        json.dumps({"codebase_version": codebase_version, "features": features})
    )


@pytest.fixture
def ffmpeg_on_path(tmp_path_factory):
    """Guarantee a resolvable ``ffmpeg`` binary for the downgrade's video splitter.

    The downgrade shells out to a bare ``ffmpeg``; if the host has none, fall back
    to the binary bundled with imageio-ffmpeg by symlinking it onto PATH.
    """
    import os
    import shutil

    if shutil.which("ffmpeg"):
        yield
        return
    imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
    bindir = tmp_path_factory.mktemp("ffmpeg_bin")
    (bindir / "ffmpeg").symlink_to(imageio_ffmpeg.get_ffmpeg_exe())
    previous = os.environ["PATH"]
    os.environ["PATH"] = f"{bindir}{os.pathsep}{previous}"
    try:
        yield
    finally:
        os.environ["PATH"] = previous


# --------------------------------------------------------------------------- #
# Data + metadata
# --------------------------------------------------------------------------- #
def test_downgrade_data_and_metadata(tmp_path):
    """The non-video core: parquet de-aggregation + v2.1 metadata reconstruction."""
    episode_lengths = (3, 5)
    source = tmp_path / "src" / "synth"
    _make_v3_dataset(source, episode_lengths=episode_lengths, cameras=())

    dest = tmp_path / "out" / "synth"
    downgrade_v30_to_v21(source, dest)

    # Source is left untouched (non-destructive).
    assert json.loads((source / "meta/info.json").read_text())["codebase_version"] == "v3.0"

    info = json.loads((dest / "meta/info.json").read_text())
    assert info["codebase_version"] == "v2.1"
    assert info["data_path"] == "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    assert "data_files_size_in_mb" not in info
    assert info["total_chunks"] == 1

    for episode_index, length in enumerate(episode_lengths):
        parquet = dest / f"data/chunk-000/episode_{episode_index:06d}.parquet"
        assert parquet.exists(), f"missing {parquet}"
        assert pq.read_table(parquet).num_rows == length

    episodes = [json.loads(line) for line in (dest / "meta/episodes.jsonl").read_text().splitlines()]
    assert len(episodes) == len(episode_lengths)
    assert (dest / "meta/tasks.jsonl").exists()

    stats = [json.loads(line) for line in (dest / "meta/episodes_stats.jsonl").read_text().splitlines()]
    state_stats = stats[0]["stats"]["observation.state"]
    assert set(state_stats) <= {"mean", "std", "min", "max", "count"}


# --------------------------------------------------------------------------- #
# Video splitting
# --------------------------------------------------------------------------- #
def test_downgrade_videos_are_frame_exact(tmp_path, ffmpeg_on_path):
    """Each per-episode mp4 must decode to exactly the episode's frames (lossless)."""
    episode_lengths = (3, 5)
    source = tmp_path / "src" / "synth"
    _make_v3_dataset(source, episode_lengths=episode_lengths, cameras=(CAM_KEY,))

    dest = tmp_path / "out" / "synth"
    report = downgrade_v30_to_v21(source, dest)

    # Both episode boundaries are keyframes, so every segment is lossless-copied.
    assert report["video"] == {"copied": len(episode_lengths), "reencoded": 0}

    info = json.loads((dest / "meta/info.json").read_text())
    assert info["video_path"] == "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
    assert info["total_videos"] == len(episode_lengths)  # one camera

    source_frames = _decode_frames(source / f"videos/{CAM_KEY}/chunk-000/file-000.mp4")
    start = 0
    for episode_index, length in enumerate(episode_lengths):
        video = dest / f"videos/chunk-000/{CAM_KEY}/episode_{episode_index:06d}.mp4"
        assert video.exists(), f"missing {video}"
        out_frames = _decode_frames(video)
        assert len(out_frames) == length
        # Lossless stream-copy => decoded frames are bit-exact to the source slice,
        # and aligned to the right episode (not snapped to a neighbouring keyframe).
        np.testing.assert_array_equal(out_frames[0], source_frames[start])
        np.testing.assert_array_equal(out_frames[-1], source_frames[start + length - 1])
        start += length


def test_downgrade_videos_reencode_fallback(tmp_path, ffmpeg_on_path, monkeypatch):
    """Force the non-keyframe path: every segment is re-encoded but still frame-exact."""
    import lerobot_v3_to_v2_1.video as video_mod

    # Pretend no boundary is keyframe-aligned, so the re-encode fallback handles all.
    monkeypatch.setattr(video_mod, "_matching_keyframe", lambda *args, **kwargs: None)

    episode_lengths = (3, 5)
    source = tmp_path / "src" / "synth"
    _make_v3_dataset(source, episode_lengths=episode_lengths, cameras=(CAM_KEY,))
    dest = tmp_path / "out" / "synth"
    report = downgrade_v30_to_v21(source, dest)

    assert report["video"] == {"copied": 0, "reencoded": len(episode_lengths)}
    for episode_index, length in enumerate(episode_lengths):
        video = dest / f"videos/chunk-000/{CAM_KEY}/episode_{episode_index:06d}.mp4"
        assert _count_decoded_frames(video) == length


def test_downgrade_multi_camera(tmp_path, ffmpeg_on_path):
    """Two cameras across three episodes: every per-camera per-episode mp4 is exact.

    (1-frame episodes are not covered: lerobot 0.5.x cannot mux a single-frame
    video when building the v3 fixture, so such a dataset is not a real input.)
    """
    episode_lengths = (2, 3, 4)
    cameras = (CAM_KEY, CAM2_KEY)
    source = tmp_path / "src" / "synth"
    _make_v3_dataset(source, episode_lengths=episode_lengths, cameras=cameras)

    dest = tmp_path / "out" / "synth"
    report = downgrade_v30_to_v21(source, dest)

    total_videos = len(episode_lengths) * len(cameras)
    assert report["video"]["copied"] + report["video"]["reencoded"] == total_videos
    assert json.loads((dest / "meta/info.json").read_text())["total_videos"] == total_videos

    for cam in cameras:
        for episode_index, length in enumerate(episode_lengths):
            video = dest / f"videos/chunk-000/{cam}/episode_{episode_index:06d}.mp4"
            assert video.exists(), f"missing {video}"
            assert _count_decoded_frames(video) == length


# --------------------------------------------------------------------------- #
# main() branching
# --------------------------------------------------------------------------- #
def test_main_passes_through_v21(tmp_path):
    source = tmp_path / "in" / "ds"
    _write_v3_meta(source, features={}, codebase_version="v2.1")
    (source / "data").mkdir()
    (source / "data" / "episode.parquet").write_bytes(b"payload")

    output = tmp_path / "out"
    output.mkdir()
    main(_stub_context(source.parent, output))

    assert (output / "ds" / "meta" / "info.json").exists()
    assert (output / "ds" / "data" / "episode.parquet").read_bytes() == b"payload"


def test_main_rejects_unknown_version(tmp_path):
    source = tmp_path / "in" / "ds"
    _write_v3_meta(source, features={}, codebase_version="v1.6")
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(ValueError, match="codebase_version"):
        main(_stub_context(source.parent, output))


def test_main_rejects_image_in_parquet(tmp_path):
    source = tmp_path / "in" / "ds"
    _write_v3_meta(source, features={CAM_KEY: {"dtype": "image"}}, codebase_version="v3.0")
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(ValueError, match="image"):
        main(_stub_context(source.parent, output))
