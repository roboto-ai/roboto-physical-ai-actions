"""Unit tests for the dataset-classification helpers.

These intentionally avoid the heavy lerobot/ffmpeg conversion path so they run
fast and without a v3 fixture; the end-to-end downgrade is covered separately by
a round-trip test against a real (small) dataset.
"""

import json
from pathlib import Path

import pytest

from lerobot_v3_to_v2_1.lerobot_dataset import (
    V30,
    detect_codebase_version,
    find_lerobot_dataset_root,
    image_in_parquet_features,
)
from lerobot_v3_to_v2_1.roboto_io import build_provenance, derive_output_name


def _write_info(root: Path, info: dict) -> None:
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "info.json").write_text(json.dumps(info))


def test_find_root_returns_meta_parent(tmp_path):
    dataset = tmp_path / "nested" / "my_dataset"
    _write_info(dataset, {"codebase_version": V30, "features": {}})
    assert find_lerobot_dataset_root(tmp_path) == dataset


def test_find_root_raises_when_no_dataset(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_lerobot_dataset_root(tmp_path)


def test_detect_codebase_version(tmp_path):
    _write_info(tmp_path, {"codebase_version": V30, "features": {}})
    assert detect_codebase_version(tmp_path) == V30


def test_image_in_parquet_features_flags_image_dtype(tmp_path):
    _write_info(
        tmp_path,
        {
            "codebase_version": V30,
            "features": {
                "observation.images.embedded": {"dtype": "image"},
                "observation.state": {"dtype": "float32"},
                "observation.images.cam": {"dtype": "video"},
            },
        },
    )
    assert image_in_parquet_features(tmp_path) == ["observation.images.embedded"]


def test_image_in_parquet_features_empty_for_video_only(tmp_path):
    _write_info(
        tmp_path,
        {
            "codebase_version": V30,
            "features": {"observation.images.cam": {"dtype": "video"}},
        },
    )
    assert image_in_parquet_features(tmp_path) == []


def test_derive_output_name_inherits_source_name():
    assert derive_output_name("My Robot Run", "ds_abc") == "My Robot Run (LeRobot v2.1)"


def test_derive_output_name_falls_back_without_name():
    assert derive_output_name(None, "ds_abc") == "LeRobot v2.1 conversion of ds_abc"
    assert derive_output_name("   ", "ds_abc") == "LeRobot v2.1 conversion of ds_abc"


def test_build_provenance_links_back_to_source():
    meta = build_provenance(
        source_dataset_id="ds_abc",
        source_dataset_name="Run",
        source_codebase_version="v3.0",
        invocation_id="iv_1",
        lerobot_version="0.5.1",
        report={"episodes": 7, "video": {"copied": 7, "reencoded": 0}},
    )
    assert meta["source_dataset_id"] == "ds_abc"
    assert meta["codebase_version"] == "v2.1"
    assert meta["converted_by"] == "lerobot-v3-to-v2_1"
    detail = meta["invocations.iv_1"]
    assert detail["source_codebase_version"] == "v3.0"
    assert detail["target_codebase_version"] == "v2.1"
    assert detail["lerobot_version"] == "0.5.1"
    assert detail["vendored_commit"] == "2ef2370d66"
    assert detail["episodes"] == 7
