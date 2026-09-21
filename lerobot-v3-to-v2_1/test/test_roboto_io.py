"""Tests for the Roboto publish flow (dataset creation/upload/provenance).

The pure helpers (name derivation, provenance shape) are covered in test_main.py.
Here the SDK is mocked so the orchestration is exercised without a live API.
"""

import types
from pathlib import Path

import pytest

roboto = pytest.importorskip("roboto")
pytest.importorskip("lerobot")

from lerobot_v3_to_v2_1.roboto_io import publish_converted_dataset  # noqa: E402


class _FakeDataset:
    def __init__(self, dataset_id="ds_new", name=None):
        self.dataset_id = dataset_id
        self.name = name
        self.uploaded = None
        self.changeset = None

    def upload_directory(self, path, **kwargs):
        self.uploaded = Path(path)

    def update(self, metadata_changeset=None):
        self.changeset = metadata_changeset


def _context(dataset_id="ds_src", source_name="Src", org_id="og_1", invocation_id="iv_1"):
    source = types.SimpleNamespace(name=source_name, dataset_id=dataset_id)
    return types.SimpleNamespace(
        dataset=source,
        dataset_id=dataset_id,
        org_id=org_id,
        invocation_id=invocation_id,
    )


def test_publish_creates_new_dataset(tmp_path, monkeypatch):
    captured = {}
    fake = _FakeDataset(name="Src (LeRobot v2.1)")

    def fake_create(**kwargs):
        captured.update(kwargs)
        return fake

    monkeypatch.setattr(roboto.Dataset, "create", fake_create)

    local = tmp_path / "tree"
    local.mkdir()
    result = publish_converted_dataset(
        _context(), local, {"episodes": 3, "video": {"copied": 3, "reencoded": 0}}, "v3.0"
    )

    assert result is fake
    assert captured["name"] == "Src (LeRobot v2.1)"
    assert captured["caller_org_id"] == "og_1"
    assert fake.uploaded == local
    assert fake.changeset is not None  # provenance stamped


def test_publish_targets_existing_dataset(tmp_path, monkeypatch):
    fake = _FakeDataset(dataset_id="ds_existing")
    monkeypatch.setattr(roboto.Dataset, "from_id", lambda dataset_id: fake)

    def must_not_create(**kwargs):
        raise AssertionError("create() must not be called when targeting an existing dataset")

    monkeypatch.setattr(roboto.Dataset, "create", must_not_create)

    local = tmp_path / "tree"
    local.mkdir()
    result = publish_converted_dataset(
        _context(), local, {"episodes": 3}, "v3.0", target_dataset_id="ds_existing"
    )

    assert result is fake
    assert fake.uploaded == local
    assert fake.changeset is not None
