"""Tests for the version-dispatched writer factory and adapter wiring.

We don't import lerobot here — the adapter modules import it lazily inside
``create``, so importing the adapter modules themselves is cheap and version-
independent. The ``create`` tests stub the lerobot module so they can run
on any supported lerobot release.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import roboto_to_lerobot.writers as writers_pkg
from roboto_to_lerobot.writers import _select_writer_class
from roboto_to_lerobot.writers.v2_1 import LeRobotWriter as V21Writer
from roboto_to_lerobot.writers.v3_0 import LeRobotWriter as V30Writer


@pytest.mark.parametrize("version", ["0.5.0", "0.5.1", "0.5.99", "0.6.0", "1.0.0"])
def test_dispatches_v3_0_for_0_5_or_newer(version: str) -> None:
    assert _select_writer_class(version) is V30Writer


@pytest.mark.parametrize("version", ["0.3.0", "0.3.3", "0.4.7"])
def test_dispatches_v2_1_for_0_3_to_0_4(version: str) -> None:
    assert _select_writer_class(version) is V21Writer


@pytest.mark.parametrize("version", ["0.2.0", "0.1.0", "0.0.1"])
def test_rejects_versions_below_0_3(version: str) -> None:
    with pytest.raises(RuntimeError, match="Unsupported lerobot version"):
        _select_writer_class(version)


@pytest.mark.parametrize(
    "version", ["0.5.0+cpu", "0.5.0a1", "0.5.0.dev1", "0.3.4rc1"]
)
def test_dispatches_for_pep440_modifiers(version: str) -> None:
    """Pre-releases and local-version installs should still dispatch."""
    cls = _select_writer_class(version)
    assert cls in (V21Writer, V30Writer)


@pytest.mark.parametrize("garbage", ["", "not-a-version", "abc.def"])
def test_rejects_unparseable_versions(garbage: str) -> None:
    with pytest.raises(RuntimeError, match=r"Could not parse|Unsupported"):
        _select_writer_class(garbage)


def test_module_exports_writer() -> None:
    """The package re-exports ``LeRobotWriter`` bound to one of the adapters."""
    assert writers_pkg.LeRobotWriter in (V21Writer, V30Writer)


# -----------------------------------------------------------------------------
# Adapter ``create`` wiring tests
#
# We monkey-patch the lerobot module that the adapter imports lazily inside
# ``create`` so we can verify the kwargs flow without depending on the
# specific lerobot release installed in the test env.
# -----------------------------------------------------------------------------


def _install_fake_lerobot(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Install a fake ``lerobot.datasets.lerobot_dataset.LeRobotDataset``.

    Returns the ``LeRobotDataset`` class mock so the test can assert on
    ``create`` calls and on instance-method dispatch.
    """
    fake_dataset_class = MagicMock(name="LeRobotDataset")
    # ``create`` must return a fresh instance mock each time it's called so
    # ``add_frame`` / ``save_episode`` / ``finalize`` calls go to the right
    # object.
    fake_dataset_class.create = MagicMock(
        return_value=MagicMock(name="LeRobotDataset_instance")
    )

    fake_module = types.ModuleType("lerobot.datasets.lerobot_dataset")
    fake_module.LeRobotDataset = fake_dataset_class

    # Construct the parent packages so ``from lerobot.datasets.lerobot_dataset
    # import LeRobotDataset`` finds our stub.
    fake_lerobot = types.ModuleType("lerobot")
    fake_datasets = types.ModuleType("lerobot.datasets")
    fake_datasets.lerobot_dataset = fake_module
    fake_lerobot.datasets = fake_datasets

    monkeypatch.setitem(sys.modules, "lerobot", fake_lerobot)
    monkeypatch.setitem(sys.modules, "lerobot.datasets", fake_datasets)
    monkeypatch.setitem(
        sys.modules, "lerobot.datasets.lerobot_dataset", fake_module
    )
    return fake_dataset_class


@pytest.fixture
def fake_lerobot(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    return _install_fake_lerobot(monkeypatch)


def _create_kwargs(tmp_path: Path) -> dict[str, Any]:
    return {
        "repo_id": "combined",
        "fps": 30,
        "features": {"observation.state": {"dtype": "float32", "shape": (3,)}},
        "root": tmp_path / "ds",
        "robot_type": "franka",
        "image_writer_threads": 4,
    }


def test_v2_1_create_passes_use_videos_and_drops_finalize(
    fake_lerobot: MagicMock, tmp_path: Path
) -> None:
    writer = V21Writer.create(**_create_kwargs(tmp_path))

    fake_lerobot.create.assert_called_once()
    call_kwargs = fake_lerobot.create.call_args.kwargs
    assert call_kwargs["use_videos"] is True
    assert call_kwargs["robot_type"] == "franka"
    assert call_kwargs["fps"] == 30
    assert call_kwargs["image_writer_threads"] == 4

    # ``task`` must reach add_frame as a separate kwarg, not inside the
    # frame dict.
    writer.add_frame({"observation.state": [1.0, 2.0, 3.0], "task": "ignored"}, task="real")
    inner = fake_lerobot.create.return_value
    inner.add_frame.assert_called_once()
    frame_arg, kwargs = inner.add_frame.call_args.args, inner.add_frame.call_args.kwargs
    assert "task" not in frame_arg[0]
    assert kwargs == {"task": "real"}

    # ``finalize`` is a no-op on 0.3.x.
    writer.finalize()
    inner.finalize.assert_not_called()


def test_v3_0_create_passes_use_videos_and_calls_finalize(
    fake_lerobot: MagicMock, tmp_path: Path
) -> None:
    writer = V30Writer.create(**_create_kwargs(tmp_path))

    fake_lerobot.create.assert_called_once()
    assert fake_lerobot.create.call_args.kwargs["use_videos"] is True

    # ``task`` must reach add_frame inside the frame dict, not as a kwarg.
    writer.add_frame({"observation.state": [1.0, 2.0, 3.0]}, task="real")
    inner = fake_lerobot.create.return_value
    inner.add_frame.assert_called_once()
    frame_arg, kwargs = inner.add_frame.call_args.args, inner.add_frame.call_args.kwargs
    assert kwargs == {}
    assert frame_arg[0]["task"] == "real"

    # ``finalize`` must be forwarded.
    writer.finalize()
    inner.finalize.assert_called_once()


def test_v3_0_kwarg_task_overrides_frame_task(
    fake_lerobot: MagicMock, tmp_path: Path
) -> None:
    """If a frame already carries ``task``, the explicit kwarg wins."""
    writer = V30Writer.create(**_create_kwargs(tmp_path))
    writer.add_frame({"task": "stale", "observation.state": [0.0]}, task="fresh")
    frame = fake_lerobot.create.return_value.add_frame.call_args.args[0]
    assert frame["task"] == "fresh"


# -----------------------------------------------------------------------------
# Writer-config plumbing
#
# The Protocol gained four encoder kwargs (batch_encoding_size,
# streaming_encoding, encoder_threads, encoder_queue_maxsize). The v3_0
# adapter forwards them and locks vcodec="libsvtav1"; the v2_1 adapter
# accepts and silently drops them so main.py can pass them uniformly.
# -----------------------------------------------------------------------------


def test_v3_0_forwards_new_encoding_kwargs(
    fake_lerobot: MagicMock, tmp_path: Path
) -> None:
    V30Writer.create(
        **_create_kwargs(tmp_path),
        batch_encoding_size=16,
        streaming_encoding=True,
        encoder_threads=None,
        encoder_queue_maxsize=30,
    )
    kwargs = fake_lerobot.create.call_args.kwargs
    assert kwargs["vcodec"] == "libsvtav1"
    assert kwargs["batch_encoding_size"] == 16
    assert kwargs["streaming_encoding"] is True
    assert kwargs["encoder_threads"] is None
    assert kwargs["encoder_queue_maxsize"] == 30


def test_v3_0_vcodec_locked_to_libsvtav1(
    fake_lerobot: MagicMock, tmp_path: Path
) -> None:
    """``vcodec`` is not a Protocol kwarg — the v3_0 adapter hard-codes it.

    Two guarantees:
      1. The forwarded kwargs always include ``vcodec="libsvtav1"``, even
         when the caller passes nothing about codec.
      2. ``LeRobotWriter.create`` itself rejects a ``vcodec`` kwarg, so
         callers can't sneak h264 in.
    """
    V30Writer.create(**_create_kwargs(tmp_path))
    assert fake_lerobot.create.call_args.kwargs["vcodec"] == "libsvtav1"

    with pytest.raises(TypeError, match="vcodec"):
        V30Writer.create(**_create_kwargs(tmp_path), vcodec="libx264")


def test_v2_1_silently_drops_new_encoding_kwargs(
    fake_lerobot: MagicMock, tmp_path: Path
) -> None:
    """The 0.3.x ``LeRobotDataset.create`` doesn't accept the new kwargs.

    main.py passes them anyway (one signature for both adapters); the v2_1
    adapter must accept-and-drop so they never reach lerobot.
    """
    V21Writer.create(
        **_create_kwargs(tmp_path),
        batch_encoding_size=16,
        streaming_encoding=True,
        encoder_threads=None,
        encoder_queue_maxsize=30,
    )
    kwargs = fake_lerobot.create.call_args.kwargs
    for forbidden in (
        "batch_encoding_size",
        "streaming_encoding",
        "encoder_threads",
        "encoder_queue_maxsize",
        "vcodec",
    ):
        assert forbidden not in kwargs, (
            f"v2_1 must not forward {forbidden!r} to lerobot 0.3.x"
        )


# -----------------------------------------------------------------------------
# discard_episode: soft-drop abort hook for the per-event try/except in
# _run_pool_drain. v3_0 delegates to lerobot 0.5.x's clear_episode_buffer.
# v2_1 has to probe for that method first because 0.3.x's surface differs.
# -----------------------------------------------------------------------------


def test_v3_0_discard_calls_clear_episode_buffer(
    fake_lerobot: MagicMock, tmp_path: Path
) -> None:
    """``discard_episode`` forwards straight to lerobot 0.5.x's own abort
    hook, passing ``delete_images=True`` so staged tempfiles are cleaned up."""
    writer = V30Writer.create(**_create_kwargs(tmp_path))
    writer.discard_episode()
    inner = fake_lerobot.create.return_value
    inner.clear_episode_buffer.assert_called_once_with(delete_images=True)


def test_v2_1_discard_with_clear_episode_buffer(
    fake_lerobot: MagicMock, tmp_path: Path
) -> None:
    """When the underlying 0.3.x ``LeRobotDataset`` exposes
    ``clear_episode_buffer`` (modern 0.3.x or a backport), use it the same
    way the v3_0 adapter does."""
    writer = V21Writer.create(**_create_kwargs(tmp_path))
    inner = fake_lerobot.create.return_value
    # MagicMock auto-creates the attribute; that's fine — it makes the
    # ``callable(fn)`` probe inside ``discard_episode`` succeed.
    writer.discard_episode()
    inner.clear_episode_buffer.assert_called_once_with(delete_images=True)


def test_v2_1_discard_falls_back_to_create_episode_buffer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When 0.3.x does not expose ``clear_episode_buffer``, fall back to
    rebuilding the buffer via ``create_episode_buffer`` so the next
    ``add_frame`` does not inherit stale state."""
    # Build a custom fake whose return-value instance is a plain ``object``
    # with only the attributes the fallback path consults — MagicMock's
    # auto-attribute creation would defeat the ``getattr(..., None)``
    # probe inside ``discard_episode``.
    fake_dataset_class = MagicMock(name="LeRobotDataset")

    class _InstanceNoClearHook:
        def __init__(self) -> None:
            self.episode_buffer: dict = {"size": 5}
            self.create_episode_buffer_called = 0

        def create_episode_buffer(self) -> dict:
            self.create_episode_buffer_called += 1
            return {"size": 0}

    instance = _InstanceNoClearHook()
    fake_dataset_class.create = MagicMock(return_value=instance)

    fake_module = types.ModuleType("lerobot.datasets.lerobot_dataset")
    fake_module.LeRobotDataset = fake_dataset_class
    fake_lerobot = types.ModuleType("lerobot")
    fake_datasets = types.ModuleType("lerobot.datasets")
    fake_datasets.lerobot_dataset = fake_module
    fake_lerobot.datasets = fake_datasets
    monkeypatch.setitem(sys.modules, "lerobot", fake_lerobot)
    monkeypatch.setitem(sys.modules, "lerobot.datasets", fake_datasets)
    monkeypatch.setitem(
        sys.modules, "lerobot.datasets.lerobot_dataset", fake_module
    )

    writer = V21Writer.create(**_create_kwargs(tmp_path))
    writer.discard_episode()
    assert instance.create_episode_buffer_called == 1
    # Buffer pointer is now the freshly-created empty one.
    assert instance.episode_buffer == {"size": 0}


def test_v2_1_discard_no_hooks_warns_but_does_not_raise(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog
) -> None:
    """If neither ``clear_episode_buffer`` nor ``create_episode_buffer`` is
    available, soft-drop must be best-effort — log a WARNING and return,
    never re-raise (re-raising here would defeat the whole feature)."""
    fake_dataset_class = MagicMock(name="LeRobotDataset")

    class _InstanceMinimal:
        pass

    instance = _InstanceMinimal()
    fake_dataset_class.create = MagicMock(return_value=instance)

    fake_module = types.ModuleType("lerobot.datasets.lerobot_dataset")
    fake_module.LeRobotDataset = fake_dataset_class
    fake_lerobot = types.ModuleType("lerobot")
    fake_datasets = types.ModuleType("lerobot.datasets")
    fake_datasets.lerobot_dataset = fake_module
    fake_lerobot.datasets = fake_datasets
    monkeypatch.setitem(sys.modules, "lerobot", fake_lerobot)
    monkeypatch.setitem(sys.modules, "lerobot.datasets", fake_datasets)
    monkeypatch.setitem(
        sys.modules, "lerobot.datasets.lerobot_dataset", fake_module
    )

    writer = V21Writer.create(**_create_kwargs(tmp_path))
    with caplog.at_level("WARNING", logger="roboto_to_lerobot"):
        writer.discard_episode()  # must not raise
    msgs = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("episode_buffer" in m for m in msgs)
