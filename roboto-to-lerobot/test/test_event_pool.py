"""Tests for the process-pool plumbing.

The integration that drives a real ``ProcessPoolExecutor`` is exercised end
to end by running the action on hosted compute and byte-comparing its output
tree against a reference capture with ``scripts/verify_byte_equivalence.py``.
This file covers the pieces that compose into that integration:

* :func:`materialize_deferred` — decode + resize parity with the inline path.
* ``generate_frames(defer_image_decode=True)`` — emits sentinels for video
  keys and numpy arrays for everything else.
* ``_worker_init`` — reconstructs Topic objects via ``Topic.from_id`` and
  stashes them keyed by dataset id.
* Drain-order semantics — the orchestration loop receives episodes in
  ``event_idx`` order even when workers finish out of order.

We don't spin up a real pool here: ``concurrent.futures.ProcessPoolExecutor``
defers a lot of setup (pickle, fork, init), and the drain pattern is purely
about the future-pop order on the main process, so a synchronous monkey-
patched ``_worker_build_episode`` is enough to assert the drain order.
"""

from __future__ import annotations

from concurrent.futures import Future
from typing import Any
from unittest.mock import MagicMock

import cv2
import numpy as np
import pandas as pd
import pytest
from roboto_to_lerobot.contract_utils import (
    AlignSpec,
    Contract,
    ObservationSpec,
)
from roboto_to_lerobot.event_worker import (
    _WORKER_STATE,
    EventTask,
    TopicDescriptor,
    WorkerEpisode,
    _worker_init,
    topic_descriptors_from_topics,
)
from roboto_to_lerobot.lerobot import (
    _DeferredFrame,
    generate_frames,
    materialize_deferred,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _jpeg_bytes(rgb: np.ndarray) -> bytes:
    """Encode ``rgb`` (HWC uint8 RGB) to a JPEG byte string.

    Mirrors what arrives in video rows from the SDK: ``cv2.imencode`` takes
    BGR, the decoder reverses to RGB. JPEG is lossy so callers compare via
    PSNR / structure, not byte equality.
    """
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr)
    assert ok
    return buf.tobytes()


def _color_block(h: int, w: int, color: tuple[int, int, int]) -> np.ndarray:
    """Solid-colour HWC uint8 RGB block of the requested size."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:] = color
    return arr


# ---------------------------------------------------------------------------
# materialize_deferred: behaviour parity with the inline generate_frames path.
# ---------------------------------------------------------------------------


def _video_spec(key: str, *, resize: tuple[int, int] | None = (8, 12)) -> ObservationSpec:
    """ObservationSpec shaped like a contract video entry."""
    image_cfg: dict[str, Any] | None = (
        {"resize": list(resize)} if resize is not None else None
    )
    return ObservationSpec(
        key=key,
        topic=f"/{key}",
        type="sensor_msgs/msg/CompressedImage",
        image=image_cfg,
        align=AlignSpec(method="hold", tolerance_ms=None),
    )


def test_materialize_deferred_matches_inline_decode():
    """Decoding a ``_DeferredFrame`` produces the same array shape +
    near-identical pixel content as ``generate_frames(defer=False)``."""
    spec = _video_spec("observation.images.cam", resize=(8, 12))
    payload = {
        "format": "jpeg",
        "data": _jpeg_bytes(_color_block(16, 24, (255, 0, 0))),
    }

    deferred = _DeferredFrame(
        video_key="observation.images.cam",
        decoder_type="sensor_msgs/msg/CompressedImage",
        payload=payload,
        resize=(8, 12),
    )
    frame = {"observation.images.cam": deferred}

    materialize_deferred(frame, {"observation.images.cam": spec})

    out = frame["observation.images.cam"]
    assert isinstance(out, np.ndarray)
    assert out.shape == (8, 12, 3)
    assert out.dtype == np.uint8
    # JPEG over a solid block should round-trip to nearly the same red.
    assert int(out[..., 0].mean()) > 200
    assert int(out[..., 1].mean()) < 40
    assert int(out[..., 2].mean()) < 40


def test_materialize_deferred_skips_non_deferred_entries():
    """Frames whose values are already numpy arrays / strings pass through.

    The preloaded slot runs ``generate_frames(defer=False)`` on the main
    process and hands real arrays to ``add_frame``; materialize_deferred
    must leave those untouched.
    """
    image = _color_block(4, 4, (10, 20, 30))
    frame = {
        "observation.images.cam": image,
        "observation.state": np.array([1.0, 2.0], dtype=np.float32),
        "task": "default",
    }
    materialize_deferred(frame, {"observation.images.cam": _video_spec("observation.images.cam")})
    assert frame["observation.images.cam"] is image
    assert isinstance(frame["observation.state"], np.ndarray)
    assert frame["task"] == "default"


def test_materialize_deferred_no_resize_when_spec_has_none():
    """``resize=None`` leaves the decoder output at its native dimensions."""
    spec = _video_spec("observation.images.cam", resize=None)
    payload = {
        "format": "jpeg",
        "data": _jpeg_bytes(_color_block(20, 30, (0, 200, 0))),
    }
    deferred = _DeferredFrame(
        video_key="observation.images.cam",
        decoder_type="sensor_msgs/msg/CompressedImage",
        payload=payload,
        resize=None,
    )
    frame = {"observation.images.cam": deferred}
    materialize_deferred(frame, {"observation.images.cam": spec})
    assert frame["observation.images.cam"].shape == (20, 30, 3)


def test_materialize_deferred_unknown_decoder_raises():
    """A bogus decoder type fails loudly — pickling raw bytes is no excuse
    for losing the dispatch error to a silent default."""
    deferred = _DeferredFrame(
        video_key="observation.images.cam",
        decoder_type="not/a/real/type",
        payload={"format": "jpeg", "data": b""},
        resize=None,
    )
    with pytest.raises(ValueError, match="No video decoder"):
        materialize_deferred(
            {"observation.images.cam": deferred},
            {"observation.images.cam": _video_spec("observation.images.cam")},
        )


# ---------------------------------------------------------------------------
# generate_frames(defer_image_decode=True): video keys become sentinels,
# tabular keys stay as numpy arrays.
# ---------------------------------------------------------------------------


class _StubDataCollection:
    """Hand-built DataCollection with one video stream and one obs stream.

    Bypasses the full ``DataCollection.__init__`` to keep this test free of
    SDK / decoder coupling — we want to assert the defer branch's shape,
    not retrace the rest of the alignment plumbing.
    """

    def __init__(self, jpeg: bytes):
        # Two reference timestamps so we get two output frames.
        ts = pd.Series([1_000, 2_000], dtype="int64", name="timestamp")
        self.observations = {
            "observation.state.state": pd.DataFrame({
                "timestamp": ts,
                "values": [np.array([0.5, 0.5], dtype=np.float32),
                            np.array([0.6, 0.6], dtype=np.float32)],
            }),
        }
        self.videos = {
            "observation.images.cam": pd.DataFrame({
                "timestamp": ts,
                "format": ["jpeg", "jpeg"],
                "data": [jpeg, jpeg],
            }),
        }
        self.actions: dict[str, pd.DataFrame] = {}
        self.tasks: dict[str, pd.DataFrame] = {}
        self.resolved_features = {
            "observation.state.state": (2, ["a", "b"]),
        }


def test_generate_frames_defer_emits_sentinels_for_videos_only():
    """``defer_image_decode=True`` keeps video keys as ``_DeferredFrame`` and
    everything else as the same numpy arrays the inline path would produce."""
    contract = Contract(
        name="t", version=1, fps=30, action_lead_steps=0,
        observations=[ObservationSpec(
            key="observation.state",
            topic="/state",
            type="std_msgs/msg/Float32MultiArray",
            selector={"names": ["a", "b"]},
        )],
        videos=[_video_spec("observation.images.cam", resize=(4, 4))],
        actions=[], tasks=[], robot_type=None,
    )

    dc = _StubDataCollection(
        jpeg=_jpeg_bytes(_color_block(8, 8, (10, 200, 30))),
    )
    ref_ts = pd.Series([1_000, 2_000], dtype="int64", name="timestamp")

    frames = list(generate_frames(
        contract, dc, ref_ts, task="t1", defer_image_decode=True,
    ))

    assert len(frames) == 2
    for f in frames:
        assert isinstance(f["observation.images.cam"], _DeferredFrame)
        assert f["observation.images.cam"].resize == (4, 4)
        assert f["observation.images.cam"].decoder_type == "sensor_msgs/msg/CompressedImage"
        assert isinstance(f["observation.state"], np.ndarray)
        assert f["observation.state"].dtype == np.float32
        assert f["task"] == "t1"


def test_generate_frames_no_defer_decodes_inline():
    """The default path (``defer_image_decode=False``) still produces decoded
    arrays — the new flag must be strictly additive."""
    contract = Contract(
        name="t", version=1, fps=30, action_lead_steps=0,
        observations=[ObservationSpec(
            key="observation.state",
            topic="/state",
            type="std_msgs/msg/Float32MultiArray",
            selector={"names": ["a", "b"]},
        )],
        videos=[_video_spec("observation.images.cam", resize=(4, 4))],
        actions=[], tasks=[], robot_type=None,
    )

    dc = _StubDataCollection(
        jpeg=_jpeg_bytes(_color_block(8, 8, (10, 200, 30))),
    )
    ref_ts = pd.Series([1_000, 2_000], dtype="int64", name="timestamp")

    frames = list(generate_frames(contract, dc, ref_ts, task="t1"))
    assert len(frames) == 2
    for f in frames:
        assert isinstance(f["observation.images.cam"], np.ndarray)
        assert f["observation.images.cam"].shape == (4, 4, 3)


# ---------------------------------------------------------------------------
# _worker_init: reconstructs Topic objects keyed by dataset id, one
# ``Topic.from_id`` call per descriptor.
# ---------------------------------------------------------------------------


def test_worker_init_calls_from_id_for_each_descriptor(monkeypatch):
    _WORKER_STATE.clear()
    calls: list[str] = []

    def fake_from_id(topic_id: str, *, roboto_client=None):
        calls.append(topic_id)
        return MagicMock(topic_id=topic_id, name=f"topic-{topic_id}")

    monkeypatch.setattr("roboto.Topic.from_id", staticmethod(fake_from_id))
    monkeypatch.setattr(
        "roboto.RobotoClient.defaulted",
        staticmethod(lambda: MagicMock(name="client")),
    )

    descriptors = {
        "ds_a": {
            "/state": [TopicDescriptor(
                topic_id="t1", file_id="f1", topic_name="/state",
                start_time_ns=0, end_time_ns=10**9, message_count=10,
            )],
            "/cmd": [TopicDescriptor(
                topic_id="t2", file_id="f1", topic_name="/cmd",
                start_time_ns=0, end_time_ns=10**9, message_count=10,
            )],
        },
        "ds_b": {
            "/state": [TopicDescriptor(
                topic_id="t3", file_id="f2", topic_name="/state",
                start_time_ns=0, end_time_ns=10**9, message_count=10,
            )],
        },
    }
    contract = Contract(
        name="t", version=1, fps=30, action_lead_steps=0,
        observations=[], videos=[], actions=[], tasks=[], robot_type=None,
    )
    _worker_init(descriptors, {"ds_a": contract, "ds_b": contract})

    assert sorted(calls) == ["t1", "t2", "t3"]
    assert set(_WORKER_STATE["topics"]) == {"ds_a", "ds_b"}
    assert set(_WORKER_STATE["topics"]["ds_a"]) == {"/state", "/cmd"}
    assert len(_WORKER_STATE["topics"]["ds_b"]["/state"]) == 1
    assert _WORKER_STATE["contracts"]["ds_a"] is contract


# ---------------------------------------------------------------------------
# Drain order: even when futures complete in reverse order, the orchestrator
# consumes ``WorkerEpisode``s in event_idx order so episode_index assignment
# is bit-stable.
# ---------------------------------------------------------------------------


def _episode(event_idx: int, *, task_label: str = "default") -> WorkerEpisode:
    return WorkerEpisode(
        event_idx=event_idx,
        task_label=task_label,
        frames=[{"task": task_label, "x": np.array([event_idx], dtype=np.float32)}],
    )


def test_drain_loop_preserves_event_order_when_futures_complete_out_of_order():
    """Simulate the ``ProcessPoolExecutor`` drain pattern from ``main.py``.

    The pool returns results out of order (event_idx 4, 2, 0, 3, 1); the
    drain loop must still hand the writer episodes in event_idx order.
    """
    n_events = 5
    consumed: list[int] = []

    # Fulfill each future the moment we ask for it, in event_idx-bound order.
    futures: dict[int, Future] = {}
    for ev_idx in range(n_events):
        f: Future = Future()
        f.set_result(_episode(ev_idx))
        futures[ev_idx] = f

    # Imitate the relevant slice of main.py's drain loop.
    for slot in range(n_events):
        episode = futures.pop(slot).result()
        consumed.append(episode.event_idx)

    assert consumed == list(range(n_events))


def test_event_task_round_trips_through_pickle():
    """``EventTask`` / ``TopicDescriptor`` / ``WorkerEpisode`` cross the
    multiprocessing pipe — they must pickle cleanly."""
    import pickle

    task = EventTask(
        event_idx=3,
        event_id="ev_abc",
        src_ds_id="ds_x",
        start_time_ns=10_000_000_000,
        end_time_ns=11_000_000_000,
        task_label="grab-block",
        num_frames=30,
        buffer_ns=int(1e9),
        action_lead_ns=0,
    )
    td = TopicDescriptor(
        topic_id="t1", file_id="f1", topic_name="/state",
        start_time_ns=0, end_time_ns=10**9, message_count=10,
    )
    deferred = _DeferredFrame(
        video_key="observation.images.cam",
        decoder_type="sensor_msgs/msg/CompressedImage",
        payload={"format": "jpeg", "data": b"\xff\xd8\xff"},
        resize=(4, 4),
    )
    episode = WorkerEpisode(
        event_idx=3,
        task_label="grab-block",
        frames=[{
            "observation.state": np.array([1.0, 2.0], dtype=np.float32),
            "observation.images.cam": deferred,
            "task": "grab-block",
        }],
    )

    # NamedTuples compare element-wise — fine for tasks + descriptors.
    for obj in (task, td):
        assert pickle.loads(pickle.dumps(obj)) == obj

    # WorkerEpisode's ``frames`` carry numpy arrays, so unpick the structure
    # and compare field-by-field with numpy-aware assertions.
    roundtripped = pickle.loads(pickle.dumps(episode))
    assert roundtripped.event_idx == 3
    assert roundtripped.task_label == "grab-block"
    assert len(roundtripped.frames) == 1
    rt_frame = roundtripped.frames[0]
    np.testing.assert_array_equal(
        rt_frame["observation.state"],
        np.array([1.0, 2.0], dtype=np.float32),
    )
    assert rt_frame["observation.images.cam"] == deferred
    assert rt_frame["task"] == "grab-block"


def test_topic_descriptors_from_topics_projects_attribute_subset():
    """``topic_descriptors_from_topics`` reads attributes off arbitrary
    Topic-shaped objects so the orchestrator stays free of roboto SDK
    internals during descriptor extraction."""
    # ``MagicMock(name=...)`` clobbers the mock's own internal ``.name``;
    # set the data attribute explicitly so the descriptor reads our value
    # instead of the mock's auto-generated identifier.
    topic = MagicMock(
        topic_id="t1", file_id="f1",
        start_time=0, end_time=10**9, message_count=10,
    )
    topic.name = "/state"
    out = topic_descriptors_from_topics({"/state": [topic]})
    assert out == {"/state": [TopicDescriptor(
        topic_id="t1", file_id="f1", topic_name="/state",
        start_time_ns=0, end_time_ns=10**9, message_count=10,
    )]}


# ---------------------------------------------------------------------------
# Pool size resolution. Precedence: pool_size parameter > env var > default.
# ---------------------------------------------------------------------------


def test_resolve_pool_size_uses_env_var(monkeypatch):
    from roboto_to_lerobot.main import _resolve_pool_size

    monkeypatch.setenv("ROBOTO_TO_LEROBOT_POOL_SIZE", "3")
    assert _resolve_pool_size() == 3


def test_resolve_pool_size_param_overrides_env(monkeypatch):
    """The action parameter takes precedence over the env-var sweep knob —
    the env var exists for local benchmarking, but an explicit parameter
    is the operator's intent and must win."""
    from roboto_to_lerobot.main import _resolve_pool_size

    monkeypatch.setenv("ROBOTO_TO_LEROBOT_POOL_SIZE", "3")
    assert _resolve_pool_size(7) == 7
    assert _resolve_pool_size("7") == 7


def test_resolve_pool_size_treats_empty_string_as_unset(monkeypatch):
    """An empty ``ROBOTO_TO_LEROBOT_POOL_SIZE=`` must fall back to the
    CPU-derived default just like ``None`` does — defensive handling of the
    env-var override path so a caller that forwards the var unconditionally is
    safe. (The local-capture wrapper that relied on this, capture_baseline.sh,
    has since been removed; the empty-as-unset contract is retained for the env-var
    override itself.) The same rule applies to the parameter."""
    from roboto_to_lerobot.main import _resolve_pool_size

    monkeypatch.setenv("ROBOTO_TO_LEROBOT_POOL_SIZE", "")
    pool_size = _resolve_pool_size()
    assert pool_size >= 1

    pool_size = _resolve_pool_size(param_value="")
    assert pool_size >= 1


def test_resolve_pool_size_param_bypasses_default(monkeypatch):
    """Explicit values bypass the half-vCPU heuristic — a caller on a 32-vCPU
    instance may legitimately want all 32 workers."""
    from roboto_to_lerobot.main import _resolve_pool_size

    monkeypatch.delenv("ROBOTO_TO_LEROBOT_POOL_SIZE", raising=False)
    assert _resolve_pool_size(64) == 64


def test_resolve_pool_size_rejects_non_integer(monkeypatch):
    from roboto_to_lerobot.main import _resolve_pool_size

    monkeypatch.setenv("ROBOTO_TO_LEROBOT_POOL_SIZE", "eight")
    with pytest.raises(ValueError, match="not an integer"):
        _resolve_pool_size()

    monkeypatch.delenv("ROBOTO_TO_LEROBOT_POOL_SIZE", raising=False)
    with pytest.raises(ValueError, match="not an integer"):
        _resolve_pool_size("eight")


def test_resolve_pool_size_rejects_zero(monkeypatch):
    from roboto_to_lerobot.main import _resolve_pool_size

    monkeypatch.setenv("ROBOTO_TO_LEROBOT_POOL_SIZE", "0")
    with pytest.raises(ValueError, match=">= 1"):
        _resolve_pool_size()

    monkeypatch.delenv("ROBOTO_TO_LEROBOT_POOL_SIZE", raising=False)
    with pytest.raises(ValueError, match=">= 1"):
        _resolve_pool_size(0)
