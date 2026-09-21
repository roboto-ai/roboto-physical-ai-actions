"""Smoke tests for :class:`runtime.live_adapter.LiveAdapter`.

The adapter is the contract→tensor seam the generated ROS node sits on
top of. These tests build tiny in-memory contracts (no YAML round-trip,
no rclpy) and drive the observation path with synthetic messages so the
test suite stays import-cheap and ROS-free.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from roboto_to_lerobot.contract_utils import (
    ActionSpec,
    AlignSpec,
    ObservationSpec,
    TransformSpec,
)
from roboto_to_lerobot.contract_utils import (
    Contract as InnerContract,
)
from roboto_to_lerobot.runtime.contract_io import Contract as RuntimeContract
from roboto_to_lerobot.runtime.live_adapter import LiveAdapter


def _make_contract(
    *,
    fps: float = 30.0,
    observations: list[ObservationSpec] | None = None,
    videos: list[ObservationSpec] | None = None,
    actions: list[ActionSpec] | None = None,
) -> RuntimeContract:
    """Build the live-runtime ``Contract`` facade without YAML I/O.

    The adapter only reads ``fps`` and the spec lists; ``source_path`` is
    surfaced for ``sha256()`` which these tests don't exercise.
    """
    inner = InnerContract(
        name="test",
        version=1,
        fps=fps,
        observations=observations or [],
        videos=videos or [],
        actions=actions or [],
        tasks=[],
    )
    return RuntimeContract(inner=inner, source_path=Path("/tmp/fake-contract.yaml"))


def _multiarray_msg(values: list[float]) -> SimpleNamespace:
    """Synthetic std_msgs/Float64MultiArray — duck-typed for the decoder."""
    return SimpleNamespace(data=values)


# ----------------------------------------------------------------------------
# Observation path — single stream
# ----------------------------------------------------------------------------


def test_single_stream_sample_returns_decoded_array():
    spec = ObservationSpec(
        key="observation.state",
        topic="/robot/state",
        type="std_msgs/msg/Float64MultiArray",
        align=AlignSpec(method="hold", tolerance_ms=100),
    )
    adapter = LiveAdapter(_make_contract(observations=[spec]))

    adapter.on_message("/robot/state", _multiarray_msg([0.1, 0.2, 0.3]), ts_ns=1_000_000_000)
    sample = adapter.sample(now_ns=1_010_000_000)

    assert sample is not None
    assert set(sample.keys()) == {"observation.state"}
    np.testing.assert_allclose(sample["observation.state"], [0.1, 0.2, 0.3])
    # The converter assembles observation vectors as float32; the live runtime
    # must match so the policy sees the dtype it trained on.
    assert sample["observation.state"].dtype == np.float32


def test_sample_returns_none_when_buffer_stale():
    spec = ObservationSpec(
        key="observation.state",
        topic="/robot/state",
        type="std_msgs/msg/Float64MultiArray",
        align=AlignSpec(method="hold", tolerance_ms=20),
    )
    adapter = LiveAdapter(_make_contract(observations=[spec]))

    adapter.on_message("/robot/state", _multiarray_msg([1.0]), ts_ns=1_000_000_000)
    # 100 ms past the push, tolerance is 20 ms → stale.
    assert adapter.sample(now_ns=1_100_000_000) is None


def test_sample_returns_none_before_any_message():
    spec = ObservationSpec(
        key="observation.state",
        topic="/robot/state",
        type="std_msgs/msg/Float64MultiArray",
        align=AlignSpec(method="hold", tolerance_ms=100),
    )
    adapter = LiveAdapter(_make_contract(observations=[spec]))
    assert adapter.sample(now_ns=0) is None


# ----------------------------------------------------------------------------
# Observation path — multi-spec same key (concat in declaration order)
# ----------------------------------------------------------------------------


def test_multi_topic_same_key_concatenates_in_declaration_order():
    # Two observation specs sharing base key "observation.state" concat in
    # declaration order. This pins the adapter's own concat ordering, not a
    # converter comparison: these specs carry no selector.names, so the
    # converter's static get_lerobot_features would emit no feature for them
    # (real conversions get the widths from runtime-resolved field counts).
    spec_a = ObservationSpec(
        key="observation.state",
        topic="/arm_a/state",
        type="std_msgs/msg/Float64MultiArray",
        align=AlignSpec(method="hold", tolerance_ms=100),
    )
    spec_b = ObservationSpec(
        key="observation.state",
        topic="/arm_b/state",
        type="std_msgs/msg/Float64MultiArray",
        align=AlignSpec(method="hold", tolerance_ms=100),
    )
    adapter = LiveAdapter(_make_contract(observations=[spec_a, spec_b]))

    adapter.on_message("/arm_a/state", _multiarray_msg([1.0, 2.0]), ts_ns=1_000_000_000)
    adapter.on_message("/arm_b/state", _multiarray_msg([3.0, 4.0]), ts_ns=1_000_000_000)
    sample = adapter.sample(now_ns=1_010_000_000)

    assert sample is not None
    np.testing.assert_allclose(
        sample["observation.state"], [1.0, 2.0, 3.0, 4.0]
    )


def test_sample_returns_none_when_one_member_of_concat_is_stale():
    # Only one of two co-keyed streams has fresh data — the whole sample
    # must return None (partial observations are unsafe).
    spec_a = ObservationSpec(
        key="observation.state",
        topic="/arm_a/state",
        type="std_msgs/msg/Float64MultiArray",
        align=AlignSpec(method="hold", tolerance_ms=20),
    )
    spec_b = ObservationSpec(
        key="observation.state",
        topic="/arm_b/state",
        type="std_msgs/msg/Float64MultiArray",
        align=AlignSpec(method="hold", tolerance_ms=20),
    )
    adapter = LiveAdapter(_make_contract(observations=[spec_a, spec_b]))

    adapter.on_message("/arm_a/state", _multiarray_msg([1.0]), ts_ns=1_000_000_000)
    # arm_b never publishes — sample must refuse rather than concat with stale.
    assert adapter.sample(now_ns=1_010_000_000) is None


# ----------------------------------------------------------------------------
# Routing
# ----------------------------------------------------------------------------


def test_init_refuses_observation_with_transforms():
    # Symmetric with the action-transforms refusal: an observation that
    # declares a transform the live runtime can't apply must fail at boot.
    spec = ObservationSpec(
        key="observation.state",
        topic="/robot/state",
        type="std_msgs/msg/Float64MultiArray",
        align=AlignSpec(method="hold", tolerance_ms=100),
        transforms=[TransformSpec(type="butterworth_lowpass", params={"cutoff_hz": 5.0})],
    )
    with pytest.raises(ValueError) as exc:
        LiveAdapter(_make_contract(observations=[spec]))
    assert "transform" in str(exc.value).lower()


def test_on_message_rejects_unknown_topic():
    spec = ObservationSpec(
        key="observation.state",
        topic="/robot/state",
        type="std_msgs/msg/Float64MultiArray",
        align=AlignSpec(method="hold", tolerance_ms=100),
    )
    adapter = LiveAdapter(_make_contract(observations=[spec]))

    with pytest.raises(ValueError) as exc:
        adapter.on_message("/unexpected", _multiarray_msg([0.0]), ts_ns=0)
    assert "/unexpected" in str(exc.value)


def test_topics_property_lists_all_subscribed_topics():
    spec_a = ObservationSpec(
        key="observation.state",
        topic="/arm_a/state",
        type="std_msgs/msg/Float64MultiArray",
        align=AlignSpec(method="hold", tolerance_ms=100),
    )
    spec_b = ObservationSpec(
        key="observation.other",
        topic="/arm_b/other",
        type="std_msgs/msg/Float64MultiArray",
        align=AlignSpec(method="hold", tolerance_ms=100),
    )
    adapter = LiveAdapter(_make_contract(observations=[spec_a, spec_b]))
    assert set(adapter.topics) == {"/arm_a/state", "/arm_b/other"}


# ----------------------------------------------------------------------------
# Tolerance auto-bound warning
# ----------------------------------------------------------------------------


def test_auto_bound_tolerance_logs_warning_at_init(caplog):
    """tolerance_ms=None (converter's 'unlimited') gets capped and warned about."""
    spec = ObservationSpec(
        key="observation.state",
        topic="/robot/state",
        type="std_msgs/msg/Float64MultiArray",
        align=AlignSpec(method="hold", tolerance_ms=None),
    )
    with caplog.at_level("WARNING"):
        adapter = LiveAdapter(_make_contract(observations=[spec]))

    assert any(
        "auto-bounded" in record.message and "/robot/state" in record.message
        for record in caplog.records
    ), f"expected auto-bound warning; got {[r.message for r in caplog.records]}"

    # And the resulting buffer still functions — auto-bound for fps=30 is ~66 ms.
    adapter.on_message("/robot/state", _multiarray_msg([1.0]), ts_ns=1_000_000_000)
    assert adapter.sample(now_ns=1_010_000_000) is not None


# ----------------------------------------------------------------------------
# butterworth_lowpass_causal — the one transform allowed to run live
# ----------------------------------------------------------------------------


def _causal_transform(*, stage: str, cutoff_hz: float = 5.0, order: int = 2, fs_hz=None):
    params = {"cutoff_hz": cutoff_hz, "order": order}
    if fs_hz is not None:
        params["fs_hz"] = fs_hz
    return TransformSpec(type="butterworth_lowpass_causal", params=params, stage=stage)


def test_pre_stage_causal_transform_filters_on_message_matching_offline_batch():
    """Feeding samples through on_message (which applies the pre-filter
    before buffer.push) must match the converter's offline batch call over
    the same sequence, at the same fs_hz."""
    from roboto_to_lerobot.transforms import TRANSFORMS

    fs_hz = 100.0
    spec = ObservationSpec(
        key="observation.state",
        topic="/robot/state",
        type="std_msgs/msg/Float64MultiArray",
        align=AlignSpec(method="hold", tolerance_ms=1000),
        transforms=[_causal_transform(stage="pre", cutoff_hz=5.0, order=2, fs_hz=fs_hz)],
    )
    adapter = LiveAdapter(_make_contract(fps=30.0, observations=[spec]))

    n = 50
    t = np.arange(n) / fs_hz
    raw = np.sin(2 * np.pi * 3.0 * t).astype(np.float64)
    data = raw[:, None]  # (T, 1) — single channel to match the multiarray of len 1

    expected = TRANSFORMS["butterworth_lowpass_causal"](
        data, {"cutoff_hz": 5.0, "order": 2}, fs_hz,
    )

    ts_ns = 1_000_000_000
    step_ns = int(1e9 / fs_hz)
    online_out = []
    for i in range(n):
        adapter.on_message("/robot/state", _multiarray_msg([float(raw[i])]), ts_ns=ts_ns)
        online_out.append(adapter.sample(now_ns=ts_ns)["observation.state"][0])
        ts_ns += step_ns

    np.testing.assert_allclose(online_out, expected[:, 0], atol=1e-5)


def test_post_stage_causal_transform_filters_at_sample_time():
    """Post-stage filtering happens in sample(), designed with contract.fps."""
    from roboto_to_lerobot.transforms import TRANSFORMS

    fps = 50.0
    spec = ObservationSpec(
        key="observation.state",
        topic="/robot/state",
        type="std_msgs/msg/Float64MultiArray",
        align=AlignSpec(method="hold", tolerance_ms=1000),
        transforms=[_causal_transform(stage="post", cutoff_hz=5.0, order=2)],
    )
    adapter = LiveAdapter(_make_contract(fps=fps, observations=[spec]))

    n = 40
    t = np.arange(n) / fps
    raw = np.sin(2 * np.pi * 4.0 * t).astype(np.float64)
    data = raw[:, None]

    expected = TRANSFORMS["butterworth_lowpass_causal"](
        data, {"cutoff_hz": 5.0, "order": 2}, fps,
    )

    ts_ns = 1_000_000_000
    step_ns = int(1e9 / fps)
    online_out = []
    for i in range(n):
        adapter.on_message("/robot/state", _multiarray_msg([float(raw[i])]), ts_ns=ts_ns)
        online_out.append(adapter.sample(now_ns=ts_ns)["observation.state"][0])
        ts_ns += step_ns

    np.testing.assert_allclose(online_out, expected[:, 0], atol=1e-5)


def test_post_filter_state_does_not_advance_on_stale_tick():
    """A None (stale) sample() call must not consume a post-filter step —
    the two-pass structure in sample() exists to guarantee this."""
    fps = 20.0
    fresh_spec = ObservationSpec(
        key="observation.fresh",
        topic="/robot/fresh",
        type="std_msgs/msg/Float64MultiArray",
        align=AlignSpec(method="hold", tolerance_ms=1000),
        transforms=[_causal_transform(stage="post", cutoff_hz=3.0, order=2)],
    )
    # Never published to: forces sample() to return None every tick.
    stale_spec = ObservationSpec(
        key="observation.stale",
        topic="/robot/stale",
        type="std_msgs/msg/Float64MultiArray",
        align=AlignSpec(method="hold", tolerance_ms=20),
    )
    adapter = LiveAdapter(
        _make_contract(fps=fps, observations=[fresh_spec, stale_spec])
    )

    ts_ns = 1_000_000_000
    step_ns = int(1e9 / fps)
    adapter.on_message("/robot/fresh", _multiarray_msg([1.0]), ts_ns=ts_ns)
    # Every tick after this returns None (observation.stale never publishes),
    # so the post-filter on observation.fresh must never advance either —
    # its internal CausalLowpass.push() should be called exactly zero times.
    for _ in range(5):
        ts_ns += step_ns
        assert adapter.sample(now_ns=ts_ns) is None

    # The filter for observation.fresh was constructed but never pushed —
    # confirm no state was seeded (push() seeds _zi lazily on first call).
    (only_key,) = adapter._post_filters
    (only_filter,) = adapter._post_filters[only_key]
    assert only_filter._zi is None


def test_refuses_pre_stage_causal_transform_without_fs_hz():
    spec = ObservationSpec(
        key="observation.state",
        topic="/robot/state",
        type="std_msgs/msg/Float64MultiArray",
        align=AlignSpec(method="hold", tolerance_ms=100),
        transforms=[_causal_transform(stage="pre")],  # no fs_hz
    )
    with pytest.raises(ValueError) as exc:
        LiveAdapter(_make_contract(observations=[spec]))
    msg = str(exc.value).lower()
    assert "fs_hz" in msg
    assert "pre" in msg


def test_refuses_post_stage_causal_transform_with_contradicting_fs_hz():
    spec = ObservationSpec(
        key="observation.state",
        topic="/robot/state",
        type="std_msgs/msg/Float64MultiArray",
        align=AlignSpec(method="hold", tolerance_ms=100),
        transforms=[_causal_transform(stage="post", fs_hz=99.0)],
    )
    with pytest.raises(ValueError) as exc:
        LiveAdapter(_make_contract(fps=30.0, observations=[spec]))
    msg = str(exc.value).lower()
    assert "fs_hz" in msg
    assert "contradicts" in msg


def test_post_stage_causal_transform_matching_fs_hz_is_accepted():
    """fs_hz equal to contract.fps is not a contradiction."""
    spec = ObservationSpec(
        key="observation.state",
        topic="/robot/state",
        type="std_msgs/msg/Float64MultiArray",
        align=AlignSpec(method="hold", tolerance_ms=100),
        transforms=[_causal_transform(stage="post", fs_hz=30.0)],
    )
    LiveAdapter(_make_contract(fps=30.0, observations=[spec]))  # must not raise


def test_refuses_transform_on_video_spec():
    video_spec = ObservationSpec(
        key="observation.cam",
        topic="/camera/compressed",
        type="sensor_msgs/msg/CompressedImage",
        image={"resize": [64, 64]},
        align=AlignSpec(method="hold", tolerance_ms=100),
        transforms=[_causal_transform(stage="post")],
    )
    with pytest.raises(ValueError) as exc:
        LiveAdapter(_make_contract(videos=[video_spec]))
    msg = str(exc.value).lower()
    assert "video" in msg
    assert "transform" in msg


@pytest.mark.parametrize("transform_type", ["butterworth_lowpass", "resample_uniform", "finite_difference"])
def test_refuses_any_transform_other_than_causal_lowpass(transform_type):
    spec = ObservationSpec(
        key="observation.state",
        topic="/robot/state",
        type="std_msgs/msg/Float64MultiArray",
        align=AlignSpec(method="hold", tolerance_ms=100),
        transforms=[TransformSpec(type=transform_type, params={"cutoff_hz": 5.0}, stage="pre")],
    )
    with pytest.raises(ValueError) as exc:
        LiveAdapter(_make_contract(observations=[spec]))
    assert transform_type in str(exc.value)


def test_action_causal_transform_boots_and_is_unfiltered_passthrough():
    """A butterworth_lowpass_causal transform on an action spec must boot
    (unlike every other transform), and encode_action must publish the
    policy's raw output unchanged — no filter is ever applied to it."""
    spec = ActionSpec(
        key="action",
        topic="/teleop/action",
        type="std_msgs/msg/Float64MultiArray",
        transforms=[_causal_transform(stage="pre", fs_hz=50.0)],
    )
    adapter = LiveAdapter(_make_contract(actions=[spec]))  # must not raise

    action = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = adapter.encode_action({"action": action}, now_ns=0)

    assert len(out) == 1
    topic, payload = out[0]
    assert topic == "/teleop/action"
    np.testing.assert_allclose(payload["data"], [1.0, 2.0, 3.0])
