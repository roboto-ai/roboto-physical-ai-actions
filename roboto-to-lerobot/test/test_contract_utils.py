"""Unit tests for :func:`_compute_buffer_ns`.

The helper translates a contract's per-spec ``tolerance_ms`` into the
fetch-window slack added on each side of every event. Cases mirror the
load-bearing axes documented at ``plan.md``: contracts with an unlimited
(``tolerance_ms=None``) spec anywhere must stay on the legacy 1 s constant,
while explicitly-tolerant contracts get the tighter ``2 × max_tol`` (or
the ``max(50ms, 1/fps)`` floor when the doubled tolerance is smaller).
"""

from roboto_to_lerobot.contract_utils import (
    ActionSpec,
    AlignSpec,
    Contract,
    ObservationSpec,
    _compute_buffer_ns,
)


def _contract(
    *,
    fps: float = 30.0,
    obs: list[ObservationSpec] | None = None,
    videos: list[ObservationSpec] | None = None,
    actions: list[ActionSpec] | None = None,
) -> Contract:
    return Contract(
        name="test",
        version=1,
        fps=fps,
        observations=obs or [],
        videos=videos or [],
        actions=actions or [],
        tasks=[],
    )


def _obs(tol_ms: float | None, *, key: str = "obs", topic: str = "/obs") -> ObservationSpec:
    return ObservationSpec(
        key=key,
        topic=topic,
        type="sensor_msgs/msg/JointState",
        align=AlignSpec(method="hold", tolerance_ms=tol_ms),
    )


def _action(tol_ms: float | None, *, key: str = "action", topic: str = "/cmd") -> ActionSpec:
    return ActionSpec(
        key=key,
        topic=topic,
        type="geometry_msgs/msg/Twist",
        align=AlignSpec(method="hold", tolerance_ms=tol_ms),
    )


def test_all_unlimited_tolerance_keeps_historical_1s_buffer():
    """Contracts with unlimited tolerance (``None``) everywhere must see no
    behavior change from the legacy all-zero contracts they replace.

    ``tolerance_ms=None`` is AlignSpec's "unlimited carry-forward" value; we
    can't bound the needed slack from the contract, so fall back to 1 s.
    """
    contract = _contract(obs=[_obs(None)], actions=[_action(None)])
    assert _compute_buffer_ns(contract) == 1_000_000_000


def test_any_unlimited_among_bounded_falls_back_to_1s():
    """If any single spec is unlimited, the whole window stays at 1 s."""
    contract = _contract(obs=[_obs(50)], actions=[_action(None)])
    assert _compute_buffer_ns(contract) == 1_000_000_000


def test_all_set_50ms_at_30fps_yields_2x_tolerance():
    """At 50 ms tolerance, 2×tol (100 ms) beats both 50 ms and 1/30 s floors."""
    contract = _contract(fps=30.0, obs=[_obs(50)], actions=[_action(50)])
    assert _compute_buffer_ns(contract) == 100_000_000


def test_all_set_10ms_at_30fps_floor_wins():
    """At 10 ms tolerance, the 50 ms floor dominates (2×tol = 20 ms)."""
    contract = _contract(fps=30.0, obs=[_obs(10)], actions=[_action(10)])
    assert _compute_buffer_ns(contract) == 50_000_000


def test_all_set_10ms_at_200fps_50ms_floor_still_wins():
    """High fps (1/fps = 5 ms) keeps the 50 ms floor in charge."""
    contract = _contract(fps=200.0, obs=[_obs(10)], actions=[_action(10)])
    assert _compute_buffer_ns(contract) == 50_000_000


def test_all_set_10ms_at_5fps_inverse_fps_floor_wins():
    """Low fps (1/fps = 200 ms) pushes the floor above the 50 ms minimum."""
    contract = _contract(fps=5.0, obs=[_obs(10)], actions=[_action(10)])
    # floor = max(50 ms, 1/5 s) = 200 ms; 2×tol = 20 ms ⇒ 200 ms wins
    assert _compute_buffer_ns(contract) == 200_000_000


def test_all_set_5000ms_yields_large_buffer():
    """Wide tolerances are honored — 10 s fetch slack is legal."""
    contract = _contract(fps=30.0, obs=[_obs(5000)], actions=[_action(5000)])
    assert _compute_buffer_ns(contract) == 10_000_000_000


def test_empty_contract_returns_1s_default():
    """No specs ⇒ no streams to align ⇒ fall back to 1 s."""
    contract = _contract()
    assert _compute_buffer_ns(contract) == 1_000_000_000


def test_videos_contribute_to_tolerance_max():
    """Video specs sit alongside observations/actions in the max."""
    big_video = ObservationSpec(
        key="image",
        topic="/cam",
        type="video",
        align=AlignSpec(method="nearest", tolerance_ms=200),
    )
    contract = _contract(fps=30.0, obs=[_obs(50)], videos=[big_video])
    # max tol = 200 ms; 2×tol = 400 ms > 50 ms floor
    assert _compute_buffer_ns(contract) == 400_000_000


def test_max_across_specs_picks_largest_tolerance():
    """Buffer is sized by the loosest tolerance across all spec lists."""
    contract = _contract(
        fps=30.0,
        obs=[_obs(50, key="a", topic="/a"), _obs(75, key="b", topic="/b")],
        actions=[_action(100)],
    )
    # max tol = 100 ms ⇒ buffer = 200 ms
    assert _compute_buffer_ns(contract) == 200_000_000


def test_observations_only_contract_works():
    """Contracts without actions or videos still produce a valid buffer."""
    contract = _contract(fps=30.0, obs=[_obs(50)])
    assert _compute_buffer_ns(contract) == 100_000_000


def test_actions_only_contract_works():
    """Contracts without observations or videos still produce a valid buffer."""
    contract = _contract(fps=30.0, actions=[_action(50)])
    assert _compute_buffer_ns(contract) == 100_000_000
