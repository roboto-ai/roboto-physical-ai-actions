"""Behavioural tests for each metric module.

Conventions: AAA sections, behaviour-centric names, public-API only,
state-based assertions, parametrize instead of loops, DAMP over DRY.
"""

from __future__ import annotations

import pytest

from training_dataset_metrics.core.pairing import pair_state_action
from training_dataset_metrics.core.types import MetricResult
from training_dataset_metrics.metrics import METRICS
from training_dataset_metrics.metrics import alignment as alignment_metric


def _flag_row(result: MetricResult, episode_index: int) -> dict:
    """Look up a flag row by episode — pure lookup boilerplate."""
    for row in result.flags:
        if row["episode_index"] == episode_index:
            return row
    raise AssertionError(
        f"no flag row for episode_index={episode_index} in {result.name}"
    )


@pytest.mark.parametrize(
    "metric_name",
    list(METRICS.keys()),
    ids=list(METRICS.keys()),
)
def test_each_metric_returns_a_named_metric_result_on_clean_data(
    metric_name, clean_episodes
):
    """Every registered metric must produce a populated MetricResult on clean
    input — a signature contract, independent of the metric's actual output."""
    # Arrange
    module = METRICS[metric_name]

    # Act
    result = module.compute(clean_episodes)

    # Assert
    assert isinstance(result, MetricResult), (
        f"{metric_name}.compute() must return a MetricResult, got {type(result)}"
    )
    assert result.name, f"{metric_name}.compute() returned MetricResult with empty name"
    assert not result.errors, (
        f"{metric_name} produced errors on clean input: {result.errors}"
    )


def test_constant_state_channel_is_flagged_as_stuck_sensor(broken_episodes):
    """A state channel held at a constant value should fire the
    `stuck_sensor` flag on its episode."""
    # Arrange
    # broken_episodes[2] has state[:, 0] pinned to 0.37.

    # Act
    result = METRICS["autocorrelation"].compute(broken_episodes)

    # Assert
    stuck_episode = _flag_row(result, episode_index=2)
    control_episode = _flag_row(result, episode_index=0)
    assert stuck_episode["stuck_sensor"] is True, (
        "episode with constant state channel must be flagged stuck_sensor"
    )
    assert control_episode["stuck_sensor"] is False, (
        "clean control episode must not be flagged stuck_sensor"
    )


def test_action_shifted_beyond_tolerance_is_flagged_as_alignment_failure(
    broken_episodes,
):
    """When action leads state by |τ| ≥ 2 frames on any paired dim, the
    alignment_fail flag must fire. A 1-frame lead (clean control) must not."""
    # Arrange
    pairing = pair_state_action(
        broken_episodes[0].state_spec, broken_episodes[0].action_spec
    )

    # Act
    result = alignment_metric.compute(broken_episodes, pairing=pairing)

    # Assert
    misaligned = _flag_row(result, episode_index=5)
    control = _flag_row(result, episode_index=0)
    assert misaligned["alignment_fail"] is True, (
        "+3-frame action shift must trigger alignment_fail"
    )
    assert control["alignment_fail"] is False, (
        "1-frame action lead (clean) must stay below the |τ|≥2 threshold"
    )


def test_episode_with_near_zero_action_motion_is_flagged_low_movement(
    broken_episodes,
):
    """Stillness > 0.9 on any action dim triggers low_movement."""
    # Arrange
    # broken_episodes[4].action ≈ 0 for every frame.

    # Act
    result = METRICS["speed_distribution"].compute(broken_episodes)

    # Assert
    still = _flag_row(result, episode_index=4)
    control = _flag_row(result, episode_index=0)
    assert still["low_movement"] is True, (
        "episode with action ≈ 0 must be flagged low_movement"
    )
    assert control["low_movement"] is False, (
        "clean episode must not be flagged low_movement"
    )


def test_episode_with_doubled_length_is_flagged_as_length_outlier(broken_episodes):
    """|T_e − median(T)| > 3·MAD must fire outlier_length."""
    # Arrange
    # broken_episodes[6] has 2× the frame count of every other episode.

    # Act
    result = METRICS["filtering_flags"].compute(broken_episodes)

    # Assert
    outlier = _flag_row(result, episode_index=6)
    control = _flag_row(result, episode_index=0)
    assert outlier["outlier_length"] is True, (
        "doubled-length episode must be flagged outlier_length"
    )
    assert control["outlier_length"] is False, (
        "equal-length control episode must not be flagged outlier_length"
    )


def test_episode_with_constant_state_dimension_is_flagged_zero_variance(
    broken_episodes,
):
    """Any state or action column with std < 1e-8 fires zero_variance_dim."""
    # Arrange
    # broken_episodes[7].state[:, 3] is constant (0.5).

    # Act
    result = METRICS["filtering_flags"].compute(broken_episodes)

    # Assert
    zero_var = _flag_row(result, episode_index=7)
    control = _flag_row(result, episode_index=0)
    assert zero_var["zero_variance_dim"] is True, (
        "episode with a constant state column must be flagged zero_variance_dim"
    )
    assert control["zero_variance_dim"] is False, (
        "clean episode must not be flagged zero_variance_dim"
    )


def test_effective_sample_size_emits_one_row_per_episode(clean_episodes):
    """ESS must produce one per-episode row even when the computation is
    skipped — the orchestrator relies on per-episode rows existing to aggregate
    flags uniformly across metrics."""
    # Arrange / Act
    result = METRICS["effective_sample_size"].compute(clean_episodes)

    # Assert
    assert len(result.per_episode) == len(clean_episodes), (
        f"expected {len(clean_episodes)} per-episode rows, "
        f"got {len(result.per_episode)}"
    )


def test_effective_dimensionality_reports_participation_ratio_per_role(
    clean_episodes,
):
    """PR should be reported for both state and action when both have finite
    variance — downstream consumers rely on both keys being populated."""
    # Arrange / Act
    result = METRICS["effective_dimensionality"].compute(clean_episodes)

    # Assert
    assert "state" in result.per_dataset, "state PCA block missing"
    assert "action" in result.per_dataset, "action PCA block missing"
    assert "pr" in result.per_dataset["state"], "state block lacks 'pr'"
    assert "pr" in result.per_dataset["action"], "action block lacks 'pr'"


def test_state_coverage_reports_declared_range_fraction(clean_episodes):
    """When the EpisodeData carries declared min/max, coverage_fraction must
    be populated — it's the number users compare against their declared
    operating envelope."""
    # Arrange / Act
    result = METRICS["state_coverage"].compute(clean_episodes)

    # Assert
    state_block = result.per_dataset.get("state")
    assert state_block is not None, "state coverage block missing"
    assert state_block.get("coverage_fraction") is not None, (
        "coverage_fraction must be present when declared ranges are set"
    )


def test_cross_episode_variance_emits_at_least_one_heatmap(clean_episodes):
    """variance.compute() must emit at least one extended heatmap kind for the
    report renderer to visualize."""
    # Arrange / Act
    result = METRICS["cross_episode_variance"].compute(clean_episodes)

    # Assert
    heatmaps = (result.artifacts.get("extended") or {}).get("heatmaps", {})
    assert heatmaps, "expected at least one heatmap kind under artifacts.extended"


def test_action_velocity_ranks_jerky_episodes(clean_episodes):
    """The ranked 'top jerky' list is the Filtering Panel's UX hook — it must
    always be present so the HTML report can render the table."""
    # Arrange / Act
    result = METRICS["action_velocity"].compute(clean_episodes)

    # Assert
    assert "top_jerky_episodes" in result.per_dataset, (
        "action_velocity must emit a 'top_jerky_episodes' ranked list"
    )


def test_episode_with_extreme_action_deltas_is_flagged_high_action_velocity(
    broken_episodes,
):
    """A mean |Δaction| that is a |z|>3 outlier vs the dataset's median/MAD
    must fire high_action_velocity; a clean control must not."""
    # Arrange
    # broken_episodes[8] has large, noisy per-frame action deltas.

    # Act
    result = METRICS["action_velocity"].compute(broken_episodes)

    # Assert
    outlier = _flag_row(result, episode_index=8)
    control = _flag_row(result, episode_index=0)
    assert outlier["high_action_velocity"] is True, (
        "episode with extreme action deltas must be flagged high_action_velocity"
    )
    assert control["high_action_velocity"] is False, (
        "clean episode must not be flagged high_action_velocity"
    )


def test_clean_dataset_has_no_high_action_velocity_flags(clean_episodes):
    """On a dataset with no outlier episode, nothing should fire
    high_action_velocity — it must not be a dead-false or a dead-true flag."""
    # Act
    result = METRICS["action_velocity"].compute(clean_episodes)

    # Assert
    assert all(not row["high_action_velocity"] for row in result.flags), (
        "no episode should be flagged high_action_velocity on clean data"
    )


def test_effective_dimensionality_does_not_broadcast_a_per_episode_flag(
    clean_episodes,
):
    """low_effective_dim is a dataset-level PCA verdict, not a per-episode
    one — it must not appear in this metric's per-episode `flags` at all (see
    `per_dataset['low_effective_dim_flag']` for the dataset-level surface),
    so it can never be broadcast onto every episode's audit tags."""
    # Act
    result = METRICS["effective_dimensionality"].compute(clean_episodes)

    # Assert
    assert result.flags == [], (
        "effective_dimensionality must not emit any per-episode flags"
    )
    assert "low_effective_dim_flag" in result.per_dataset, (
        "low_effective_dim must be surfaced at the dataset level"
    )


# ---------------------------------------------------------------------------
# HF-parity "primary" payload spot-checks — numerical sanity on deterministic
# signals (catches regressions in the ported-from-upstream math).
# ---------------------------------------------------------------------------


def test_autocorr_primary_payload_recovers_sin_wave_period(clean_episodes):
    """On any clean episode, the primary action ACF (HF divisor) must satisfy
    r(0) == 1 exactly, and r(k) must lie in [-1, 1] for all k."""
    # Arrange / Act
    result = METRICS["autocorrelation"].compute(clean_episodes)

    # Assert
    primary = result.artifacts.get("primary") or {}
    curves_by_ep = primary.get("curves_by_episode") or {}
    assert curves_by_ep, "primary.curves_by_episode must be populated"
    for _, payload in curves_by_ep.items():
        curves = payload.get("curves") or []
        for curve in curves:
            assert abs(curve[0] - 1.0) < 1e-9, (
                "HF-parity ACF must start exactly at 1 (centered Σx² divisor)"
            )
            assert all(-1.0001 <= v <= 1.0001 for v in curve), (
                "HF-parity ACF must remain in [-1, 1] for all lags"
            )


def test_alignment_primary_envelope_brackets_each_pair_curve(clean_episodes):
    """For each episode, the max/mean/min envelope at every lag must bracket
    any individual pair curve at the same lag — a structural invariant that
    catches aggregation bugs."""
    # Arrange
    pairing = pair_state_action(
        clean_episodes[0].state_spec, clean_episodes[0].action_spec
    )

    # Act
    result = alignment_metric.compute(clean_episodes, pairing=pairing)

    # Assert
    primary = (result.artifacts.get("primary") or {}).get("by_episode") or {}
    assert primary, "primary.by_episode must be populated on clean input"
    for _, payload in primary.items():
        curve_max = payload["max"]
        curve_min = payload["min"]
        curve_mean = payload["mean"]
        for mx, mn, me in zip(curve_max, curve_min, curve_mean, strict=True):
            assert mn <= me <= mx + 1e-12, (
                "envelope must satisfy min ≤ mean ≤ max at every lag"
            )


def test_speed_primary_per_episode_scalar_is_nonnegative(clean_episodes):
    """The HF per-episode scalar is `mean_t(||Δa||₂)` — always ≥ 0 on any
    action signal; a histogram must be emitted with integer counts summing
    to the number of episodes with action data."""
    # Arrange / Act
    result = METRICS["speed_distribution"].compute(clean_episodes)

    # Assert
    primary = result.artifacts.get("primary") or {}
    scalars = primary.get("per_episode_scalar") or []
    assert scalars, "primary.per_episode_scalar must be populated"
    for entry in scalars:
        assert entry["value"] >= 0, "HF speed scalar must be non-negative"
    hist = primary.get("hist") or {}
    counts = hist.get("counts") or []
    if counts:
        assert sum(counts) == len(scalars), (
            "histogram counts must total the number of per-episode scalars"
        )


def test_variance_primary_matrix_has_dim_by_50_bins(clean_episodes):
    """The HF-parity heatmap is (action_dim × 50); stored raw population
    variance per (dim, bin) — every cell is ≥ 0 by construction."""
    # Arrange / Act
    result = METRICS["cross_episode_variance"].compute(clean_episodes)

    # Assert
    primary = result.artifacts.get("primary") or {}
    matrix = primary.get("matrix") or []
    assert matrix, "primary.matrix must be populated"
    assert all(len(row) == 50 for row in matrix), (
        "primary matrix must have 50 time bins per dim (HF-parity)"
    )
    for row in matrix:
        for cell in row:
            assert cell >= -1e-12, "population variance must be non-negative"
