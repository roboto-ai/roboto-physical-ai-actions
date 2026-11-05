"""Dataset sizing must accumulate evidence across episode boundaries."""

from __future__ import annotations

import importlib
import json
import logging
from types import SimpleNamespace

import numpy as np
import pytest

from training_dataset_metrics.core.types import EpisodeData, SourceDescriptor
from training_dataset_metrics.metrics import ess

# ---------------------------------------------------------------------------
# Synthetic input helpers
# ---------------------------------------------------------------------------


def _episode(index: int, length: int, channels: int = 1) -> EpisodeData:
    """Give the estimator a known ESS without mocking its signal processing.

    Alternating samples have a negative first-lag ACF, so the implemented
    first-negative-crossing estimator yields tau=1 and ESS=length per channel.
    """
    signal = np.tile([1.0, -1.0], length // 2)
    return EpisodeData(
        episode_index=index,
        fps=30.0,
        n_frames=length,
        action=np.repeat(signal[:, None], channels, axis=1),
    )


# ---------------------------------------------------------------------------
# Production metric and report regressions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("channels", [1, 3])
def test_dataset_ess_adds_episode_evidence_without_multiplying_channels(channels):
    episodes = [_episode(7, 4, channels), _episode(2, 8, channels)]
    original_actions = [ep.action.copy() for ep in episodes]

    result = ess.compute(episodes)

    assert [row["episode_index"] for row in result.per_episode] == [7, 2]
    assert [row["action_ess_mean"] for row in result.per_episode] == pytest.approx([4, 8])
    assert result.per_dataset["total_frames"] == 12
    assert result.per_dataset["total_ess_action_approx"] == pytest.approx(12)
    for episode, original in zip(episodes, original_actions, strict=True):
        np.testing.assert_array_equal(episode.action, original)


def test_audit_report_headline_uses_dataset_total(tmp_path, monkeypatch):
    main_module = importlib.import_module("training_dataset_metrics.main")
    episodes = [_episode(0, 4), _episode(1, 8)]
    source = SourceDescriptor(kind="lerobot_dataset", identifier="synthetic")
    monkeypatch.setattr(
        main_module, "load_post_conversion_episodes",
        lambda _: (episodes, source, {}),
    )
    monkeypatch.setattr(main_module, "METRICS", {"effective_sample_size": ess})
    context = SimpleNamespace(
        input_dir=tmp_path / "input",
        output_dir=tmp_path / "output",
        invocation_id="iv_local_ess",
        log_level=logging.INFO,
        get_optional_parameter=lambda _: None,
    )

    main_module.main(context)

    run_dir = context.output_dir / "audit_reports" / context.invocation_id
    report = json.loads((run_dir / "report.json").read_text())
    assert report["dataset_summary"]["ess_total"] == pytest.approx(12)
    assert "<div class='n'>12</div><div class='l'>effective sample size</div>" in (
        run_dir / "audit_report.html"
    ).read_text()


def test_dataset_ess_preserves_episode_estimates_and_skips_missing_action():
    first = _episode(0, 8, channels=2)
    second = _episode(1, 12, channels=2)
    # Different temporal correlation in each episode prevents a frame-count
    # shortcut or concatenation across the episode boundary from passing.
    first.action[:, 1] = np.arange(first.n_frames)
    second.action[:, 0] = np.arange(second.n_frames)
    missing = EpisodeData(episode_index=2, fps=30.0, n_frames=6)
    short = _episode(3, 2, channels=2)
    standalone = [ess.compute([ep]) for ep in (first, second)]

    combined = ess.compute([first, second, missing, short])

    assert combined.per_episode[:2] == [r.per_episode[0] for r in standalone]
    assert combined.per_episode[2:] == [
        {"episode_index": 2, "n_frames": 6},
        {"episode_index": 3, "n_frames": 2},
    ]
    assert combined.per_dataset["total_frames"] == 28
    expected = sum(r.per_dataset["total_ess_action_approx"] for r in standalone)
    assert combined.per_dataset["total_ess_action_approx"] == pytest.approx(expected)


def test_dataset_ess_averages_only_finite_channels_within_each_episode():
    episodes = [_episode(0, 4, channels=2), _episode(1, 8, channels=2)]
    episodes[0].action[:, 1] = 0  # Undefined ACF must not dilute the valid channel.

    result = ess.compute(episodes)

    assert [row["action_ess_mean"] for row in result.per_episode] == pytest.approx([4, 8])
    assert result.per_dataset["total_ess_action_approx"] == pytest.approx(12)


def test_dataset_ess_without_eligible_action_remains_unavailable():
    result = ess.compute([EpisodeData(episode_index=0, fps=30.0, n_frames=4)])

    assert np.isnan(result.per_dataset["total_ess_action_approx"])
    assert result.per_episode == [{"episode_index": 0, "n_frames": 4}]
