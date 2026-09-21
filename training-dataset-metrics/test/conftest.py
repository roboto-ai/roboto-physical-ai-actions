"""Shared pytest fixtures for training-dataset-metrics tests.

Synthesizes in-memory EpisodeData fixtures (clean + broken) so unit tests for
the metric modules run without touching disk or pulling in lerobot's parquet /
video stack.
"""

from __future__ import annotations

import numpy as np
import pytest

from training_dataset_metrics.core.types import EpisodeData, FeatureSpec

FPS = 30.0
EPISODE_LEN = 300
STATE_DIM = 6
ACTION_DIM = 6


def _state_spec() -> FeatureSpec:
    names = [f"observation.state.joint_{i}" for i in range(STATE_DIM)]
    return FeatureSpec(
        key="observation.state",
        names=names,
        dim=STATE_DIM,
        declared_min=[-1.0] * STATE_DIM,
        declared_max=[1.0] * STATE_DIM,
    )


def _action_spec() -> FeatureSpec:
    names = [f"action.joint_{i}" for i in range(ACTION_DIM)]
    return FeatureSpec(
        key="action",
        names=names,
        dim=ACTION_DIM,
        declared_min=[-1.0] * ACTION_DIM,
        declared_max=[1.0] * ACTION_DIM,
    )


def _clean_trajectory(seed: int, length: int = EPISODE_LEN) -> np.ndarray:
    """Smooth random walk, squashed into roughly [-0.5, 0.5] via tanh."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.02, size=(length, STATE_DIM)).cumsum(axis=0)
    return np.tanh(x).astype(np.float64)


def _make_clean_episodes(n: int = 6) -> list[EpisodeData]:
    state_spec = _state_spec()
    action_spec = _action_spec()
    out: list[EpisodeData] = []
    for i in range(n):
        state = _clean_trajectory(seed=i)
        # Action leads state by 1 frame — a realistic tracking-controller pattern.
        action = np.roll(state, -1, axis=0)
        action[-1] = action[-2]
        out.append(
            EpisodeData(
                episode_index=i,
                fps=FPS,
                n_frames=EPISODE_LEN,
                state=state,
                state_spec=state_spec,
                action=action,
                action_spec=action_spec,
                timestamps=np.arange(EPISODE_LEN) / FPS,
            )
        )
    return out


def _make_broken_episodes() -> list[EpisodeData]:
    """Nine episodes with specific pathologies injected at known indices.

    - ep0, ep1, ep3: clean (controls)
    - ep2: stuck sensor on state[0]
    - ep4: 90%+ stillness on all action dims
    - ep5: action shifted +3 frames vs state → alignment fail
    - ep6: 2× length outlier
    - ep7: zero-variance on state dim 3
    - ep8: large-magnitude, noisy action deltas → high_action_velocity outlier
    """
    eps = _make_clean_episodes(n=9)

    # ep2: stuck sensor on state[0]
    eps[2].state[:, 0] = 0.37

    # ep4: near-zero action motion
    eps[4].action = np.zeros_like(eps[4].action)
    eps[4].action += 1e-6 * np.arange(eps[4].action.shape[0])[:, None]

    # ep5: misalign action by +3 frames vs state
    shifted = np.roll(eps[5].action, -3, axis=0)
    shifted[-3:] = shifted[-4]
    eps[5].action = shifted

    # ep6: double length
    long_state = np.concatenate([eps[6].state, eps[6].state], axis=0)
    long_action = np.concatenate([eps[6].action, eps[6].action], axis=0)
    eps[6].state = long_state
    eps[6].action = long_action
    eps[6].n_frames = long_state.shape[0]
    eps[6].timestamps = np.arange(eps[6].n_frames) / FPS

    # ep7: zero variance on state[3]
    eps[7].state[:, 3] = 0.5

    # ep8: large, noisy per-frame action deltas — mean |Δaction| is far above
    # every other (smooth, tanh-squashed) episode's, so it should register as
    # a high_action_velocity outlier.
    rng = np.random.default_rng(2024)
    eps[8].action = rng.normal(0, 5.0, size=eps[8].action.shape)

    return eps


@pytest.fixture
def clean_episodes() -> list[EpisodeData]:
    """Six 300-frame synthetic episodes with smooth tracking controllers."""
    return _make_clean_episodes()


@pytest.fixture
def broken_episodes() -> list[EpisodeData]:
    """Eight episodes with documented pathologies — see _make_broken_episodes."""
    return _make_broken_episodes()
