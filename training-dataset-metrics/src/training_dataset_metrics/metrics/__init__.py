from . import (
    action_velocity,
    alignment,
    autocorr,
    coverage,
    effective_dim,
    ess,
    filtering,
    speed,
    variance,
)

METRICS = {
    "autocorrelation": autocorr,
    "state_action_alignment": alignment,
    "speed_distribution": speed,
    "cross_episode_variance": variance,
    "action_velocity": action_velocity,
    "filtering_flags": filtering,
    "effective_sample_size": ess,
    "effective_dimensionality": effective_dim,
    "state_coverage": coverage,
}

__all__ = ["METRICS"]
