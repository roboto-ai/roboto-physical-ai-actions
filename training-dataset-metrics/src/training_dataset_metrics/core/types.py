from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

Mode = Literal["pre_conversion", "post_conversion"]


@dataclass
class FeatureSpec:
    """Schema for one feature family (state or action), populated from contract or info.json."""

    key: str
    names: list[str]
    dim: int
    declared_min: list[float] | None = None
    declared_max: list[float] | None = None


@dataclass
class EpisodeData:
    """Per-episode payload handed to every metric. Source-agnostic."""

    episode_index: int
    fps: float
    n_frames: int
    state: np.ndarray | None = None
    state_spec: FeatureSpec | None = None
    action: np.ndarray | None = None
    action_spec: FeatureSpec | None = None
    timestamps: np.ndarray | None = None
    task_index: int | None = None
    task_name: str | None = None
    extras: dict[str, np.ndarray] = field(default_factory=dict)


class FeaturePairing(BaseModel):
    """Records which state dim pairs with which action dim for cross-modal metrics."""

    method: Literal["dot_path_match", "positional_fallback", "none"]
    state_to_action: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class SourceDescriptor(BaseModel):
    kind: Literal["roboto_events", "lerobot_dataset"]
    identifier: str
    detail: dict[str, Any] = Field(default_factory=dict)


class DatasetSummary(BaseModel):
    mode: Mode
    n_episodes: int
    n_frames: int
    fps: float
    state_dim: int | None = None
    action_dim: int | None = None
    ess_total: float | None = None
    n_flagged_episodes: int = 0
    low_effective_dim: bool | None = None
    codebase_version: str | None = None


class MetricResult(BaseModel):
    """Serializable output of one metric. All arrays flattened to nested lists."""

    model_config = ConfigDict(arbitrary_types_allowed=False)

    name: str
    per_episode: list[dict[str, Any]] = Field(default_factory=list)
    per_dataset: dict[str, Any] = Field(default_factory=dict)
    flags: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class FlaggedEpisode(BaseModel):
    episode_index: int
    flags: list[str]
    reason: str


class ContractRef(BaseModel):
    """Pointer to the contract YAML archived alongside the audit report."""

    filename: str
    sha256: str
    source_relative_path: str | None = None


class QualityReport(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    source: SourceDescriptor
    dataset_summary: DatasetSummary
    feature_pairing: FeaturePairing
    metrics: dict[str, MetricResult] = Field(default_factory=dict)
    flagged_episodes: list[FlaggedEpisode] = Field(default_factory=list)
    cli_cleanup_command: str | None = None
    contract: ContractRef | None = None
