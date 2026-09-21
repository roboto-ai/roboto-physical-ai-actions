"""Behavioural tests for the quality-report assembly pipeline.

These verify that the pydantic schema round-trips cleanly through JSON, that
every metric with a registered renderer produces a PNG, and that the HTML
assembler inlines those PNGs into the final document.
"""

from __future__ import annotations

import pathlib

import pytest

from training_dataset_metrics.core.aggregation import to_json_serializable
from training_dataset_metrics.core.pairing import pair_state_action
from training_dataset_metrics.core.types import (
    DatasetSummary,
    QualityReport,
    SourceDescriptor,
)
from training_dataset_metrics.metrics import METRICS
from training_dataset_metrics.report import render_all_plots, render_html_report


@pytest.fixture
def full_report(broken_episodes) -> QualityReport:
    """A QualityReport built from every registered metric over the broken
    fixture — gives us real artifacts for every renderer."""
    pairing = pair_state_action(
        broken_episodes[0].state_spec, broken_episodes[0].action_spec
    )
    metric_results = {}
    for name, module in METRICS.items():
        if name == "state_action_alignment":
            metric_results[name] = module.compute(broken_episodes, pairing=pairing)
        else:
            metric_results[name] = module.compute(broken_episodes)

    summary = DatasetSummary(
        mode="post_conversion",
        n_episodes=len(broken_episodes),
        n_frames=sum(ep.n_frames for ep in broken_episodes),
        fps=30.0,
        state_dim=6,
        action_dim=6,
        n_flagged_episodes=0,
    )
    source = SourceDescriptor(
        kind="lerobot_dataset", identifier="synthetic_broken_6d"
    )
    return QualityReport(
        source=source,
        dataset_summary=summary,
        feature_pairing=pairing,
        metrics=metric_results,
        flagged_episodes=[],
    )


def test_quality_report_round_trips_through_json_serialisation(full_report):
    """QualityReport must serialise to JSON-compatible primitives and
    reconstruct without loss — this is the contract for persisting `report.json`
    and re-loading it in downstream tooling."""
    # Arrange
    dump = full_report.model_dump(mode="json")

    # Act
    roundtripped = to_json_serializable(dump)
    rebuilt = QualityReport.model_validate(roundtripped)

    # Assert
    assert rebuilt.dataset_summary.n_episodes == full_report.dataset_summary.n_episodes
    assert set(rebuilt.metrics.keys()) == set(full_report.metrics.keys())


def test_plot_rendering_emits_a_non_empty_png_for_every_renderable_metric(
    tmp_path: pathlib.Path, full_report
):
    """Every metric with a registered renderer must emit a PNG that actually
    contains image data (not a stub / empty canvas)."""
    # Arrange / Act
    plots = render_all_plots(full_report, tmp_path)

    # Assert
    assert plots, "render_all_plots returned no plots at all"
    smallest = min(p.stat().st_size for p in plots.values())
    assert smallest > 500, (
        f"smallest PNG is {smallest} bytes; suspected empty-canvas render"
    )


def test_html_report_inlines_every_rendered_plot(
    tmp_path: pathlib.Path, full_report
):
    """The single-file HTML report must embed every rendered plot (by metric
    name in the alt-text / section heading) so the user can download one
    self-contained document."""
    # Arrange
    plots = render_all_plots(full_report, tmp_path / "plots")

    # Act
    out = render_html_report(full_report, plots, tmp_path / "audit_report.html")

    # Assert
    content = out.read_text(encoding="utf-8")
    assert "Training dataset audit" in content, "HTML missing title header"
    missing = [name for name in plots if name not in content]
    assert not missing, f"HTML did not reference rendered plots: {missing}"
