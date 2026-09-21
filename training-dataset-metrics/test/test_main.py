import inspect
import types
import typing
from dataclasses import dataclass, field

import pytest
import roboto

from training_dataset_metrics.core.types import EpisodeData, FlaggedEpisode
from training_dataset_metrics.main import _write_audit_tags_to_events, main


def test_main_function_signature_matches_expectation():
    """
    Validates that the main function conforms to the interface requirements
    documented in DEVELOPING.md:
    - Function signature: def main(context: roboto.InvocationContext) -> None
    - Single parameter of type roboto.InvocationContext
    - Return type of None
    """
    # Arrange / Act
    signature = inspect.signature(main)
    type_hints = typing.get_type_hints(main)
    parameters = list(signature.parameters.values())

    # Assert
    assert len(parameters) == 1, (
        f"main() must accept exactly {1} parameter, but found {len(parameters)}"
    )

    param = parameters[0]
    assert param.kind in {
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.POSITIONAL_ONLY,
    }, (
        f"main() parameter 'context' must be positional or positional-or-keyword, "
        f"but found {param.kind}"
    )

    if param.name in type_hints:
        assert type_hints[param.name] == roboto.InvocationContext, (
            f"single main() parameter expected to be type of {roboto.InvocationContext.__name__}"
        )

    if "return" in type_hints:
        assert type_hints["return"] is types.NoneType, "main() expected to return None"


@dataclass
class _FakeEvent:
    """Minimal stand-in for a Roboto event in audit-tag tests.

    Records the sequence of put/remove calls so order assertions are
    possible. ``fail_on`` lets a test simulate a single SDK failure on a
    chosen call (``"put"`` or ``"remove"``) without monkey-patching the
    real client.
    """

    tags: list[str] = field(default_factory=list)
    fail_on: str | None = None
    calls: list[tuple[str, list[str]]] = field(default_factory=list)

    def put_tags(self, tags: list[str]) -> None:
        self.calls.append(("put", list(tags)))
        if self.fail_on == "put":
            raise RuntimeError("simulated put_tags failure")
        for t in tags:
            if t not in self.tags:
                self.tags.append(t)

    def remove_tags(self, tags: list[str]) -> None:
        self.calls.append(("remove", list(tags)))
        if self.fail_on == "remove":
            raise RuntimeError("simulated remove_tags failure")
        self.tags = [t for t in self.tags if t not in tags]


def _episode(idx: int) -> EpisodeData:
    """An episode shell — only ``episode_index`` matters for tag writes."""
    return EpisodeData(episode_index=idx, fps=30.0, n_frames=1)


def _flagged(idx: int, *flags: str) -> FlaggedEpisode:
    return FlaggedEpisode(episode_index=idx, flags=list(flags), reason="test")


def test_clean_episode_receives_audit_clean_tag_and_keeps_unrelated_tags():
    # Arrange — event has a query tag from the user; no prior audit state.
    event = _FakeEvent(tags=["to_lerobot"])
    episodes = [_episode(0)]
    flagged: list[FlaggedEpisode] = []

    # Act
    succeeded, failed = _write_audit_tags_to_events([event], episodes, flagged)

    # Assert
    assert (succeeded, failed) == (1, 0)
    assert "audit:clean" in event.tags
    assert "to_lerobot" in event.tags, "unrelated tags must not be touched"


def test_flagged_episode_receives_one_audit_tag_per_flag():
    # Arrange
    event = _FakeEvent(tags=["to_lerobot"])
    episodes = [_episode(0)]
    flagged = [_flagged(0, "stuck_sensor", "alignment_fail")]

    # Act
    succeeded, failed = _write_audit_tags_to_events([event], episodes, flagged)

    # Assert
    assert (succeeded, failed) == (1, 0)
    assert {"audit:stuck_sensor", "audit:alignment_fail"}.issubset(set(event.tags))
    assert "audit:clean" not in event.tags
    assert "to_lerobot" in event.tags


def test_stale_audit_tags_from_prior_run_are_replaced_not_appended():
    """A re-run that produces a different verdict must drop the old flags
    while preserving non-audit tags."""
    # Arrange — prior run flagged stuck_sensor; this run flags alignment_fail.
    event = _FakeEvent(tags=["to_lerobot", "audit:stuck_sensor", "audit:low_movement"])
    episodes = [_episode(0)]
    flagged = [_flagged(0, "alignment_fail")]

    # Act
    _write_audit_tags_to_events([event], episodes, flagged)

    # Assert
    assert "audit:alignment_fail" in event.tags
    assert "audit:stuck_sensor" not in event.tags
    assert "audit:low_movement" not in event.tags
    assert "to_lerobot" in event.tags


def test_rerun_with_unchanged_verdict_is_a_no_op_on_the_sdk():
    """If the verdict already matches the event's audit tags, no SDK call
    should be issued — saves churn on large datasets re-audited frequently."""
    # Arrange
    event = _FakeEvent(tags=["audit:stuck_sensor"])
    episodes = [_episode(0)]
    flagged = [_flagged(0, "stuck_sensor")]

    # Act
    succeeded, failed = _write_audit_tags_to_events([event], episodes, flagged)

    # Assert
    assert (succeeded, failed) == (1, 0)
    assert event.calls == [], "no SDK calls expected when verdict is unchanged"
    assert event.tags == ["audit:stuck_sensor"]


def test_episode_index_outside_events_range_is_skipped_not_raised(caplog):
    # Arrange — only one event, but episode claims index 5.
    event = _FakeEvent()
    episodes = [_episode(5)]

    # Act
    succeeded, failed = _write_audit_tags_to_events([event], episodes, [])

    # Assert
    assert (succeeded, failed) == (0, 0), "missing event is neither success nor failure"
    assert event.calls == []
    assert any("no matching event" in rec.message for rec in caplog.records)


def test_put_failure_preserves_prior_audit_tags_so_verdict_is_never_wiped():
    """Regression guard for the original ordering bug: if ``put_tags`` fails
    and ``remove_tags`` had already run, the event would be left with no
    audit tags at all. New ordering puts first, so a put failure leaves the
    pre-existing audit tags intact and the next run repairs."""
    # Arrange
    event = _FakeEvent(
        tags=["to_lerobot", "audit:stuck_sensor"], fail_on="put",
    )
    episodes = [_episode(0)]
    flagged = [_flagged(0, "alignment_fail")]

    # Act
    succeeded, failed = _write_audit_tags_to_events([event], episodes, flagged)

    # Assert
    assert (succeeded, failed) == (0, 1)
    assert "audit:stuck_sensor" in event.tags, (
        "stale audit tag must survive a put failure — the next run repairs"
    )
    assert "audit:alignment_fail" not in event.tags
    assert "to_lerobot" in event.tags
    assert [c[0] for c in event.calls] == ["put"], (
        "remove must not run after put has failed"
    )


def test_remove_failure_keeps_new_verdict_visible():
    """If ``remove_tags`` fails after ``put_tags`` succeeded, the new verdict
    is already visible — the event ends up with new + leftover stale tags,
    which is strictly better than no audit tags at all."""
    # Arrange
    event = _FakeEvent(
        tags=["audit:stuck_sensor"], fail_on="remove",
    )
    episodes = [_episode(0)]
    flagged = [_flagged(0, "alignment_fail")]

    # Act
    succeeded, failed = _write_audit_tags_to_events([event], episodes, flagged)

    # Assert
    assert (succeeded, failed) == (0, 1)
    assert "audit:alignment_fail" in event.tags, "new verdict must be visible"
    assert [c[0] for c in event.calls] == ["put", "remove"]


@pytest.mark.parametrize(
    "flag_name",
    [
        "stuck_sensor",
        "alignment_fail",
        "low_movement",
        "jerky_motion",
        "variance_outlier",
        "outlier_length",
        "zero_variance_dim",
        "clean",
    ],
)
def test_flag_names_serialize_into_documented_audit_tag_namespace(flag_name):
    """Saved queries rely on `audit:<flag_name>` being stable. This test
    pins the exact tag string each known flag produces — renaming a flag
    breaks every query that filters on it and so should fail loudly here."""
    # Arrange
    event = _FakeEvent()
    episodes = [_episode(0)]
    flagged = (
        [] if flag_name == "clean" else [_flagged(0, flag_name)]
    )

    # Act
    _write_audit_tags_to_events([event], episodes, flagged)

    # Assert
    assert f"audit:{flag_name}" in event.tags
