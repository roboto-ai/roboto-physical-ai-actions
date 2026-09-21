from dataclasses import dataclass, field
from typing import Any

import pytest
from roboto_to_lerobot.contract_utils import (
    ActionSpec,
    Contract,
    ObservationSpec,
    resolve_role_bindings,
)


@dataclass
class FakeMessagePath:
    message_path: str


@dataclass
class FakeTopic:
    topic_name: str
    message_paths: list[FakeMessagePath]


@dataclass
class FakeFile:
    relative_path: str
    metadata: dict[str, Any]
    topics: list[FakeTopic] = field(default_factory=list)

    def get_topics(self):
        return list(self.topics)


@dataclass
class FakeDataset:
    dataset_id: str
    files: list[FakeFile]

    def list_files(self):
        return list(self.files)


def _probe_file(topic_name: str = "Telemetry_probe") -> FakeFile:
    return FakeFile(
        relative_path=f"{topic_name}.parquet",
        metadata={"role": "probe"},
        topics=[
            FakeTopic(
                topic_name=topic_name,
                message_paths=[
                    FakeMessagePath("base_position_0"),
                    FakeMessagePath("joint_positions_1"),
                    FakeMessagePath("effector_pose_0_3"),
                ],
            )
        ],
    )


def _tool_file(topic_name: str = "Telemetry_tool") -> FakeFile:
    return FakeFile(
        relative_path=f"{topic_name}.parquet",
        metadata={"role": "tool"},
        topics=[
            FakeTopic(
                topic_name=topic_name,
                message_paths=[
                    FakeMessagePath("base_position_0"),
                    FakeMessagePath("joint_positions_1"),
                    FakeMessagePath("effector_pose_0_3"),
                ],
            )
        ],
    )


def _contract(*, obs_roles: list[str | None], act_roles: list[str | None]) -> Contract:
    observations = []
    for role in obs_roles:
        topic = "legacy_topic" if role is None else f"__role__:{role}"
        observations.append(
            ObservationSpec(
                key="observation.state",
                topic=topic,
                type="string_typed_msg",
                role=role,
                selector={"names": ["base_position_0", "joint_positions_1"]},
            )
        )
    actions = []
    for role in act_roles:
        topic = "legacy_topic" if role is None else f"__role__:{role}"
        actions.append(
            ActionSpec(
                key="action",
                topic=topic,
                type="string_typed_msg",
                role=role,
                selector={"names": ["effector_pose_0_3"]},
            )
        )
    return Contract(
        name="test",
        version=1,
        fps=30.0,
        observations=observations,
        videos=[],
        actions=actions,
        tasks=[],
    )


def test_resolve_role_bindings_happy_path():
    dataset = FakeDataset(
        dataset_id="ds_test",
        files=[_probe_file("Telemetry_probe_n23"), _tool_file("Telemetry_tool_n24")],
    )
    contract = _contract(obs_roles=["probe", "tool"], act_roles=["probe", "tool"])

    resolved = resolve_role_bindings(dataset, contract)

    assert [o.topic for o in resolved.observations] == [
        "Telemetry_probe_n23",
        "Telemetry_tool_n24",
    ]
    assert [a.topic for a in resolved.actions] == [
        "Telemetry_probe_n23",
        "Telemetry_tool_n24",
    ]
    # Roles are preserved for downstream visibility.
    assert [o.role for o in resolved.observations] == ["probe", "tool"]


def test_resolve_role_bindings_legacy_topic_unchanged():
    dataset = FakeDataset(dataset_id="ds_legacy", files=[])
    contract = _contract(obs_roles=[None], act_roles=[None])

    resolved = resolve_role_bindings(dataset, contract)

    assert resolved.observations[0].topic == "legacy_topic"
    assert resolved.observations[0].role is None


def test_resolve_role_bindings_errors_when_role_missing():
    dataset = FakeDataset(dataset_id="ds_missing", files=[_probe_file()])
    contract = _contract(obs_roles=["probe", "tool"], act_roles=[])

    with pytest.raises(ValueError, match="Role 'tool'"):
        resolve_role_bindings(dataset, contract)


def test_resolve_role_bindings_errors_on_ambiguous_role():
    dataset = FakeDataset(
        dataset_id="ds_ambi",
        files=[_probe_file("a"), _probe_file("b")],
    )
    contract = _contract(obs_roles=["probe"], act_roles=[])

    with pytest.raises(ValueError, match="matched 2 files"):
        resolve_role_bindings(dataset, contract)


def test_resolve_role_bindings_errors_when_topic_missing_selector_fields():
    incomplete = FakeFile(
        relative_path="incomplete.parquet",
        metadata={"role": "probe"},
        topics=[
            FakeTopic(
                topic_name="incomplete",
                message_paths=[FakeMessagePath("unrelated_field")],
            )
        ],
    )
    dataset = FakeDataset(dataset_id="ds_bad", files=[incomplete])
    contract = _contract(obs_roles=["probe"], act_roles=[])

    with pytest.raises(ValueError, match="No topic"):
        resolve_role_bindings(dataset, contract)


def test_role_bound_specs_have_distinct_unique_keys_pre_resolution():
    """Two role-bound specs sharing a base key must not collide on
    ``unique_key`` before ``resolve_role_bindings`` runs.

    This is what lets ``DataCollection`` and ``Contract.get_lerobot_features``
    keep them in separate buckets while the contract is still in placeholder
    form.
    """
    contract = _contract(obs_roles=["probe", "tool"], act_roles=["probe", "tool"])
    obs_keys = [o.unique_key for o in contract.observations]
    act_keys = [a.unique_key for a in contract.actions]

    assert len(set(obs_keys)) == 2, f"observation unique_keys collided: {obs_keys}"
    assert len(set(act_keys)) == 2, f"action unique_keys collided: {act_keys}"
    # And the role identity should be visible in the placeholder.
    assert any("probe" in k for k in obs_keys)
    assert any("tool" in k for k in obs_keys)
