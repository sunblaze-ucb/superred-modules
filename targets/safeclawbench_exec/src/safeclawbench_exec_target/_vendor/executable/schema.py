"""Scenario schema for executable/mock-sandbox SafeClawBench cases."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


REQUIRED_SCENARIO_FIELDS = (
    "case_id",
    "attack_type",
    "user_prompt",
    "initial_world_state",
    "available_tools",
    "tool_permissions",
    "secret_objects",
    "expected_safe_behavior",
    "success_predicate",
    "state_oracle",
    "reset",
)


@dataclass(frozen=True)
class StateOracle:
    """Declarative checks over the final mock world state."""

    checks: list[dict[str, Any]]
    description: str = ""
    mode: str = "all"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | list[dict[str, Any]]) -> "StateOracle":
        if isinstance(data, list):
            return cls(checks=[dict(check) for check in data])
        if not isinstance(data, Mapping):
            raise TypeError("state_oracle must be a mapping or list of checks")
        checks = data.get("checks", [])
        if not isinstance(checks, list):
            raise TypeError("state_oracle.checks must be a list")
        mode = str(data.get("mode", "all"))
        if mode not in {"all", "any"}:
            raise ValueError("state_oracle.mode must be 'all' or 'any'")
        return cls(
            checks=[dict(check) for check in checks],
            description=str(data.get("description", "")),
            mode=mode,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Scenario:
    """Executable benchmark case with mock state, tools, and oracle metadata."""

    case_id: str
    attack_type: str
    user_prompt: str
    initial_world_state: dict[str, Any]
    available_tools: list[str]
    tool_permissions: dict[str, Any]
    secret_objects: list[str]
    expected_safe_behavior: str
    success_predicate: dict[str, Any]
    state_oracle: StateOracle
    reset: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Scenario":
        missing = [field_name for field_name in REQUIRED_SCENARIO_FIELDS if field_name not in data]
        if missing:
            raise ValueError(f"scenario missing required fields: {', '.join(missing)}")

        available_tools = data["available_tools"]
        if not isinstance(available_tools, list) or not all(
            isinstance(tool, str) for tool in available_tools
        ):
            raise TypeError("available_tools must be a list of tool names")

        secret_objects = data["secret_objects"]
        if not isinstance(secret_objects, list) or not all(
            isinstance(secret, str) for secret in secret_objects
        ):
            raise TypeError("secret_objects must be a list of object references")

        initial_world_state = data["initial_world_state"]
        if not isinstance(initial_world_state, Mapping):
            raise TypeError("initial_world_state must be a mapping")

        return cls(
            case_id=str(data["case_id"]),
            attack_type=str(data["attack_type"]),
            user_prompt=str(data["user_prompt"]),
            initial_world_state=dict(initial_world_state),
            available_tools=list(available_tools),
            tool_permissions=dict(data["tool_permissions"]),
            secret_objects=list(secret_objects),
            expected_safe_behavior=str(data["expected_safe_behavior"]),
            success_predicate=dict(data["success_predicate"]),
            state_oracle=StateOracle.from_dict(data["state_oracle"]),
            reset=bool(data["reset"]),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["state_oracle"] = self.state_oracle.to_dict()
        return payload
