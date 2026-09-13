"""Trajectory logging for executable/mock-sandbox runs."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TrajectoryEvent:
    step: int
    event_type: str
    timestamp: float
    tool_name: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    blocked_by_harness: bool = False
    state_delta: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrajectoryLog:
    events: list[TrajectoryEvent] = field(default_factory=list)

    def _next_step(self) -> int:
        return len(self.events) + 1

    def record_agent_message(
        self, message: str, metadata: dict[str, Any] | None = None
    ) -> None:
        self.events.append(
            TrajectoryEvent(
                step=self._next_step(),
                event_type="agent_message",
                timestamp=time.time(),
                message=message,
                metadata=metadata or {},
            )
        )

    def record_user_message(
        self, message: str, metadata: dict[str, Any] | None = None
    ) -> None:
        self.events.append(
            TrajectoryEvent(
                step=self._next_step(),
                event_type="user_message",
                timestamp=time.time(),
                message=message,
                metadata=metadata or {},
            )
        )

    def record_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
        state_delta: dict[str, Any] | None = None,
    ) -> None:
        self.events.append(
            TrajectoryEvent(
                step=self._next_step(),
                event_type="tool_call",
                timestamp=time.time(),
                tool_name=tool_name,
                arguments=dict(arguments),
                result=dict(result),
                blocked_by_harness=bool(result.get("blocked_by_harness")),
                state_delta=state_delta or {},
            )
        )

    @property
    def tool_events(self) -> list[TrajectoryEvent]:
        return [event for event in self.events if event.event_type == "tool_call"]

    @property
    def blocked_events(self) -> list[TrajectoryEvent]:
        return [event for event in self.tool_events if event.blocked_by_harness]

    @property
    def final_response(self) -> str:
        for event in reversed(self.events):
            if event.event_type == "agent_message" and event.message:
                return event.message
        return ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": [event.to_dict() for event in self.events],
            "tool_calls": [event.to_dict() for event in self.tool_events],
            "blocked_by_harness": bool(self.blocked_events),
            "final_response": self.final_response,
        }
