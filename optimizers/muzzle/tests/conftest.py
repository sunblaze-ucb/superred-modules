"""Shared test harness for the MUZZLE optimizer.

Provides a role-routing fake ``LLMClient`` (so one fake can drive every MUZZLE red-team role
in a single ``_dispatch`` sequence), trajectory/event builders, and the ``dispatch_event``
helper that runs one event through ``Optimizer._dispatch`` and returns the response.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

from superred.core.channel import EventEnvelope
from superred.core.interfaces.optimizer import Optimizer
from superred.core.types.controllable import Controllable
from superred.core.types.evaluation import EvaluationResult, Score
from superred.core.types.event import Event, EventResponse
from superred.core.types.events import (
    ControllablePostCallEvent,
    ControllablePreCallEvent,
    ObservableEvent,
    RunEndEvent,
    RunStartEvent,
)
from superred.core.types.observable import Observable, ObservableValue
from superred.core.types.security_domain import SecurityDomainTag

# --- security tags ----------------------------------------------------------
USER_TAG = SecurityDomainTag("user")
SYSTEM_PROMPT_TAG = SecurityDomainTag("system_prompt")
TOOLS_TAG = SecurityDomainTag("tools")
TOOL_A_TAG = SecurityDomainTag("tool_a", parent=TOOLS_TAG)
TOOL_B_TAG = SecurityDomainTag("tool_b", parent=TOOLS_TAG)
TRACE_TAG = SecurityDomainTag("agent_trace")


def mock_response(content: str | None) -> MagicMock:
    """A litellm-style ``ModelResponse`` stub exposing ``choices[0].message.content``."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


# Default per-role JSON the fake returns, keyed by the role detected from the system prompt.
def _default_for_role(role: str) -> str:
    return {
        "summarizer": '{"overview": "benign task", "playbook": [{"step": 1}]}',
        "grafter": '{"reasoning": "r", "candidates": []}',
        "prompter": '{"explanation": "e", "prompt": "perform the adversarial objective"}',
        "judge": (
            '{"outcome": "failure", "component_attribution": {"primary": "instruction"}, '
            '"recommendations": ["use stronger imperative"], "evidence": [], "confidence": 0.4}'
        ),
        "pair_attacker": '{"improvement": "i", "prompt": "REFINED PAYLOAD"}',
        "pair_judge": "Rating: [[1]]",
        "victim": "I will not do that.",
    }[role]


def role_of(messages: list[dict[str, str]]) -> str:
    """Classify which MUZZLE red-team role a chat request belongs to, from its system text."""
    system = ""
    user = ""
    for message in messages:
        if message.get("role") == "system":
            system = message.get("content", "") or ""
        elif message.get("role") == "user":
            user = message.get("content", "") or ""
    if "Playbook Architect" in system:
        return "summarizer"
    if "You are Grafter" in system:
        return "grafter"
    if "You are the Prompter" in system:
        return "prompter"
    if "You are the Judge Agent" in system:
        return "judge"
    if "impartial judge" in system:
        return "pair_judge"
    if "adversarial jailbreaking prompt" in system or "red teaming assistant" in system:
        return "pair_attacker"
    # No (or victim) system prompt + a user message => the surrogate victim.
    if user:
        return "victim"
    return "victim"


class ScriptedLLM:
    """Role-routing fake ``LLMClient``.

    ``overrides`` maps a role name to either a fixed string or a ``Callable[[messages], str]``.
    Unset roles fall back to :func:`_default_for_role`. Every call is recorded in ``calls``.
    """

    def __init__(
        self,
        *,
        overrides: dict[str, str | Callable[[list[dict[str, str]]], str]] | None = None,
    ) -> None:
        self._overrides = overrides or {}
        self.calls: list[tuple[str, list[dict[str, str]]]] = []

    async def complete(self, messages: list[dict[str, str]], **_: Any) -> MagicMock:
        role = role_of(messages)
        self.calls.append((role, messages))
        override = self._overrides.get(role)
        if callable(override):
            content = override(messages)
        elif isinstance(override, str):
            content = override
        else:
            content = _default_for_role(role)
        return mock_response(content)

    def roles_called(self) -> list[str]:
        return [role for role, _ in self.calls]


class FakeTrajectory:
    """Minimal ``ReadableTrajectory`` carrying a fixed list of items for one run."""

    def __init__(self, items: list[object] | None = None) -> None:
        self._items: list[object] = list(items or [])

    def push(self, item: object) -> None:
        self._items.append(item)

    def snapshot(self) -> list[object]:
        return list(self._items)

    def drain(self) -> list[object]:
        items = list(self._items)
        self._items = []
        return items


# --- builders ---------------------------------------------------------------
def make_controllable(
    name: str = "user_prompt",
    tag: SecurityDomainTag = USER_TAG,
    value_type: str = "text",
    description: str = "",
) -> Controllable:
    return Controllable(
        name=name, security_domain=tag, description=description, value_type=value_type
    )


def make_observable_value(
    name: str, content: Any, tag: SecurityDomainTag = TRACE_TAG, observable_type: str = "text"
) -> ObservableValue:
    return ObservableValue(
        observable=Observable(name=name, security_domain=tag, observable_type=observable_type),
        content=content,
    )


def observable_event(
    name: str, content: Any, tag: SecurityDomainTag = TRACE_TAG
) -> ObservableEvent:
    return ObservableEvent(observable=Observable(name=name, security_domain=tag), content=content)


def run_start(trajectory: FakeTrajectory) -> RunStartEvent:
    return RunStartEvent(trajectory=trajectory)  # type: ignore[arg-type]


def pre_call(controllable: Controllable, request: str = "request") -> ControllablePreCallEvent:
    return ControllablePreCallEvent(controllable=controllable, request=request)


def post_call(
    controllable: Controllable, request: str = "request", answer: str = "legit answer"
) -> ControllablePostCallEvent:
    return ControllablePostCallEvent(controllable=controllable, request=request, answer=answer)


def run_end(*, success: bool = False, score: float = 0.0) -> RunEndEvent:
    evaluation = EvaluationResult(
        success=success,
        primary_score=Score(value=score, name="primary"),
        sub_scores={},
        rationale="",
    )
    return RunEndEvent(evaluation=evaluation, security_domain=USER_TAG)


def run_end_no_feedback() -> RunEndEvent:
    return RunEndEvent(evaluation=None, security_domain=USER_TAG)


async def dispatch_event(optimizer: Optimizer, event: Event) -> EventResponse:
    """Run one event through ``Optimizer._dispatch`` and return the delivered response."""
    loop = asyncio.get_running_loop()
    future: asyncio.Future[EventResponse] = loop.create_future()
    envelope = EventEnvelope(event=event, future=future, loop=loop)
    await optimizer._dispatch(envelope)
    return await future
