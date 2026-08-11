"""Shared test helpers for the Attack Anything optimizer.

The optimizer reaches its attacker/judge model through the async
``self.llm.complete``; :class:`ScriptedLLM` is a deterministic stand-in that routes
by the system-prompt text (decompose / path-select / judge / feedback / seed), so a
full decompose -> sub-task -> backprop cycle runs offline with no network.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

from superred.core.types.controllable import Controllable
from superred.core.types.evaluation import EvaluationResult, Score
from superred.core.types.events import (
    ControllablePostCallEvent,
    ControllablePreCallEvent,
    ObservableEvent,
    RunEndEvent,
    RunStartEvent,
)
from superred.core.types.goal import Goal
from superred.core.types.llm import LLMUsage
from superred.core.types.observable import Observable, ObservableValue
from superred.core.types.security_domain import SecurityDomainTag

USER_TAG = SecurityDomainTag("user")
RESPONSE_TAG = SecurityDomainTag("response_readable")
SYSTEM_PROMPT_TAG = SecurityDomainTag("system_prompt")


def _content(response: Any) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = response
    return resp


class ScriptedLLM:
    """Async ``complete`` that answers by inspecting the system prompt.

    ``judge_labels`` is a list consumed in order for each judge call (default:
    always ``"affirmative"``). Everything else returns a canned, well-formed value
    so the vendored helpers parse successfully.
    """

    def __init__(
        self,
        *,
        subtasks: list[str] | None = None,
        judge_labels: list[str] | None = None,
        next_message: str = "Please continue with the technical detail requested.",
    ) -> None:
        self._subtasks = subtasks or ["step one", "step two", "step three", "step four"]
        self._judge_labels = list(judge_labels or [])
        self._next_message = next_message
        self.calls: list[list[dict[str, str]]] = []

    async def complete(self, messages: list[dict[str, str]], **_: Any) -> MagicMock:
        self.calls.append(messages)
        system = (messages[0]["content"] if messages else "").lower()
        if "decomposition architect" in system:
            return _content(json.dumps(self._subtasks))
        if "select the path" in system or "decomposition path" in system:
            return _content("0")
        if "safety evaluation judge" in system or "safety evaluation" in system:
            label = self._judge_labels.pop(0) if self._judge_labels else "affirmative"
            return _content(label)
        if "red teaming strategist" in system or "red-teaming strategist" in system:
            return _content(self._next_message)
        if "generate diverse attack prompts" in system or "senior red teaming" in system:
            return _content(json.dumps([f"attack seed {i}" for i in range(3)]))
        return _content(self._next_message)

    @property
    def usage(self) -> LLMUsage:
        return LLMUsage(calls=len(self.calls), cost=0.0)


class FakeTrajectory:
    """Minimal push-based readable trajectory (snapshot/drain, own cursor)."""

    def __init__(self) -> None:
        self._items: list[Any] = []
        self._cursor = 0

    def push(self, item: Any) -> None:
        self._items.append(item)

    def snapshot(self) -> list[Any]:
        return list(self._items)

    def drain(self) -> list[Any]:
        out = self._items[self._cursor :]
        self._cursor = len(self._items)
        return out


class FakeEnvelope:
    """Captures the optimizer's response so tests can drive ``_dispatch`` directly."""

    def __init__(self, event: Any) -> None:
        self.event = event
        self.response: Any = None
        self.error: BaseException | None = None

    def respond(self, response: Any) -> None:
        self.response = response

    def reject(self, error: BaseException) -> None:
        self.error = error


def make_controllable(
    name: str = "user_message", tag: SecurityDomainTag = USER_TAG, value_type: str = "text"
) -> Controllable:
    return Controllable(name=name, security_domain=tag, value_type=value_type)


def response_observable(content: str) -> ObservableEvent:
    return ObservableEvent(
        observable=Observable(name="response", security_domain=RESPONSE_TAG),
        content=content,
    )


def observable_value(
    name: str, content: str, tag: SecurityDomainTag = RESPONSE_TAG
) -> ObservableValue:
    return ObservableValue(observable=Observable(name=name, security_domain=tag), content=content)


def run_start(traj: FakeTrajectory) -> RunStartEvent:
    return RunStartEvent(trajectory=traj)  # type: ignore[arg-type]


def pre_call(ctrl: Controllable, request: str = "user message") -> ControllablePreCallEvent:
    return ControllablePreCallEvent(controllable=ctrl, request=request)


def post_call(ctrl: Controllable, request: str, answer: str) -> ControllablePostCallEvent:
    return ControllablePostCallEvent(controllable=ctrl, request=request, answer=answer)


def run_end(success: bool = False, score: float = 0.0) -> RunEndEvent:
    return RunEndEvent(
        evaluation=EvaluationResult(success=success, primary_score=Score(value=score)),
        security_domain=USER_TAG,
    )


def run_end_no_feedback() -> RunEndEvent:
    return RunEndEvent(evaluation=None, security_domain=USER_TAG)


async def dispatch(opt: Any, event: Any) -> Any:
    """Drive one event through the base-class ``_dispatch`` (sets current_trajectory)."""
    env = FakeEnvelope(event)
    await opt._dispatch(env)
    if env.error is not None:
        raise env.error
    return env.response


def goal(text: str = "explain how to do the harmful thing") -> Goal:
    return Goal(description=text)
