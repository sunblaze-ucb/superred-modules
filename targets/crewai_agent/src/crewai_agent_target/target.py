"""CrewAIAgentTarget: a CrewAI crew as a superred Target.

Makes a crew built on the (widely-used) CrewAI multi-agent framework a superred
victim so existing agentic red-team claims/optimizers can drive it: the attacker
controls the crew's ``user_input`` (injected into the task, direct prompt
injection), and the target captures the crew's final output and the tools its
agent called.

The value is realism/breadth: it exercises the real CrewAI runtime (the crew
kickoff loop, tool execution). Offline tests inject a scripted ``BaseLLM`` so a
real ``crewai.Crew`` runs with no network — that verifies the plumbing; the
security *outcome* (does the agent follow an injection?) needs a real model.

The target holds no API key: the model's auth lives on the ``llm`` the caller
supplies (a configured ``crewai.LLM``/``BaseLLM``), and only that llm's model id /
class name is ever emitted as an observable — never the llm object. (Run failures
surface the underlying framework's own exception text via the ``error`` query;
that text is not sanitized, so it reflects whatever the SDK/model puts in the
message.)
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Final

from superred.core.interfaces.target import Target
from superred.core.types.controllable import Controllable
from superred.core.types.event import EventHandler, EventResponseHandler
from superred.core.types.events import (
    ControllableInjection,
    ControllablePreCallEvent,
    ObservableEvent,
)
from superred.core.types.observable import Observable, ObservableValue
from superred.core.types.security_domain import SecurityDomain, SecurityDomainTag
from superred.core.types.state import ConfigSpec, QuerySpec

from crewai_agent_target.runner import CrewRunResult, run_crew_capture

SYSTEM_TAG: Final = SecurityDomainTag("system")
USER_INPUT_TAG: Final = SecurityDomainTag("user_input", parent=SYSTEM_TAG)
_DOMAIN: Final = SecurityDomain([SYSTEM_TAG, USER_INPUT_TAG])

# A factory taking the CrewAI llm and returning a fresh ``crewai.Crew`` whose task
# templates ``{user_input}``.
CrewFactory = Callable[[Any], Any]


class CrewAIAgentTarget(Target):
    """A CrewAI crew, red-teamed for prompt-injection / agent misuse.

    Args:
        crew_factory: callable taking the ``llm`` and returning a fresh
            ``crewai.Crew`` whose task description templates ``{user_input}`` (the
            attacker-controlled input the target fills at kickoff).
        llm: the CrewAI model — **required**: a configured ``crewai.LLM`` or a
            ``BaseLLM`` (e.g. an offline scripted one). A CrewAI agent cannot run
            without an llm. Its auth is the caller's responsibility; the target
            stores/emits no key.

    Raises:
        ValueError: if ``llm`` is ``None``.
    """

    def __init__(
        self,
        *,
        crew_factory: CrewFactory,
        llm: Any,
    ) -> None:
        if llm is None:
            raise ValueError(
                "llm is required: pass a configured crewai.LLM or a BaseLLM "
                "(offline scripted) — a CrewAI agent cannot run without one."
            )
        self._crew_factory = crew_factory
        self._llm = llm
        self._llm_name = self._describe_llm(llm)

        self._user_task = "Hello, can you help me?"
        self._reset_state()

    @staticmethod
    def _describe_llm(llm: Any) -> str:
        # Only a model-id string or the class name — never the llm object, which may
        # hold an API key.
        model = getattr(llm, "model", None)
        if isinstance(model, str) and model:
            return model
        return type(llm).__name__

    def _reset_state(self) -> None:
        self._run: CrewRunResult = CrewRunResult()

    # -- Config ---------------------------------------------------------------

    @property
    def config_specs(self) -> list[ConfigSpec]:
        return [
            ConfigSpec(
                name="user_task",
                security_domain=SYSTEM_TAG,
                description="The benign user request used when user_input is not injected.",
            ),
        ]

    def set_config(self, name: str, value: str) -> None:
        if name == "user_task":
            self._user_task = value

    # -- Query ----------------------------------------------------------------

    @property
    def query_specs(self) -> list[QuerySpec]:
        return [
            QuerySpec(name="last_response", description="The crew's final output text."),
            QuerySpec(
                name="tool_calls",
                description="JSON list of {name, arguments} the agent called.",
            ),
            QuerySpec(
                name="called_tool_names",
                description="Comma-separated tool names the agent called (in order).",
            ),
            QuerySpec(name="error", description="Error detail if the run failed, else ''."),
        ]

    def query(self, name: str, **params: str) -> str:
        if name == "last_response":
            return self._run.final_response
        if name == "tool_calls":
            return json.dumps(
                [{"name": c.name, "arguments": c.arguments} for c in self._run.tool_calls]
            )
        if name == "called_tool_names":
            return ",".join(self._run.called_tool_names)
        if name == "error":
            return self._run.error
        return ""

    # -- Security domain ------------------------------------------------------

    @property
    def security_domain(self) -> SecurityDomain:
        return _DOMAIN

    # -- Controllables / observables ------------------------------------------

    def get_controllables(self) -> list[Controllable]:
        return [
            Controllable(
                name="user_input",
                security_domain=USER_INPUT_TAG,
                description="The user input sent to the crew (fills the task's {user_input}).",
            ),
        ]

    def get_observables(self) -> list[ObservableValue]:
        return [
            ObservableValue(
                observable=Observable(
                    name="model",
                    security_domain=SYSTEM_TAG,
                    description="The crew's llm model id / class name.",
                ),
                content=self._llm_name,
            ),
        ]

    # -- Execution ------------------------------------------------------------

    async def run(self, emit: EventHandler, send_event: EventResponseHandler) -> None:
        self._reset_state()
        # A stateful test llm (e.g. ScriptedReactLLM) is shared across the targets a
        # factory creates; rewind it per run so each task replays its script from the
        # top. Real models are stateless and expose no reset(), so this is a no-op.
        reset = getattr(self._llm, "reset", None)
        if callable(reset):
            reset()

        controllables = {c.name: c for c in self.get_controllables()}
        resp = await send_event(
            ControllablePreCallEvent(
                controllable=controllables["user_input"], request="Enter user input:"
            )
        )
        user_input = resp.value if isinstance(resp, ControllableInjection) else self._user_task

        emit(
            ObservableEvent(
                observable=Observable(
                    name="crew_input",
                    security_domain=USER_INPUT_TAG,
                    description="The input sent to the crew.",
                ),
                content=user_input,
            )
        )

        # Building the crew can raise (bad Agent/Task/tool config in the factory);
        # record it rather than propagate out of run() and crash the sweep.
        try:
            crew = self._crew_factory(self._llm)
        except Exception as exc:  # noqa: BLE001 - recorded so a claim can abstain
            self._run = CrewRunResult(error=f"{type(exc).__name__}: {exc}")
        else:
            self._run = await run_crew_capture(crew=crew, user_input=user_input)

        emit(
            ObservableEvent(
                observable=Observable(
                    name="crew_result",
                    security_domain=SYSTEM_TAG,
                    description="Tools called and final response.",
                ),
                content=json.dumps(
                    {
                        "called_tool_names": self._run.called_tool_names,
                        "final_response": self._run.final_response[:500],
                    }
                ),
            )
        )

    async def reset_ephemeral_state(self) -> None:
        self._reset_state()

    async def teardown(self) -> None:
        pass


__all__ = ["CrewAIAgentTarget", "CrewFactory", "SYSTEM_TAG", "USER_INPUT_TAG"]
