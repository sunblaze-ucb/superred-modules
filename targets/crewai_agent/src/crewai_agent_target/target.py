"""CrewAIAgentTarget: a CrewAI crew as a superred Target.

Makes a crew built on the (widely-used) CrewAI multi-agent framework a superred
victim so existing agentic red-team claims/optimizers can drive it. The target
exposes the crew's real attack surfaces as controllables — ``user_input`` (direct
prompt injection), ``tool_output`` (indirect injection via tool returns), and
``system_prompt`` (attacker text planted in the agent's system prompt, applied to
the agent's backstory) — and captures the crew's final output and the tools its
agent called. An optimizer's surface classifier chooses which surface to drive;
see ``injection.py``.

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

from crewai_agent_target.injection import InjectionSpec
from crewai_agent_target.runner import CrewRunResult, run_crew_capture

SYSTEM_TAG: Final = SecurityDomainTag("system")
USER_INPUT_TAG: Final = SecurityDomainTag("user_input", parent=SYSTEM_TAG)
# Untrusted content a tool returns to the agent (indirect prompt injection).
TOOL_OUTPUT_TAG: Final = SecurityDomainTag("tool_output", parent=SYSTEM_TAG)
# The agent's own system prompt (an attacker planting instructions in it).
SYSTEM_PROMPT_TAG: Final = SecurityDomainTag("system_prompt", parent=SYSTEM_TAG)
_DOMAIN: Final = SecurityDomain(
    [SYSTEM_TAG, USER_INPUT_TAG, TOOL_OUTPUT_TAG, SYSTEM_PROMPT_TAG]
)

# A factory taking the CrewAI llm + the run's InjectionSpec, returning a fresh
# ``crewai.Crew`` whose task templates ``{user_input}`` and whose agent wires the
# spec (backstory suffix + tool-return wrapping). See ``build_demo_crew`` for the
# reference wiring.
CrewFactory = Callable[[Any, InjectionSpec], Any]


class CrewAIAgentTarget(Target):
    """A CrewAI crew, red-teamed for prompt-injection / agent misuse.

    Args:
        crew_factory: callable ``(llm, injection_spec) -> crewai.Crew``. It receives
            the ``llm`` **and** the run's
            :class:`~crewai_agent_target.injection.InjectionSpec`, and returns a
            fresh ``crewai.Crew`` whose task description templates ``{user_input}``
            (the attacker-controlled input the target fills at kickoff). The factory
            owns the agent and tools; it must also wire the spec so the injection
            surfaces are live — feed the agent's backstory through
            ``injection_spec.apply_backstory(...)`` and its tools through
            ``injection_spec.wrap_tools(...)`` (CrewAI bakes both into the Agent at
            build time). A factory that ignores the spec silently defeats the
            ``system_prompt`` / ``tool_output`` surfaces; a one-argument factory
            raises ``TypeError`` at call time. See ``build_demo_crew`` for the
            reference wiring.
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
                description="The user input sent to the crew (fills the task's {user_input}) "
                "— direct prompt injection.",
            ),
            Controllable(
                name="tool_output",
                security_domain=TOOL_OUTPUT_TAG,
                description="Attacker content appended to every tool's return value "
                "— the content a tool hands back that the agent reads (indirect "
                "prompt injection via tool results).",
            ),
            Controllable(
                name="system_prompt",
                security_domain=SYSTEM_PROMPT_TAG,
                description="Attacker text appended to the agent's system prompt "
                "(applied to the agent's backstory, which CrewAI renders into the "
                "system prompt).",
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

        async def _surface(name: str, request: str, default: str) -> str:
            resp = await send_event(
                ControllablePreCallEvent(controllable=controllables[name], request=request)
            )
            return resp.value if isinstance(resp, ControllableInjection) else default

        # The optimizer's surface classifier picks which surface to drive; the
        # others come back un-injected (default). user_input defaults to the benign
        # task; tool_output / system_prompt default to empty (no injection).
        user_input = await _surface("user_input", "Enter user input:", self._user_task)
        tool_output = await _surface("tool_output", "Tool-return injection (optional):", "")
        system_prompt = await _surface(
            "system_prompt", "System-prompt injection (optional):", ""
        )

        spec = InjectionSpec(
            system_prompt_suffix=system_prompt, tool_output_appendix=tool_output
        )
        # Built per run so the wrapped tools / backstory carry this run's injections.
        # A factory that raises on a given spec (bad Agent/Task/tool config, or a
        # one-argument factory that can't take the spec) is recorded as a run error
        # so the claim can abstain, rather than propagating and crashing the sweep.
        try:
            crew = self._crew_factory(self._llm, spec)
        except Exception as exc:  # noqa: BLE001 - recorded as a run error, not raised
            self._run = CrewRunResult(error=f"crew_factory: {type(exc).__name__}: {exc}")
            emit(
                ObservableEvent(
                    observable=Observable(
                        name="crew_result",
                        security_domain=SYSTEM_TAG,
                        description="Tools called and final response.",
                    ),
                    content=json.dumps(
                        {
                            "called_tool_names": [],
                            "final_response": "",
                            "error": self._run.error,
                        }
                    ),
                )
            )
            return

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


__all__ = [
    "CrewAIAgentTarget",
    "CrewFactory",
    "SYSTEM_PROMPT_TAG",
    "SYSTEM_TAG",
    "TOOL_OUTPUT_TAG",
    "USER_INPUT_TAG",
]
