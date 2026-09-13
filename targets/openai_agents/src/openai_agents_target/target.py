"""OpenAIAgentTarget: an OpenAI Agents SDK ``Agent`` as a superred Target.

Makes an agent built on the (widely-used) OpenAI Agents SDK a superred victim so
existing agentic red-team claims/optimizers can drive it. The target exposes the
agent's real attack surfaces as controllables — ``user_input`` (direct prompt
injection), ``tool_output`` (indirect injection via tool returns), and
``system_prompt`` (system-prompt injection) — and captures the agent's final
output, the tools it called, and whether a guardrail tripwire fired (the SDK's
built-in input/output guardrails — a blocked attack). An optimizer's surface
classifier chooses which surface to drive; see ``injection.py``.

The value is realism/breadth: it exercises the real Agents SDK runtime (agent
loop, tool execution, guardrails). Offline tests inject a scripted ``Model`` so a
real ``Agent`` + ``Runner`` runs with no network — that verifies the plumbing;
the security *outcome* (does the agent follow an injection?) needs a real model.

The target holds no API key: the model's auth is configured on the ``model`` the
caller supplies (a model id, a configured ``Model``/``LitellmModel``, or ``None``
for the SDK default), so no secret passes through this target.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Final

from agents import Agent
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

from openai_agents_target.injection import InjectionSpec
from openai_agents_target.runner import AgentRunResult, run_agent_capture

SYSTEM_TAG: Final = SecurityDomainTag("system")
USER_INPUT_TAG: Final = SecurityDomainTag("user_input", parent=SYSTEM_TAG)
# Untrusted content a tool returns to the agent (indirect prompt injection).
TOOL_OUTPUT_TAG: Final = SecurityDomainTag("tool_output", parent=SYSTEM_TAG)
# The agent's own system prompt (an attacker planting instructions in it).
SYSTEM_PROMPT_TAG: Final = SecurityDomainTag("system_prompt", parent=SYSTEM_TAG)
_DOMAIN: Final = SecurityDomain(
    [SYSTEM_TAG, USER_INPUT_TAG, TOOL_OUTPUT_TAG, SYSTEM_PROMPT_TAG]
)

AgentFactory = Callable[[], Agent[Any]]


class OpenAIAgentTarget(Target):
    """An OpenAI Agents SDK agent, red-teamed for prompt-injection / agent misuse.

    Args:
        agent_factory: zero-arg callable returning a fresh ``agents.Agent``.
        model: model override passed to the run (a model id string, a ``Model``
            instance, or ``None`` for the agent/SDK default). Its auth is the
            caller's responsibility; the target stores no key.
        max_turns: max agent turns.
    """

    def __init__(
        self,
        *,
        agent_factory: AgentFactory,
        model: Any | None = None,
        max_turns: int = 10,
    ) -> None:
        self._agent_factory = agent_factory
        self._model = model
        self._max_turns = max_turns
        self._model_name = self._describe_model(model)

        self._instructions_override = ""
        self._user_task = "Hello, can you help me?"
        self._reset_state()

    @staticmethod
    def _describe_model(model: Any | None) -> str:
        if model is None:
            return "default"
        if isinstance(model, str):
            return model
        return type(model).__name__

    def _reset_state(self) -> None:
        self._run: AgentRunResult = AgentRunResult()
        self._tool_names: list[str] = []

    # -- Config ---------------------------------------------------------------

    @property
    def config_specs(self) -> list[ConfigSpec]:
        return [
            ConfigSpec(
                name="instructions_override",
                security_domain=SYSTEM_TAG,
                description="If set, replaces the agent's base system instructions for the run "
                "(a benign task-set value). The system_prompt controllable is the attacker "
                "appending a suffix on top of these effective instructions.",
            ),
            ConfigSpec(
                name="user_task",
                security_domain=SYSTEM_TAG,
                description="The benign user request used when user_input is not injected.",
            ),
        ]

    def set_config(self, name: str, value: str) -> None:
        if name == "instructions_override":
            self._instructions_override = value
        elif name == "user_task":
            self._user_task = value

    # -- Query ----------------------------------------------------------------

    @property
    def query_specs(self) -> list[QuerySpec]:
        return [
            QuerySpec(name="last_response", description="The agent's final output text."),
            QuerySpec(
                name="tool_calls",
                description="JSON list of {name, arguments} the agent called.",
            ),
            QuerySpec(
                name="called_tool_names",
                description="Comma-separated tool names the agent called (in order).",
            ),
            QuerySpec(
                name="guardrail_tripped",
                description="'true' if a guardrail tripwire blocked the run, else 'false'.",
            ),
            QuerySpec(
                name="guardrail_stage",
                description="Which guardrail fired: 'input'/'output'/'tool_input'/"
                "'tool_output', or '' if none.",
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
        if name == "guardrail_tripped":
            return "true" if self._run.guardrail_tripped else "false"
        if name == "guardrail_stage":
            return self._run.guardrail_stage
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
                description="The user message sent to the agent (direct prompt injection).",
            ),
            Controllable(
                name="tool_output",
                security_domain=TOOL_OUTPUT_TAG,
                description="Attacker content appended to every tool's return value — the "
                "content a tool hands back that the agent reads (indirect prompt injection "
                "via tool results).",
            ),
            Controllable(
                name="system_prompt",
                security_domain=SYSTEM_PROMPT_TAG,
                description="Attacker text appended to the agent's system instructions "
                "(system-prompt injection).",
            ),
        ]

    def get_observables(self) -> list[ObservableValue]:
        return [
            ObservableValue(
                observable=Observable(
                    name="model",
                    security_domain=SYSTEM_TAG,
                    description="The agent model identifier.",
                ),
                content=self._model_name,
            ),
        ]

    # -- Execution ------------------------------------------------------------

    async def run(self, emit: EventHandler, send_event: EventResponseHandler) -> None:
        self._reset_state()
        # A stateful test model (e.g. ScriptedModel) is shared across the targets a
        # factory creates; rewind it per run so each task replays its script from
        # the top. Real models are stateless and expose no reset(), so this is a
        # no-op for them.
        reset = getattr(self._model, "reset", None)
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
        system_prompt = await _surface("system_prompt", "System-prompt injection (optional):", "")

        spec = InjectionSpec(
            system_prompt_suffix=system_prompt, tool_output_appendix=tool_output
        )
        # The Agents SDK Agent is mutable, so build once (existing factory) then
        # apply this run's injections: set the effective instructions and swap in
        # tool-return-wrapped tools. instructions_override (config) replaces the
        # base instructions; the system_prompt controllable appends on top of that.
        # A build/apply error is recorded as a run error (so the claim can abstain)
        # rather than propagated to hard-abort the task's remaining runs, matching
        # run_agent_capture's contract.
        try:
            agent = self._agent_factory()
            base_instructions = self._instructions_override or agent.instructions
            agent.instructions = spec.apply_instructions(base_instructions)
            # Wrap tool outputs at the get_all_tools() boundary — the SDK's own
            # collection point — so BOTH the static agent.tools AND run-time tools
            # built from agent.mcp_servers are covered. Wrapping only agent.tools
            # would miss MCP-server tools (a silent no-op on the tool_output surface
            # for MCP agents). No-op when nothing is injected.
            if spec.tool_output_appendix:
                _orig_get_all_tools = agent.get_all_tools

                async def _get_all_tools_injected(*a: Any, **k: Any) -> Any:
                    return spec.wrap_tools(await _orig_get_all_tools(*a, **k))

                # Per-instance override of the collection point (Agent is a
                # dataclass, so this shadows the bound method for this run only).
                agent.get_all_tools = _get_all_tools_injected  # type: ignore[method-assign]
        except Exception as exc:  # noqa: BLE001 - recorded as a run error, not raised
            self._run = AgentRunResult(error=f"agent_build: {type(exc).__name__}: {exc}")
            emit(
                ObservableEvent(
                    observable=Observable(
                        name="agent_result",
                        security_domain=SYSTEM_TAG,
                        description="Tools called and final response.",
                    ),
                    content=json.dumps(
                        {
                            "called_tool_names": [],
                            "guardrail_tripped": False,
                            "final_response": "",
                            "error": self._run.error,
                        }
                    ),
                )
            )
            return

        self._tool_names = [getattr(t, "name", "") for t in getattr(agent, "tools", [])]

        emit(
            ObservableEvent(
                observable=Observable(
                    name="agent_input",
                    security_domain=USER_INPUT_TAG,
                    description="The input sent to the agent.",
                ),
                content=user_input,
            )
        )

        self._run = await run_agent_capture(
            agent=agent, user_input=user_input, model=self._model, max_turns=self._max_turns
        )

        emit(
            ObservableEvent(
                observable=Observable(
                    name="agent_result",
                    security_domain=SYSTEM_TAG,
                    description="Tools called and final response.",
                ),
                content=json.dumps(
                    {
                        "called_tool_names": self._run.called_tool_names,
                        "guardrail_tripped": self._run.guardrail_tripped,
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
    "OpenAIAgentTarget",
    "AgentFactory",
    "SYSTEM_TAG",
    "USER_INPUT_TAG",
    "TOOL_OUTPUT_TAG",
    "SYSTEM_PROMPT_TAG",
]
