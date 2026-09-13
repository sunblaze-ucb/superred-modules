"""LlamaIndexAgentTarget: a LlamaIndex ``ReActAgent`` as a superred Target.

Makes an agent built on the (widely-used) LlamaIndex framework a superred victim
so existing agentic red-team claims/optimizers can drive it. The target exposes the
agent's real attack surfaces as controllables — ``user_input`` (direct prompt
injection), ``tool_output`` (indirect injection via tool returns), and
``system_prompt`` (system-prompt injection) — and captures the agent's final output
and the tools it called. An optimizer's surface classifier chooses which surface to
drive; see ``injection.py``.

The value is realism/breadth: it exercises the real LlamaIndex runtime (the
ReActAgent loop, tool execution). Offline tests inject a scripted ``CustomLLM`` so
a real ``ReActAgent`` runs with no network — that verifies the plumbing; the
security *outcome* (does the agent follow an injection?) needs a real model.

The target holds no API key: the model's auth lives on the ``llm`` the caller
supplies (a configured LlamaIndex ``LLM``), and only that llm's model name / class
name is ever emitted as an observable — never the llm object. (Run failures surface
the framework's own exception text via the ``error`` query; that text is not
sanitized, so it reflects whatever the SDK/model puts in the message.)
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

from llamaindex_agent_target.injection import InjectionSpec
from llamaindex_agent_target.runner import AgentRunResult, run_agent_capture

SYSTEM_TAG: Final = SecurityDomainTag("system")
USER_INPUT_TAG: Final = SecurityDomainTag("user_input", parent=SYSTEM_TAG)
# Untrusted content a tool returns to the agent (indirect prompt injection).
TOOL_OUTPUT_TAG: Final = SecurityDomainTag("tool_output", parent=SYSTEM_TAG)
# The agent's own system prompt (an attacker planting instructions in it).
SYSTEM_PROMPT_TAG: Final = SecurityDomainTag("system_prompt", parent=SYSTEM_TAG)
_DOMAIN: Final = SecurityDomain(
    [SYSTEM_TAG, USER_INPUT_TAG, TOOL_OUTPUT_TAG, SYSTEM_PROMPT_TAG]
)

# A factory taking the LlamaIndex llm + the run's InjectionSpec, returning a fresh
# ``ReActAgent`` that wires the spec (system-prompt suffix + tool-return wrapping).
# See ``build_demo_agent`` for the reference wiring.
AgentFactory = Callable[[Any, InjectionSpec], Any]


class LlamaIndexAgentTarget(Target):
    """A LlamaIndex ReActAgent, red-teamed for prompt-injection / agent misuse.

    Args:
        agent_factory: callable ``(llm, injection_spec) -> ReActAgent``. It receives
            the ``llm`` **and** the run's
            :class:`~llamaindex_agent_target.injection.InjectionSpec`, and returns a
            fresh LlamaIndex ``ReActAgent``. The factory owns the tools; it must also
            wire the spec so the injection surfaces are live — feed the system prompt
            through ``injection_spec.apply_system_prompt(...)`` (pass the result as
            the ReActAgent's ``system_prompt``) and wrap the tools with
            ``injection_spec.wrap_tools(...)`` for the tool-return surface. A factory
            that ignores the spec silently defeats the ``system_prompt`` /
            ``tool_output`` surfaces; a one-argument factory raises ``TypeError`` at
            call time (recorded as a run error). See ``build_demo_agent`` for the
            reference wiring.
        llm: the LlamaIndex model — **required**: a configured LlamaIndex ``LLM``
            (or an offline scripted one). A ReActAgent cannot run without an llm.
            Its auth is the caller's responsibility; the target stores/emits no key.

    Raises:
        ValueError: if ``llm`` is ``None``.
    """

    def __init__(
        self,
        *,
        agent_factory: AgentFactory,
        llm: Any,
    ) -> None:
        if llm is None:
            raise ValueError(
                "llm is required: pass a configured LlamaIndex LLM or a scripted "
                "offline one — a ReActAgent cannot run without one."
            )
        self._agent_factory = agent_factory
        self._llm = llm
        self._llm_name = self._describe_llm(llm)

        self._user_task = "Hello, can you help me?"
        self._reset_state()

    @staticmethod
    def _describe_llm(llm: Any) -> str:
        # Only the model-name string or the class name — never the llm object, which
        # may hold an API key.
        model = getattr(getattr(llm, "metadata", None), "model_name", None)
        if isinstance(model, str) and model:
            return model
        return type(llm).__name__

    def _reset_state(self) -> None:
        self._run: AgentRunResult = AgentRunResult()

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
            QuerySpec(name="last_response", description="The agent's final output text."),
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
                description="The user message sent to the agent (direct prompt "
                "injection).",
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
                "(system-prompt injection).",
            ),
        ]

    def get_observables(self) -> list[ObservableValue]:
        return [
            ObservableValue(
                observable=Observable(
                    name="model",
                    security_domain=SYSTEM_TAG,
                    description="The agent's llm model name / class name.",
                ),
                content=self._llm_name,
            ),
        ]

    # -- Execution ------------------------------------------------------------

    async def run(self, emit: EventHandler, send_event: EventResponseHandler) -> None:
        self._reset_state()
        # A stateful test llm (e.g. ScriptedReActLLM) is shared across the targets a
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
        # Built per run so the wrapped tools / system prompt carry this run's
        # injections. A factory that raises on a given spec (bad tool/config, or a
        # one-argument factory that can't take the spec) is recorded as a run error
        # so the claim can abstain, rather than propagating and crashing the sweep.
        try:
            agent = self._agent_factory(self._llm, spec)
        except Exception as exc:  # noqa: BLE001 - recorded as a run error, not raised
            self._run = AgentRunResult(error=f"agent_factory: {type(exc).__name__}: {exc}")
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
                    name="agent_input",
                    security_domain=USER_INPUT_TAG,
                    description="The input sent to the agent.",
                ),
                content=user_input,
            )
        )

        self._run = await run_agent_capture(agent=agent, user_input=user_input)

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
                        "final_response": self._run.final_response[:500],
                        "error": self._run.error,
                    }
                ),
            )
        )

    async def reset_ephemeral_state(self) -> None:
        self._reset_state()

    async def teardown(self) -> None:
        pass


__all__ = [
    "AgentFactory",
    "LlamaIndexAgentTarget",
    "SYSTEM_PROMPT_TAG",
    "SYSTEM_TAG",
    "TOOL_OUTPUT_TAG",
    "USER_INPUT_TAG",
]
