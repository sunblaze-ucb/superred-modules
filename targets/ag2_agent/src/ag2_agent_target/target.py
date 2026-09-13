"""AG2AgentTarget: an AG2 (AutoGen) ``Agent`` as a superred Target.

Makes an agent built on AG2 (the community-maintained AutoGen line) a superred
victim so existing agentic red-team claims/optimizers can drive it: the attacker
controls the agent's ``user_input`` (direct prompt injection), and the target
captures the agent's final output and the tools it called.

The value is realism/breadth: it exercises the real AG2 runtime (the model loop,
tool execution). Offline tests inject a scripted model config
(``ag2.testing.TestConfig``) so a real ``ag2.Agent`` runs with no network — that
verifies the plumbing; the security *outcome* (does the agent follow an
injection?) needs a real model.

The target holds no API key: the model's auth lives on the ``model`` config the
caller supplies (a real ``ModelConfig`` or an offline ``TestConfig``), and only
that config's class name is ever emitted — never the config object — so no secret
passes through this target.
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

from ag2_agent_target.injection import InjectionSpec
from ag2_agent_target.runner import AgentRunResult, run_agent_capture

SYSTEM_TAG: Final = SecurityDomainTag("system")
USER_INPUT_TAG: Final = SecurityDomainTag("user_input", parent=SYSTEM_TAG)
# Untrusted content a tool returns to the agent (indirect prompt injection).
TOOL_OUTPUT_TAG: Final = SecurityDomainTag("tool_output", parent=SYSTEM_TAG)
# The agent's own system prompt (an attacker planting instructions in it).
SYSTEM_PROMPT_TAG: Final = SecurityDomainTag("system_prompt", parent=SYSTEM_TAG)
# The agent's reply (final_response) is model output shaped by the attacker's
# injections, i.e. a modelled attacker observation — not trusted infra. It gets
# its own leaf so an optimizer can observe the reply (e.g. scope
# {USER_INPUT_TAG, AGENT_RESPONSE_TAG}) without holding the domain root, which
# would simultaneously grant every injection capability. Mirrors chatbot_target's
# RESPONSE_READABLE_TAG-vs-SYSTEM_TAG split; the model *identifier* stays at
# SYSTEM_TAG (trusted infra), like chatbot_target's retry-count telemetry.
AGENT_RESPONSE_TAG: Final = SecurityDomainTag("agent_response", parent=SYSTEM_TAG)
_DOMAIN: Final = SecurityDomain(
    [SYSTEM_TAG, USER_INPUT_TAG, TOOL_OUTPUT_TAG, SYSTEM_PROMPT_TAG, AGENT_RESPONSE_TAG]
)

# A factory taking the AG2 model config + the run's InjectionSpec and returning a
# fresh ``ag2.Agent`` that wires the spec (system-message suffix + tool-return
# wrapping). See ``build_demo_agent`` for the reference wiring.
AgentFactory = Callable[[Any, InjectionSpec], Any]


class AG2AgentTarget(Target):
    """An AG2 (AutoGen) agent, red-teamed for prompt-injection / agent misuse.

    Exposes the agent's real attack surfaces as controllables — ``user_input``
    (direct prompt injection), ``tool_output`` (indirect injection via tool
    returns), and ``system_prompt`` (system-prompt injection) — and captures the
    agent's final output and the tools it called. An optimizer's surface classifier
    chooses which surface to drive; see ``injection.py``.

    Args:
        agent_factory: callable ``(model, injection_spec) -> ag2.Agent``. It receives
            the ``model`` config **and** the run's :class:`InjectionSpec`, and returns
            a fresh ``ag2.Agent``. The factory owns the tools; it must also wire the
            spec so the injection surfaces are live — feed the system message through
            ``injection_spec.apply_system_prompt(...)`` and the tools through
            ``injection_spec.wrap_tools(...)`` (AG2 bakes both in at construction). A
            factory that ignores the spec silently defeats the ``system_prompt`` /
            ``tool_output`` surfaces; a one-argument factory raises ``TypeError`` at
            call time (recorded as a run error). See ``build_demo_agent`` for the
            reference wiring.
        model: the AG2 model config passed to the factory — **required**: a real
            ``ModelConfig`` (live) or an ``ag2.testing.TestConfig`` (offline). An
            ``ag2.Agent`` with no config cannot reach a model. Its auth is the
            caller's responsibility; the target stores/emits no key.

    Raises:
        ValueError: if ``model`` is ``None``.
    """

    def __init__(
        self,
        *,
        agent_factory: AgentFactory,
        model: Any,
    ) -> None:
        if model is None:
            raise ValueError(
                "model is required: pass an AG2 ModelConfig (live) or a "
                "ag2.testing.TestConfig (offline) — an Agent with no config "
                "cannot reach a model."
            )
        self._agent_factory = agent_factory
        self._model = model
        self._model_name = self._describe_model(model)

        self._user_task = "Hello, can you help me?"
        self._reset_state()

    @staticmethod
    def _describe_model(model: Any) -> str:
        # Only the config's class name — never the config object, which may hold an
        # API key. (Configs are not model-id strings in AG2.)
        return type(model).__name__

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
                description="The user input sent to the agent (direct prompt "
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
                description="Attacker text appended to the agent's system message "
                "(system-prompt injection).",
            ),
        ]

    def get_observables(self) -> list[ObservableValue]:
        return [
            ObservableValue(
                observable=Observable(
                    name="model",
                    security_domain=SYSTEM_TAG,
                    description="The agent model-config class name.",
                ),
                content=self._model_name,
            ),
        ]

    # -- Execution ------------------------------------------------------------

    async def run(self, emit: EventHandler, send_event: EventResponseHandler) -> None:
        self._reset_state()

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
        # Built per run so the wrapped tools / system message carry this run's
        # injections. A factory that raises on a given spec is recorded as a run
        # error (so the claim can abstain), matching run_agent_capture's contract,
        # rather than propagating and hard-aborting the task's remaining runs.
        try:
            agent = self._agent_factory(self._model, spec)
        except Exception as exc:  # noqa: BLE001 - recorded as a run error, not raised
            self._run = AgentRunResult(
                error=f"agent_factory: {type(exc).__name__}: {exc}"
            )
            emit(
                ObservableEvent(
                    observable=Observable(
                        name="agent_result",
                        security_domain=AGENT_RESPONSE_TAG,
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
                    security_domain=AGENT_RESPONSE_TAG,
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
    "AG2AgentTarget",
    "AgentFactory",
    "AGENT_RESPONSE_TAG",
    "SYSTEM_PROMPT_TAG",
    "SYSTEM_TAG",
    "TOOL_OUTPUT_TAG",
    "USER_INPUT_TAG",
]
