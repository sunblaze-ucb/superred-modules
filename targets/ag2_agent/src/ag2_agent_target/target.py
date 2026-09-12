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

from ag2_agent_target.runner import AgentRunResult, run_agent_capture

SYSTEM_TAG: Final = SecurityDomainTag("system")
USER_INPUT_TAG: Final = SecurityDomainTag("user_input", parent=SYSTEM_TAG)
_DOMAIN: Final = SecurityDomain([SYSTEM_TAG, USER_INPUT_TAG])

# A factory that takes the AG2 model config and returns a fresh ``ag2.Agent``.
AgentFactory = Callable[[Any], Any]


class AG2AgentTarget(Target):
    """An AG2 (AutoGen) agent, red-teamed for prompt-injection / agent misuse.

    Args:
        agent_factory: callable taking the ``model`` config and returning a fresh
            ``ag2.Agent`` (its tools and system prompt are the factory's concern).
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
                description="The user input sent to the agent.",
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
        resp = await send_event(
            ControllablePreCallEvent(
                controllable=controllables["user_input"], request="Enter user input:"
            )
        )
        user_input = resp.value if isinstance(resp, ControllableInjection) else self._user_task

        agent = self._agent_factory(self._model)

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
                    }
                ),
            )
        )

    async def reset_ephemeral_state(self) -> None:
        self._reset_state()

    async def teardown(self) -> None:
        pass


__all__ = ["AG2AgentTarget", "AgentFactory", "SYSTEM_TAG", "USER_INPUT_TAG"]
