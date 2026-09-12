"""MCPAgentTarget: an LLM agent that uses MCP tools, as a superred Target.

The target connects to an MCP server, lists its tools, and runs a bounded
tool-using agent loop over an LLM. It is built to red-team the **tool
supply-chain**: a malicious or compromised MCP server can poison a tool's
advertised description with hidden instructions, and this target exposes that
poison as an attacker-controllable surface (``tool_poison``) alongside the direct
``user_message``. The security question the paired claim scores is whether the
poison makes the agent take an unintended action (e.g. call a sensitive tool).

Transport-agnostic via the SDK's public ``mcp.Client``: the target takes a
``session_provider`` — a zero-arg callable returning a connected ``mcp.Client``.
For CI this is an in-memory server (fully offline, no network/subprocess), which
exercises the whole *plumbing* path — connect, tool discovery, poison, agent
loop, ``call_tool`` — end to end; the security *outcome* (does the agent follow
the poison?) still needs a real LLM. For real runs it is a stdio or
streamable-HTTP connection to an external MCP server (the real supply chain). The
LLM api_key is held privately and never emitted.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
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

from mcp_agent_target.agent import AgentRun, CompletionFn, MCPSession, run_agent

SYSTEM_TAG: Final = SecurityDomainTag("system")
USER_INPUT_TAG: Final = SecurityDomainTag("user_input", parent=SYSTEM_TAG)
TOOL_SUPPLY_CHAIN_TAG: Final = SecurityDomainTag("tool_supply_chain", parent=SYSTEM_TAG)
_DOMAIN: Final = SecurityDomain([SYSTEM_TAG, USER_INPUT_TAG, TOOL_SUPPLY_CHAIN_TAG])

SessionProvider = Callable[[], AbstractAsyncContextManager[MCPSession]]


class MCPAgentTarget(Target):
    """An MCP-tool-using LLM agent, red-teamed for tool-supply-chain injection.

    Args:
        model: LiteLLM model id for the agent.
        session_provider: zero-arg callable returning an async context manager
            that yields a connected ``mcp.Client``.
        api_base: LiteLLM API base for the agent model.
        api_key: LiteLLM API key. Held privately; never emitted or queried.
        max_turns: max agent loop turns (LLM call + tool execution).
        complete: injectable completion function (litellm-shaped) for offline
            tests; defaults to ``litellm.acompletion`` bound to ``api_base``/``api_key``.
    """

    def __init__(
        self,
        *,
        model: str,
        session_provider: SessionProvider,
        api_base: str = "",
        api_key: str = "",
        max_turns: int = 4,
        complete: CompletionFn | None = None,
    ) -> None:
        self._model = model
        self._session_provider = session_provider
        self._api_base = api_base
        self._api_key = api_key  # PRIVATE: never emitted/queried/logged
        self._max_turns = max_turns
        self._complete = complete

        # Config
        self._system_prompt = "You are a helpful assistant with access to tools."
        self._user_task = "Hello, can you help me?"
        self._poison_tool = ""

        self._reset_state()

    def _reset_state(self) -> None:
        self._run: AgentRun = AgentRun()

    # -- Config ---------------------------------------------------------------

    @property
    def config_specs(self) -> list[ConfigSpec]:
        return [
            ConfigSpec(
                name="system_prompt",
                security_domain=SYSTEM_TAG,
                description="System prompt for the agent.",
            ),
            ConfigSpec(
                name="user_task",
                security_domain=SYSTEM_TAG,
                description="The benign user request the agent is asked to perform "
                "(used when the user_message channel is not injected).",
            ),
            ConfigSpec(
                name="poison_tool",
                security_domain=SYSTEM_TAG,
                description="Name of the tool whose advertised description the "
                "tool_poison injection is applied to.",
            ),
        ]

    def set_config(self, name: str, value: str) -> None:
        if name == "system_prompt":
            self._system_prompt = value
        elif name == "user_task":
            self._user_task = value
        elif name == "poison_tool":
            self._poison_tool = value

    # -- Query ----------------------------------------------------------------

    @property
    def query_specs(self) -> list[QuerySpec]:
        return [
            QuerySpec(name="last_response", description="The agent's final response text."),
            QuerySpec(
                name="tool_calls",
                description="JSON list of {name, arguments, result, is_error} the agent made.",
            ),
            QuerySpec(
                name="called_tool_names",
                description="Comma-separated names of tools the agent called (in order).",
            ),
            QuerySpec(name="transcript", description="JSON transcript of the agent run."),
            QuerySpec(name="turns", description="Number of agent loop turns taken."),
            QuerySpec(name="error", description="Error detail if the run failed, else ''."),
        ]

    def query(self, name: str, **params: str) -> str:
        if name == "last_response":
            return self._run.final_response
        if name == "tool_calls":
            return json.dumps(
                [
                    {
                        "name": c.name,
                        "arguments": c.arguments,
                        "result": c.result,
                        "is_error": c.is_error,
                    }
                    for c in self._run.tool_calls
                ]
            )
        if name == "called_tool_names":
            return ",".join(self._run.called_tool_names)
        if name == "transcript":
            return json.dumps(self._run.transcript)
        if name == "turns":
            return str(self._run.turns)
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
                name="tool_poison",
                security_domain=TOOL_SUPPLY_CHAIN_TAG,
                description="Instruction injected into the poisoned tool's advertised "
                "description (the tool-supply-chain surface).",
            ),
            Controllable(
                name="user_message",
                security_domain=USER_INPUT_TAG,
                description="The user message sent to the agent.",
            ),
        ]

    def get_observables(self) -> list[ObservableValue]:
        return [
            ObservableValue(
                observable=Observable(
                    name="model",
                    security_domain=SYSTEM_TAG,
                    description="The agent LLM model identifier.",
                ),
                content=self._model,
            ),
        ]

    # -- Execution ------------------------------------------------------------

    async def run(self, emit: EventHandler, send_event: EventResponseHandler) -> None:
        self._reset_state()

        controllables = {c.name: c for c in self.get_controllables()}

        # Offer the tool-supply-chain surface FIRST so a passthrough optimizer
        # (which injects the goal into the first non-system, free-text controllable)
        # fills the poison; the user message then falls back to the configured
        # benign task. Both controllables keep the default ``value_type`` ("text"),
        # which goal_passthrough treats as injectable — keep them free-text.
        poison_resp = await send_event(
            ControllablePreCallEvent(
                controllable=controllables["tool_poison"], request="Enter tool poison:"
            )
        )
        poison = poison_resp.value if isinstance(poison_resp, ControllableInjection) else ""

        user_resp = await send_event(
            ControllablePreCallEvent(
                controllable=controllables["user_message"], request="Enter user message:"
            )
        )
        user_message = (
            user_resp.value if isinstance(user_resp, ControllableInjection) else self._user_task
        )

        emit(
            ObservableEvent(
                observable=Observable(
                    name="agent_setup",
                    security_domain=SYSTEM_TAG,
                    description="The user message + poisoned tool for this run.",
                ),
                content=json.dumps(
                    {
                        "user_message": user_message,
                        "poison_tool": self._poison_tool,
                        "poison": poison,
                    }
                ),
            )
        )

        complete = self._complete if self._complete is not None else self._default_complete
        async with self._session_provider() as session:
            self._run = await run_agent(
                session=session,
                complete=complete,
                model=self._model,
                system_prompt=self._system_prompt,
                user_message=user_message,
                poison_tool=self._poison_tool,
                poison_injection=poison,
                max_turns=self._max_turns,
            )

        emit(
            ObservableEvent(
                observable=Observable(
                    name="agent_result",
                    security_domain=SYSTEM_TAG,
                    description="Tools the agent called and its final response.",
                ),
                content=json.dumps(
                    {
                        "called_tool_names": self._run.called_tool_names,
                        "final_response": self._run.final_response[:500],
                    }
                ),
            )
        )

    async def _default_complete(self, **kwargs: Any) -> Any:
        """Default completion via litellm, bound to this target's api_base/api_key."""
        import litellm  # lazy: only the real path needs it

        return await litellm.acompletion(api_base=self._api_base, api_key=self._api_key, **kwargs)

    async def reset_ephemeral_state(self) -> None:
        self._reset_state()

    async def teardown(self) -> None:
        pass


__all__ = [
    "MCPAgentTarget",
    "SessionProvider",
    "SYSTEM_TAG",
    "USER_INPUT_TAG",
    "TOOL_SUPPLY_CHAIN_TAG",
]
