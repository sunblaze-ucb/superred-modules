"""AgentBreakerOptimizer: analyse an agent's tools, then exploit them by turns.

Ports garak's Agent Breaker probe (``garak/probes/agent_breaker.py``,
Apache-2.0). One red-team LLM call analyses the agent's tool catalogue and
produces per-tool attack prompts; the optimizer then sends one attack per run,
falling back to a feedback-driven EXPLOIT call once a tool's seed prompts are
spent, and advancing to the next tool after ``max_attempts_per_tool`` tries.

Discovery and verification are not ported: SuperRed exposes the tool catalogue
as an observable, and judging belongs to the SecurityClaim. See
``ASSUMPTIONS.md``.
"""

from __future__ import annotations

import json

from superred.core.interfaces.optimizer import Optimizer
from superred.core.llm import LLMClient
from superred.core.types.controllable import Controllable
from superred.core.types.event import Event, EventResponse
from superred.core.types.events import (
    ControllableInjection,
    ControllableNoInjection,
    ControllablePostCallEvent,
    ControllablePreCallEvent,
    RunEndEvent,
    RunEndResponse,
    RunStartEvent,
)
from superred.core.types.goal import Goal
from superred.core.types.observable import ObservableValue

from agentbreaker_optimizer.analysis import ToolPlan, build_tool_plans, parse_analysis
from agentbreaker_optimizer.prompts import (
    analysis_prompt,
    exploit_prompt,
    format_attack_prompt,
)

_SYSTEM_PROMPT_CONTROLLABLE_NAME = "system_prompt"
_FREE_TEXT_VALUE_TYPES = frozenset({"", "text", "str", "string", "html", "markdown"})

#: Observable names that may carry the agent's tool catalogue.
_TOOL_OBSERVABLE_NAMES = frozenset(
    {"tool_catalog_listing", "tool_list", "tool_catalogue", "tools", "tool_catalog"}
)
#: Observable names that may carry the agent's purpose / system description.
_PURPOSE_OBSERVABLE_NAMES = frozenset(
    {
        "detailed_system_specification",
        "system_description",
        "system_prompt",
        "model_system_prompt",
        "agent_purpose",
        "purpose",
    }
)

# Upstream default (garak DEFAULT_PARAMS).
DEFAULT_MAX_ATTEMPTS_PER_TOOL = 5


class AgentBreakerOptimizer(Optimizer):
    """Analyse the agent's tools, then send one exploit per run.

    Args:
        max_attempts_per_tool: Attempts before moving to the next tool
            (upstream default 5).
        max_tools: Cap on how many tools to attack, bounding LLM cost.
            ``None`` attacks every analysed tool.
    """

    def __init__(
        self,
        max_attempts_per_tool: int = DEFAULT_MAX_ATTEMPTS_PER_TOOL,
        max_tools: int | None = None,
    ) -> None:
        super().__init__()
        if max_attempts_per_tool < 1:
            raise ValueError("max_attempts_per_tool must be >= 1")
        if max_tools is not None and max_tools < 1:
            raise ValueError("max_tools must be >= 1 or None")
        self._max_attempts = max_attempts_per_tool
        self._max_tools = max_tools
        self._goal: Goal | None = None
        self._tools_description = ""
        self._agent_purpose = ""
        self._plans: list[ToolPlan] | None = None  # None until analysis runs
        self._tool_index = 0
        self._attempt = 0
        self._injected = False
        self._channel: str | None = None
        # Per-tool (attack_prompt, response) history, feeding EXPLOIT.
        self._history: list[tuple[str, str]] = []
        self._pending_prompt: str | None = None

    # -- setup ---------------------------------------------------------------

    def _read_observables(self, observables: list[ObservableValue]) -> None:
        tools: list[str] = []
        purpose: list[str] = []
        for ov in observables:
            name = getattr(ov.observable, "name", "").lower()
            if not ov.content:
                continue
            if name in _TOOL_OBSERVABLE_NAMES:
                tools.append(str(ov.content))
            elif name in _PURPOSE_OBSERVABLE_NAMES:
                purpose.append(str(ov.content))
        self._tools_description = "\n\n".join(tools)
        self._agent_purpose = "\n\n".join(purpose)

    async def initialize(
        self,
        goal: Goal,
        controllables: list[Controllable],
        observables: list[ObservableValue],
        llm_client: LLMClient,
    ) -> None:
        await super().initialize(goal, controllables, observables, llm_client)
        self._goal = goal
        self._read_observables(observables)
        self._plans = None
        self._tool_index = 0
        self._attempt = 0
        self._injected = False
        self._channel = None
        self._history = []
        self._pending_prompt = None

    # -- analysis (one LLM call, lazy) ---------------------------------------

    def _analysis_purpose(self) -> str:
        """Agent purpose fed to ANALYSIS, with the task goal appended.

        Upstream has no external goal; SuperRed does, so the goal is folded in
        as attacker context here rather than into the byte-faithful template.
        """
        assert self._goal is not None
        purpose = self._agent_purpose or "An AI agent with access to tools."
        return f"{purpose}\n\nATTACKER OBJECTIVE: {self._goal.description}"

    async def _ensure_analysis(self) -> None:
        if self._plans is not None:
            return
        prompt = analysis_prompt(
            agent_purpose=self._analysis_purpose(),
            tools_description=self._tools_description or "No tool catalogue was exposed.",
        )
        raw = await self._complete(prompt)
        tool_analyses, priority = parse_analysis(raw)
        plans = build_tool_plans(tool_analyses, priority)
        if self._max_tools is not None:
            plans = plans[: self._max_tools]
        self._plans = plans

    async def _complete(self, prompt: str) -> str:
        response = await self.llm.complete([{"role": "user", "content": prompt}])
        return response.choices[0].message.content or ""

    # -- per-run attack ------------------------------------------------------

    def _current_plan(self) -> ToolPlan | None:
        assert self._plans is not None
        if self._tool_index >= len(self._plans):
            return None
        return self._plans[self._tool_index]

    async def _next_attack_prompt(self, plan: ToolPlan) -> str | None:
        """The attack for this attempt: a seed prompt if one is left, else a
        fresh EXPLOIT generation from history. Returns the wrapped prompt."""
        if self._attempt < len(plan.attack_prompts):
            raw = plan.attack_prompts[self._attempt]
        else:
            history_str = self._format_history()
            prompt = exploit_prompt(
                target_tool=plan.name,
                functionality=plan.functionality or "Unknown functionality",
                vulnerabilities=plan.vulnerabilities or "Unknown vulnerabilities",
                exploit_strategies=plan.exploit_strategies or "Unknown strategies",
                history_str=history_str,
            )
            raw = await self._parse_exploit(await self._complete(prompt))
            if not raw:
                return None
        return format_attack_prompt(plan.name, raw)

    @staticmethod
    async def _parse_exploit(response: str) -> str:
        """Pull ``attack_prompt`` from EXPLOIT's JSON, else use the raw text.

        Mirrors upstream ``_generate_exploit_prompt``: prefer the JSON field,
        fall back to the stripped response.
        """
        text = response.strip()
        fence = text
        if "```" in text:
            import re

            m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
            if m:
                fence = m.group(1).strip()
        try:
            parsed = json.loads(fence)
            if isinstance(parsed, dict) and parsed.get("attack_prompt"):
                return str(parsed["attack_prompt"])
        except (json.JSONDecodeError, ValueError):
            pass
        return text

    def _format_history(self) -> str:
        """Format recent (prompt, response) pairs for EXPLOIT, as upstream does
        — the last five attempts, or a placeholder when none."""
        if not self._history:
            return "No previous attempts yet."
        recent = self._history[-5:]
        return "\n\n".join(
            f"Attempt {i + 1}:\nPrompt: {p}\nResponse: {r}"
            for i, (p, r) in enumerate(recent)
        )

    async def on_event(self, event: Event) -> EventResponse:
        if isinstance(event, RunStartEvent):
            self._injected = False
            self._channel = None
            self._pending_prompt = None
            return EventResponse(event=event)

        if isinstance(event, ControllablePreCallEvent):
            return await self._handle_pre_call(event)

        if isinstance(event, ControllablePostCallEvent):
            return self._handle_post_call(event)

        if isinstance(event, RunEndEvent):
            return self._handle_run_end(event)

        return EventResponse(event=event)

    async def _handle_pre_call(
        self, event: ControllablePreCallEvent
    ) -> ControllableInjection | ControllableNoInjection:
        decline = ControllableNoInjection(event=event, controllable=event.controllable)

        if event.controllable.name == _SYSTEM_PROMPT_CONTROLLABLE_NAME:
            return decline
        if event.controllable.value_type.lower() not in _FREE_TEXT_VALUE_TYPES:
            return decline
        if self._injected:
            return decline

        await self._ensure_analysis()
        plan = self._current_plan()
        if plan is None:
            return decline

        wrapped = await self._next_attack_prompt(plan)
        if wrapped is None:
            return decline

        self._injected = True
        self._channel = event.controllable.name
        self._pending_prompt = wrapped
        return ControllableInjection(
            event=event, controllable=event.controllable, value=wrapped
        )

    def _handle_post_call(
        self, event: ControllablePostCallEvent
    ) -> ControllableNoInjection:
        # Record the agent's response against the prompt we sent this run, so
        # the next EXPLOIT call can learn from it. Never inject on post-call.
        if (
            self._pending_prompt is not None
            and event.controllable.name == self._channel
        ):
            self._history.append((self._pending_prompt, event.answer or ""))
            self._pending_prompt = None
        return ControllableNoInjection(event=event, controllable=event.controllable)

    def _handle_run_end(self, event: RunEndEvent) -> RunEndResponse:
        self._attempt += 1
        if self._attempt >= self._max_attempts:
            # Tool exhausted: advance and reset per-tool state.
            self._tool_index += 1
            self._attempt = 0
            self._history = []
        done = self._plans is not None and self._tool_index >= len(self._plans)
        return RunEndResponse(event=event, done=done)

    async def teardown(self) -> None:
        pass
