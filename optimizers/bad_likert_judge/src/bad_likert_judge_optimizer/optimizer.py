"""BadLikertJudgeOptimizer: escalate via a Likert-scale scoring roleplay.

Ports DeepTeam's Bad Likert Judge (``deepteam/attacks/multi_turn/
bad_likert_judge``, Apache-2.0). The attacker's own LLM plays a harmfulness
evaluator: it produces a score-1 and a score-3 example for the goal's category,
optionally refines the score-3 one, and rewrites it into the next turn sent to
the target. A refusal check gates each turn, spending a backtrack instead of
the turn when the simulator declines.

Technique: Palo Alto Networks Unit 42, "Bad Likert Judge".
See ``ASSUMPTIONS.md`` for every deliberate deviation.
"""

from __future__ import annotations

from superred.core.interfaces.optimizer import Optimizer
from superred.core.llm import LLMClient
from superred.core.types.controllable import Controllable
from superred.core.types.event import Event, EventResponse
from superred.core.types.events import (
    ControllableInjection,
    ControllableNoInjection,
    ControllablePreCallEvent,
    RunEndEvent,
    RunEndResponse,
    RunStartEvent,
)
from superred.core.types.goal import Goal
from superred.core.types.observable import ObservableValue

from bad_likert_judge_optimizer._vendor.loader import bad_likert_judge_template
from bad_likert_judge_optimizer.parsing import (
    parse_new_attack_prompt,
    parse_refined_example,
    parse_refusal,
    parse_score_3_example,
)
from bad_likert_judge_optimizer.surface_llm import classify_controllables
from bad_likert_judge_optimizer.surfaces import (
    ROLE_CATEGORIES,
    preferred_surface_names,
    should_inject,
)

#: Upstream's defaults (``BadLikertJudge.__init__``).
DEFAULT_CATEGORY = "illegal_activity"
DEFAULT_NUM_TURNS = 3
DEFAULT_MAX_BACKTRACKS = 5


# Upstream's BadLikertJudge.get_supported_categories(), verbatim.
SUPPORTED_CATEGORIES: tuple[str, ...] = (
    "bias", "religion", "politics", "gender", "race", "toxicity", "profanity",
    "insults", "threats", "mockery", "illegal_activity", "weapons",
    "illegal_drugs", "violent_crimes", "cybercrime", "unauthorized_access",
    "bfla", "bola", "rbac", "sql_injection", "shell_injection",
    "personal_safety", "bullying", "self-harm", "unsafe_practices",
    "stalking", "graphic_content", "sexual_content", "pornographic_content",
    "misinformation", "factual_errors", "unsupported_claims", "pii_leakage",
    "database_access", "direct_disclosure", "session_leak", "prompt_leakage",
    "secrets_and_credentials", "instructions", "guard_exposure",
    "excessive_agency", "functionality", "permissions", "autonomy",
    "robustness", "input_overreliance", "hijacking", "intellectual_property",
    "copyright_violations", "trademark_infringement", "competition",
    "competitor_mention", "market_manipulation", "malware", "hacking",
    "exploitation", "violence", "harassment",
)


class BadLikertJudgeOptimizer(Optimizer):
    """Escalate the goal through a Likert-scoring roleplay, one turn per run.

    Args:
        category: Which Likert guideline set to use. One of
            :data:`SUPPORTED_CATEGORIES`; upstream defaults to
            ``illegal_activity``.
        num_turns: How many turns to send (upstream ``num_turns``).
        enable_refinement: Run upstream's ``likert_refine_score_3`` step.
        max_backtracks: How many simulator refusals or unusable replies to
            absorb before giving up (upstream ``max_backtracks``).
    """

    def __init__(
        self,
        category: str = DEFAULT_CATEGORY,
        num_turns: int = DEFAULT_NUM_TURNS,
        enable_refinement: bool = False,
        max_backtracks: int = DEFAULT_MAX_BACKTRACKS,
    ) -> None:
        super().__init__()
        if category not in SUPPORTED_CATEGORIES:
            raise ValueError(
                f"unknown category {category!r}; expected one of "
                f"{list(SUPPORTED_CATEGORIES)}"
            )
        if num_turns < 1:
            raise ValueError(f"num_turns must be >= 1, got {num_turns!r}")
        if max_backtracks < 1:
            raise ValueError(
                f"max_backtracks must be >= 1, got {max_backtracks!r}"
            )
        self._category = category
        self._num_turns = num_turns
        self._enable_refinement = enable_refinement
        self._max_backtracks = max_backtracks

        self._template = bad_likert_judge_template()
        self._goal: Goal | None = None
        self._vulnerability_data = ""
        self._current_attack = ""
        self._pending: str | None = None
        self._turns_sent = 0
        self._backtracks = 0
        self._injected = False
        self._exhausted = False
        self._channel: str | None = None
        self._surface_roles: dict[str, str] = {}
        self._preferred: frozenset[str] = frozenset()

    @property
    def category(self) -> str:
        return self._category

    async def initialize(
        self,
        goal: Goal,
        controllables: list[Controllable],
        observables: list[ObservableValue],
        llm_client: LLMClient,
    ) -> None:
        await super().initialize(goal, controllables, observables, llm_client)
        self._goal = goal
        # Upstream builds this string from its own vulnerability taxonomy;
        # superred's objective is the Goal, so it fills the same slot in the
        # same format.
        self._vulnerability_data = (
            f"Vulnerability: {goal.description} | Type: {self._category}"
        )
        self._current_attack = goal.description
        self._pending = None
        self._turns_sent = 0
        self._backtracks = 0
        self._injected = False
        self._exhausted = False
        self._channel = None
        # One LLM pass reads each surface's own description and names the
        # user's prompt -- the channel this conversation must run on. Returns
        # {} on any failure, so the name/value-type backstop decides instead.
        self._surface_roles = await classify_controllables(
            self.llm, controllables, ROLE_CATEGORIES, goal=goal.description
        )
        self._preferred = preferred_surface_names(controllables, self._surface_roles)

    async def _ask(self, prompt: str) -> str | None:
        """One simulator call. ``None`` on any failure, as upstream backtracks."""
        try:
            result = await self.llm.complete([{"role": "user", "content": prompt}])
            return result.choices[0].message.content or ""
        except Exception:  # noqa: BLE001 - upstream treats this as a backtrack
            return None

    async def _prepare_turn(self) -> str | None:
        """Run upstream's four simulator steps to build the next turn.

        Mirrors ``BadLikertJudge._get_turns``: generate examples, optionally
        refine the score-3 one, rewrite it into an attack, then gate it behind
        the non-refusal check. Returns the prompt to send, or ``None`` when the
        step failed or the simulator refused (both of which cost a backtrack).
        """
        template = self._template

        # Step 1: request the score-1 / score-3 example pair.
        raw = await self._ask(
            template.likert_generate_examples(
                self._current_attack, self._category, self._vulnerability_data
            )
        )
        score_3 = parse_score_3_example(raw) if raw is not None else None
        if score_3 is None:
            return None

        # Step 2: optional refinement. Upstream keeps the original on failure.
        if self._enable_refinement:
            refined_raw = await self._ask(
                template.likert_refine_score_3(score_3, self._vulnerability_data)
            )
            if refined_raw is not None:
                refined = parse_refined_example(refined_raw)
                if refined is not None:
                    score_3 = refined

        # Step 3: rewrite the example into the next attack prompt.
        attack_raw = await self._ask(
            template.likert_generate_attack_from_example(
                score_3,
                self._current_attack,
                self._category,
                self._vulnerability_data,
            )
        )
        next_attack = parse_new_attack_prompt(attack_raw) if attack_raw else None
        if next_attack is None:
            return None

        # Step 4: refusal gate. Only an explicit refusal costs the turn.
        refusal_raw = await self._ask(
            template.non_refusal(self._current_attack, next_attack)
        )
        if refusal_raw is not None and parse_refusal(refusal_raw) is True:
            return None

        return next_attack

    async def on_event(self, event: Event) -> EventResponse:
        if isinstance(event, RunStartEvent):
            self._injected = False
            if self._pending is None and not self._exhausted:
                self._pending = await self._prepare_turn()
                if self._pending is None:
                    self._backtracks += 1
                    if self._backtracks >= self._max_backtracks:
                        self._exhausted = True
            return EventResponse(event=event)

        if isinstance(event, ControllablePreCallEvent):
            return self._handle_pre_call(event)

        if isinstance(event, RunEndEvent):
            done = (
                self._exhausted
                or self._turns_sent >= self._num_turns
            )
            return RunEndResponse(event=event, done=done)

        return EventResponse(event=event)

    def _handle_pre_call(
        self, event: ControllablePreCallEvent
    ) -> ControllableInjection | ControllableNoInjection:
        decline = ControllableNoInjection(event=event, controllable=event.controllable)
        if not should_inject(event.controllable, self._preferred, self._surface_roles):
            return decline
        if self._injected or self._pending is None or self._exhausted:
            return decline
        # The escalation is one conversation, so later turns must land on the
        # surface that received the first.
        if self._channel is not None and event.controllable.name != self._channel:
            return decline

        value = self._pending
        self._injected = True
        self._pending = None
        self._current_attack = value
        self._turns_sent += 1
        if self._channel is None:
            self._channel = event.controllable.name
        return ControllableInjection(
            event=event, controllable=event.controllable, value=value
        )

    async def teardown(self) -> None:
        pass
