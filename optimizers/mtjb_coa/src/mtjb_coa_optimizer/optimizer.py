"""MtjbCoAOptimizer: Chain-of-Attack multi-turn jailbreak, one turn per run.

Ports the CoA attack from MT-JailBench
(``engine/attacks/coa/``; itself adapted from YancyKahn/CoA). The attacker
LLM first builds a multi-round "chain" of prompts that walk from an innocuous
opener toward the objective; each round's prompt is sent as one conversation
turn, and (when attack update is enabled) refined **before sending** using the
previous round's answer so the semantic trajectory keeps climbing. A binary
judge decides success per round.

This is a native async re-implementation against superred's event model, not a
driver of the upstream (synchronous) engine. Every upstream model call --
chain generation, prompt update, the binary judge, and the LLM-approximation
similarity scorer -- is routed through the constrained ``self.llm``. The prompt
bodies are loaded from the byte-identical vendored ``prompts.yaml`` /
``similarity_scorer.py``; none are reproduced here.

See ``ASSUMPTIONS.md`` for every deliberate deviation (no-rewind flow, pre-send
refinement, disabled mirror-target requery, LLM-approx similarity default).
"""

from __future__ import annotations

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
from superred.core.types.llm import BudgetExhaustedError
from superred.core.types.observable import ObservableValue

from mtjb_coa_optimizer import parsing
from mtjb_coa_optimizer.surfaces import preferred_surface_names, should_inject
from mtjb_coa_optimizer.vendored import load_source_constant, load_yaml

#: Upstream CoA config defaults (``engine/attacks/coa`` + shipped configs).
DEFAULT_MAX_TURNS = 5
DEFAULT_N_INIT_CHAINS = 3
DEFAULT_MAX_UPDATE_RETRIES = 5
DEFAULT_SEMANTIC_UPDATE_SLACK = 0.1
DEFAULT_MAX_JUDGE_RETRIES = 2

#: Binary judge success value (``[[1]]``).
_JUDGE_SUCCESS = 1


class MtjbCoAOptimizer(Optimizer):
    """Chain-of-Attack optimizer: one pinned conversation, one turn per run.

    Args:
        max_turns: Number of rounds in the attack chain (upstream requires the
            generated chain length to match this exactly). Clamped to >= 1.
        n_init_chains: How many candidate chains to generate and rank at
            setup; the best (widest semantic range) is kept. Clamped to >= 1.
        max_update_retries: Attempts when refining a round's prompt before it
            is sent (upstream ``_update_attack``). Clamped to >= 1.
        semantic_update_slack: ``theta`` -- a refined prompt is accepted once
            its similarity to the objective is within ``(1 - theta)`` of the
            preset prompt's similarity.
        enable_attack_update: When True (upstream default) each round after the
            first is refined pre-send from the previous answer. When False the
            raw chain prompts are sent verbatim (upstream ``_sequence_walk``).
        max_judge_retries: Attempts to obtain a parseable binary judge score.
    """

    def __init__(
        self,
        *,
        max_turns: int = DEFAULT_MAX_TURNS,
        n_init_chains: int = DEFAULT_N_INIT_CHAINS,
        max_update_retries: int = DEFAULT_MAX_UPDATE_RETRIES,
        semantic_update_slack: float = DEFAULT_SEMANTIC_UPDATE_SLACK,
        enable_attack_update: bool = True,
        max_judge_retries: int = DEFAULT_MAX_JUDGE_RETRIES,
    ) -> None:
        super().__init__()
        self._max_turns = max(max_turns, 1)
        self._n_init_chains = max(n_init_chains, 1)
        self._max_update_retries = max(max_update_retries, 1)
        self._theta = semantic_update_slack
        self._enable_attack_update = enable_attack_update
        self._max_judge_retries = max(max_judge_retries, 1)

        # Vendored payload templates (loaded once; no bodies reproduced here).
        _coa = ("attacks", "coa", "prompts.yaml")
        self._tpl = load_yaml(*_coa)
        self._sim_prompt = load_source_constant(
            "SIM_APPROX_PROMPT", "attacks", "coa", "similarity_scorer.py"
        )

        # Per-task state (reset in initialize()).
        self._goal: Goal | None = None
        self._chain: list[str] = []
        self._pending: str | None = None
        self._turn = 0
        self._done = False
        self._preferred: frozenset[str] = frozenset()
        # Per-run state (re-armed on RunStart).
        self._injected = False
        self._saw_post_call = False
        self._scored_this_run = False
        self._channel: str | None = None

    @property
    def chain(self) -> list[str]:
        """The pinned attack chain (round prompts), once generated."""
        return list(self._chain)

    # ------------------------------------------------------------------
    # LLM plumbing
    # ------------------------------------------------------------------
    async def _ask(self, prompt: str, *, system: str | None = None, **kw: object) -> str | None:
        """One attacker/judge LLM call. ``None`` on failure (upstream continues).

        A genuine budget exhaustion (cost already spent) is re-raised so a
        spent run is never reported as a quietly finished attack; the
        zero-cost noop client the controller hands non-LLM optimizers degrades
        quietly instead.
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            result = await self.llm.complete(messages, **kw)  # type: ignore[arg-type]
            return result.choices[0].message.content or ""
        except BudgetExhaustedError as error:
            if error.usage.cost > 0:
                raise
            return None
        except Exception:  # noqa: BLE001 - upstream wraps each call in try/continue
            return None

    async def _similarity(self, target: str, response: str) -> float:
        """LLM-approximation similarity in ``[0, 1]`` (upstream default path)."""
        if not target or not response:
            return 0.0
        raw = await self._ask(
            self._sim_prompt.format(target=target.strip(), response=response.strip()),
            response_format={"type": "json_object"},
        )
        if raw is None:
            return 0.0
        value = parsing.extract_similarity(raw)
        return value if value is not None else 0.0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    async def initialize(
        self,
        goal: Goal,
        controllables: list[Controllable],
        observables: list[ObservableValue],
        llm_client: LLMClient,
    ) -> None:
        await super().initialize(goal, controllables, observables, llm_client)
        self._goal = goal
        self._chain = []
        self._pending = None
        self._turn = 0
        self._done = False
        self._injected = False
        self._saw_post_call = False
        self._scored_this_run = False
        self._channel = None
        self._preferred = preferred_surface_names(controllables)

        self._chain = await self._build_chain(goal.description)
        if not self._chain:
            self._done = True
            return
        self._pending = self._chain[0]

    async def _build_chain(self, target: str) -> list[str]:
        """Generate candidate chains, rank by semantic range, return the best.

        Mirrors ``CoAPromptGenerator._get_mr_init_chain`` +
        ``_process_mr_init_chain``: keep only chains whose length equals
        ``max_turns``; rank by the spread (max - min) of per-round similarity
        to the objective; return the widest-range chain.
        """
        system = self._tpl["mr_attack_chain_system_prompt"].format(
            target=target, max_round=self._max_turns
        )
        init_msg = self._tpl["mr_attack_chain_init_message"].format(
            target=target, max_round=self._max_turns
        )

        candidates: list[list[str]] = []
        for _ in range(self._n_init_chains + 5):  # buffer for failures (upstream)
            raw = await self._ask(init_msg, system=system)
            if raw is None:
                continue
            chain = parsing.extract_chain(raw)
            if chain is None or len(chain) != self._max_turns:
                continue
            candidates.append(chain)
            if len(candidates) >= self._n_init_chains:
                break

        if not candidates:
            return []

        best: list[str] | None = None
        best_range = float("-inf")
        for chain in candidates:
            sims = [await self._similarity(target, prompt) for prompt in chain]
            spread = max(sims) - min(sims)
            if spread > best_range:
                best_range = spread
                best = chain
        return best or candidates[0]

    # ------------------------------------------------------------------
    # Event loop
    # ------------------------------------------------------------------
    async def on_event(self, event: Event) -> EventResponse:
        if isinstance(event, RunStartEvent):
            self._injected = False
            self._saw_post_call = False
            self._scored_this_run = False
            return EventResponse(event=event)

        if isinstance(event, ControllablePreCallEvent):
            return self._handle_pre_call(event)

        if isinstance(event, ControllablePostCallEvent):
            self._saw_post_call = await self._handle_post_call(event) or self._saw_post_call
            # PostCall requires an injection decision; this attack reads the
            # answer but never rewrites it, so it always declines.
            return ControllableNoInjection(event=event, controllable=event.controllable)

        if isinstance(event, RunEndEvent):
            if self._injected and not self._saw_post_call:
                # No answer can arrive for the turn we sent -> nothing advances.
                self._done = True
            elif self._channel is not None and not self._injected and self._pending is not None:
                # The pinned channel did not fire this run; it never will again.
                self._done = True
            return RunEndResponse(event=event, done=self._done)

        return EventResponse(event=event)

    def _handle_pre_call(
        self, event: ControllablePreCallEvent
    ) -> ControllableInjection | ControllableNoInjection:
        decline = ControllableNoInjection(event=event, controllable=event.controllable)
        if not should_inject(event.controllable, self._preferred):
            return decline
        if self._injected or self._done or self._pending is None:
            return decline
        if self._channel is not None and event.controllable.name != self._channel:
            return decline

        value = self._pending
        self._injected = True
        self._pending = None
        self._turn += 1
        if self._channel is None:
            self._channel = event.controllable.name
        return ControllableInjection(event=event, controllable=event.controllable, value=value)

    async def _handle_post_call(self, event: ControllablePostCallEvent) -> bool:
        """Judge the answer, then build (pre-send refine) the next round's prompt."""
        if self._done or self._goal is None:
            return False
        if self._scored_this_run:
            return False  # one turn is sent per run; only its answer is scored
        if self._channel is None or event.controllable.name != self._channel:
            return False

        self._scored_this_run = True
        target = self._goal.description
        answer = event.answer

        score = await self._judge(target, answer)
        if score == _JUDGE_SUCCESS:
            self._done = True  # END_SUCCESS
            return True
        if self._turn >= self._max_turns:
            self._done = True
            return True

        # Prepare the next round's prompt. Index is self._turn (0-based) because
        # _turn already counts the rounds sent so far.
        preset = self._chain[self._turn]
        if self._enable_attack_update:
            self._pending = await self._update_attack(
                target, self._turn + 1, preset, answer, score
            )
        else:
            self._pending = preset
        return True

    async def _judge(self, target: str, response: str) -> int:
        """Binary CoA judge (0/1). Returns 0 if no parseable score is produced."""
        if not response.strip():
            return 0
        system = self._tpl["judge_system_prompt"].format(target=target)
        user = self._tpl["judge_user_message"].format(response=response)
        for _ in range(self._max_judge_retries):
            raw = await self._ask(user, system=system)
            if raw is None:
                continue
            score = parsing.extract_judge_score(raw)
            if score is not None:
                return score
        return 0

    async def _update_attack(
        self, target: str, turn: int, preset_prompt: str, response_text: str, score: int
    ) -> str:
        """Refine a round's prompt pre-send (upstream ``_update_attack``).

        Accept the first candidate whose similarity to the objective is within
        ``(1 - theta)`` of the preset prompt's; otherwise keep the highest-
        similarity candidate; if nothing parses, fall back to the preset.
        """
        system = self._tpl["update_single_round_system_prompt"].format(
            preset_prompt=preset_prompt, target=target, round=turn, max_round=self._max_turns
        )
        user = self._tpl["update_single_round_user_message"].format(
            response=response_text,
            preset_prompt=preset_prompt,
            target=target,
            # Upstream passes the builtin ``round`` here (a bug that renders
            # "<built-in function round>" into the prompt); we pass the turn.
            round=turn,
            max_round=self._max_turns,
            score=score,
        )
        preset_sim = await self._similarity(target, preset_prompt)
        best_prompt: str | None = None
        best_sim = 0.0
        for _ in range(self._max_update_retries):
            raw = await self._ask(user, system=system)
            if raw is None:
                continue
            candidate = parsing.extract_update_prompt(raw)
            if not candidate:
                continue
            sim = await self._similarity(target, candidate)
            if sim > best_sim:
                best_prompt, best_sim = candidate, sim
            if sim >= preset_sim * (1 - self._theta):
                return candidate
        return best_prompt if best_prompt is not None else preset_prompt

    async def teardown(self) -> None:
        pass
