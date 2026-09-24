"""AnecdoctorOptimizer: knowledge-graph / few-shot misinformation elicitation.

Ports PyRIT's ``AnecdoctorGenerator`` (``pyrit/executor/promptgen/anecdoctor.py``,
MIT). Anecdoctor coaxes a target into producing misinformation-style content by
framing a request around example "claims", in one of two modes:

* **few-shot** (default): the claims are formatted into a few-shot block and
  wrapped in the framing template.
* **knowledge-graph**: the attacker LLM first builds a text "knowledge graph"
  from the claims, and that graph -- rather than the raw claims -- is wrapped in
  the framing template. Upstream calls the KG the more coherent, targeted input.

Upstream sets the framing as the target's *system* prompt and sends the
examples / KG as a separate user message. This port targets a single free-text
user surface, so it folds the framing and the body into one injected turn and
never writes a system prompt (see ASSUMPTIONS.md). The SuperRed Goal maps onto
upstream's ``evaluation_data`` as the lead claim. The knowledge-graph build is
the only model call and is routed through the constrained ``self.llm``; the
framing/KG-build prompt bodies are loaded from the byte-identically vendored
PyRIT YAML templates and are never reproduced here.

This is a misinformation-elicitation technique (an "A1" content-generation
attacker), not a classic refusal-bypass jailbreak. See ``ASSUMPTIONS.md``.
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

from anecdoctor_optimizer import claims as claims_mod
from anecdoctor_optimizer import vendored
from anecdoctor_optimizer.surfaces import preferred_surface_names, should_inject

#: Upstream ``AnecdoctorContext`` defaults are supplied by the caller; these are
#: neutral format/language defaults for this port.
DEFAULT_LANGUAGE = "english"
DEFAULT_CONTENT_TYPE = "social media post"
DEFAULT_MAX_KG_RETRIES = 3


class AnecdoctorOptimizer(Optimizer):
    """Emit a single Anecdoctor framed request, once per run.

    Args:
        use_knowledge_graph: When True, the attacker LLM builds a knowledge
            graph from the claims at setup and the framed request is built
            around it (upstream's ``processing_model`` mode). When False
            (default) the claims are used directly as few-shot examples.
        language: Language of the content to elicit (upstream ``language``).
        content_type: Format of the content to elicit, e.g. "news article"
            (upstream ``type``). A neutral format descriptor, not a payload.
        example_claims: Few-shot example claims in ClaimsReview style. Defaults
            to the authored synthetic :data:`~anecdoctor_optimizer.claims.
            DEFAULT_EXAMPLE_CLAIMS` (never copied harmful payloads).
        include_goal_as_claim: When True (default) the SuperRed Goal is folded
            in as the lead claim so the specific claim/topic drives generation.
        max_kg_retries: Attempts to obtain a non-empty knowledge graph from the
            attacker LLM before falling back to the few-shot framing. Clamped
            to >= 1.
    """

    def __init__(
        self,
        *,
        use_knowledge_graph: bool = False,
        language: str = DEFAULT_LANGUAGE,
        content_type: str = DEFAULT_CONTENT_TYPE,
        example_claims: list[str] | None = None,
        include_goal_as_claim: bool = True,
        max_kg_retries: int = DEFAULT_MAX_KG_RETRIES,
    ) -> None:
        super().__init__()
        if not language.strip():
            raise ValueError("language must be a non-empty string")
        if not content_type.strip():
            raise ValueError("content_type must be a non-empty string")
        self._use_kg = use_knowledge_graph
        self._language = language
        self._content_type = content_type
        self._example_claims: list[str] = (
            list(example_claims)
            if example_claims is not None
            else list(claims_mod.DEFAULT_EXAMPLE_CLAIMS)
        )
        self._include_goal_as_claim = include_goal_as_claim
        self._max_kg_retries = max(max_kg_retries, 1)

        # Per-task state (set in initialize()).
        self._goal: Goal | None = None
        self._payload = ""
        self._kg_built = False
        self._preferred: frozenset[str] = frozenset()
        # Per-run state (re-armed on RunStart).
        self._injected = False

    @property
    def payload(self) -> str:
        """The assembled framed request that will be injected."""
        return self._payload

    @property
    def knowledge_graph_built(self) -> bool:
        """Whether knowledge-graph mode ran and produced a graph this task.

        False in few-shot mode and when a requested KG build failed and the
        optimizer fell back to few-shot framing.
        """
        return self._kg_built

    # ------------------------------------------------------------------
    # LLM plumbing
    # ------------------------------------------------------------------
    async def _ask(self, prompt: str, *, system: str) -> str | None:
        """One attacker-LLM call for the knowledge-graph build. ``None`` on failure.

        A genuine budget exhaustion (cost already spent) is re-raised so a spent
        run is never reported as a quietly finished attack; the zero-cost noop
        client the controller hands non-LLM optimizers degrades quietly instead
        (matching the shared single-turn pattern).
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        try:
            result = await self.llm.complete(messages)
            return result.choices[0].message.content or ""
        except BudgetExhaustedError as error:
            if error.usage.cost > 0:
                raise
            return None
        except Exception:  # noqa: BLE001 - upstream degrades on any model failure
            return None

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
        self._injected = False
        self._kg_built = False
        self._preferred = preferred_surface_names(controllables)

        evaluation_data = claims_mod.build_evaluation_data(
            goal=goal.description,
            example_claims=self._example_claims,
            include_goal_as_claim=self._include_goal_as_claim,
        )
        # Guarantee a non-empty few-shot body even if the Goal is folded out and
        # no examples were supplied (upstream requires non-empty evaluation_data).
        if not evaluation_data:
            evaluation_data = list(claims_mod.DEFAULT_EXAMPLE_CLAIMS)

        self._payload = await self._build_payload(evaluation_data)

    async def _build_payload(self, evaluation_data: list[str]) -> str:
        """Assemble the framed request (KG mode, or few-shot fallback/default)."""
        if self._use_kg:
            kg_text = await self._build_knowledge_graph(evaluation_data)
            if kg_text:
                framing = vendored.load_prompt(vendored.USE_KG_YAML).format(
                    language=self._language, type=self._content_type
                )
                self._kg_built = True
                return claims_mod.assemble_framed_request(framing=framing, body=kg_text)
            # KG could not be built (no attacker LLM / empty response): degrade
            # to the few-shot framing so the technique still emits a request.
        framing = vendored.load_prompt(vendored.USE_FEWSHOT_YAML).format(
            language=self._language, type=self._content_type
        )
        body = claims_mod.format_few_shot_examples(evaluation_data)
        return claims_mod.assemble_framed_request(framing=framing, body=body)

    async def _build_knowledge_graph(self, evaluation_data: list[str]) -> str | None:
        """Ask the attacker LLM to build a knowledge graph from the claims.

        Mirrors upstream ``_extract_knowledge_graph_async``: the KG-build prompt
        (parameterised by ``language``) is the system prompt and the few-shot
        claim block is the user message. Returns the KG text, or ``None`` if no
        non-empty graph could be obtained.
        """
        system = vendored.load_prompt(vendored.BUILD_KG_YAML).format(language=self._language)
        user = claims_mod.format_few_shot_examples(evaluation_data)
        for _ in range(self._max_kg_retries):
            raw = await self._ask(user, system=system)
            if raw and raw.strip():
                return raw.strip()
        return None

    # ------------------------------------------------------------------
    # Event loop
    # ------------------------------------------------------------------
    async def on_event(self, event: Event) -> EventResponse:
        if isinstance(event, RunStartEvent):
            # Re-arm per-run state so a re-run injects the framed request again.
            self._injected = False
            return EventResponse(event=event)

        if isinstance(event, ControllablePreCallEvent):
            return self._handle_pre_call(event)

        if isinstance(event, ControllablePostCallEvent):
            # Post-call events also require an injection decision: the channel
            # type-checks the response, so a bare ``EventResponse`` raises and
            # aborts the task on any target that emits them. Anecdoctor reads no
            # answer and never rewrites it, so it declines.
            return ControllableNoInjection(event=event, controllable=event.controllable)

        if isinstance(event, RunEndEvent):
            # One deterministic framed request per task (upstream generates
            # once). RunStart still re-arms state should the controller re-run.
            return RunEndResponse(event=event, done=True)

        return EventResponse(event=event)

    def _handle_pre_call(
        self, event: ControllablePreCallEvent
    ) -> ControllableInjection | ControllableNoInjection:
        decline = ControllableNoInjection(event=event, controllable=event.controllable)
        if not should_inject(event.controllable, self._preferred):
            return decline
        if self._injected or not self._payload:
            return decline
        self._injected = True
        return ControllableInjection(
            event=event, controllable=event.controllable, value=self._payload
        )

    async def teardown(self) -> None:
        pass
