"""GPTFuzzerOptimizer: superred adapter for the GPTFuzzer jailbreak attack.

Ports the fuzzing loop from GPTFuzzer (Yu et al., 2023) into the
superred event-driven optimizer interface. Each run: select a seed
via MCTS (or other policy), mutate it with an LLM, inject the result,
observe the target response, and update seed pool statistics.

Reference: https://github.com/sherdencooper/GPTFuzz (MIT license)
"""

from __future__ import annotations

import csv
import logging
from importlib import resources
from pathlib import Path

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

from gptfuzzer_optimizer.mutate import QUESTION_PLACEHOLDER, random_mutator
from gptfuzzer_optimizer.selection import (
    MCTSExploreSelectPolicy,
    SeedNode,
    SelectPolicy,
)

logger = logging.getLogger(__name__)

# Default success threshold for the primary score (0-1 scale).
# Scores above this are considered successful jailbreaks.
DEFAULT_SUCCESS_THRESHOLD = 0.5


def _load_bundled_seeds() -> list[str]:
    """Load the bundled GPTFuzzer seed templates from the CSV."""
    seeds_dir = resources.files("gptfuzzer_optimizer") / "seeds"
    csv_path = seeds_dir / "gptfuzzer_seeds.csv"
    seed_templates: list[str] = []
    with resources.as_file(csv_path) as path:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get("text", "").strip()
                if text and QUESTION_PLACEHOLDER in text:
                    seed_templates.append(text)
    return seed_templates


def _load_seeds_from_file(path: str | Path) -> list[str]:
    """Load seed templates from an external CSV file."""
    seed_templates: list[str] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("text", "").strip()
            if text and QUESTION_PLACEHOLDER in text:
                seed_templates.append(text)
    return seed_templates


class GPTFuzzerOptimizer(Optimizer):
    """GPTFuzzer jailbreak optimizer adapted for superred.

    Maintains a pool of jailbreak template seeds. Each run:
    1. Select a seed via MCTS (or other selection policy).
    2. Mutate it using an LLM call.
    3. Insert the goal into the mutated template.
    4. Inject the complete prompt at the controllable.
    5. Observe the target response.
    6. Update seed pool statistics based on the evaluation score.

    Successful mutations (those that achieve jailbreak) are added
    back to the seed pool for future selection, maintaining the
    parent-child tree structure used by MCTS selection.

    The original GPTFuzzer's MutateRandomSinglePolicy with
    concatentate=True prepends the mutation result to the original
    seed. This behavior is controlled by concatenate_mutation.

    Args:
        max_iterations: Maximum number of fuzzing iterations (runs).
            -1 for unlimited (runs until controller's max_runs_per_task).
        energy: Number of mutations to try per selected seed.
            In the superred model (one mutation per run), this controls
            how many consecutive runs use the same selected seed.
        select_policy: Seed selection strategy. Defaults to
            MCTSExploreSelectPolicy (the primary strategy in the original).
        seed_path: Path to a CSV file with seed templates. If None,
            uses the bundled GPTFuzzer seeds.
        initial_seeds: Explicit list of seed template strings. If
            provided, overrides seed_path and bundled seeds.
        success_threshold: Score threshold above which a run is
            considered a successful jailbreak (for seed pool updates).
        concatenate_mutation: If True, prepend mutation result to the
            original seed (matching original GPTFuzzer's MutateRandomSinglePolicy
            with concatentate=True). If False, use the mutation result directly.
        stop_on_success: If True, signal done after first jailbreak.
            If False, continue fuzzing (matching original GPTFuzzer behavior).
            Default is False to match the original.
    """

    def __init__(
        self,
        *,
        max_iterations: int = 500,
        energy: int = 1,
        select_policy: SelectPolicy | None = None,
        seed_path: str | Path | None = None,
        initial_seeds: list[str] | None = None,
        success_threshold: float = DEFAULT_SUCCESS_THRESHOLD,
        concatenate_mutation: bool = True,
        stop_on_success: bool = False,
    ) -> None:
        super().__init__()

        self._max_iterations = max_iterations
        self._energy = energy
        self._seed_path = seed_path
        self._initial_seeds = initial_seeds
        self._success_threshold = success_threshold
        self._concatenate_mutation = concatenate_mutation
        self._stop_on_success = stop_on_success

        # Will be set by select_policy argument or created in initialize()
        self._select_policy_arg = select_policy

        # -- State set during initialize() --
        self._goal: Goal | None = None
        self._controllables: list[Controllable] = []
        self._seed_pool: list[SeedNode] = []
        self._initial_prompt_nodes: list[SeedNode] = []
        self._select_policy: SelectPolicy | None = None

        # -- Per-run state --
        self._current_iteration = 0
        self._current_jailbreak = 0
        self._current_reject = 0
        self._current_query = 0
        self._energy_remaining = 0
        self._current_seed: SeedNode | None = None
        self._current_mutated_template: str | None = None
        self._current_attack_prompt: str | None = None
        self._injected_this_run = False
        self._budget_exhausted = False
        self._mutation_failed = False

    async def initialize(
        self,
        goal: Goal,
        controllables: list[Controllable],
        observables: list[ObservableValue],
        llm_client: LLMClient,
    ) -> None:
        """Set up the optimizer: load seeds, initialize selection policy."""
        await super().initialize(goal, controllables, observables, llm_client)
        self._goal = goal
        self._controllables = list(controllables)

        # Load seed templates
        if self._initial_seeds is not None:
            raw_seeds = list(self._initial_seeds)
        elif self._seed_path is not None:
            raw_seeds = _load_seeds_from_file(self._seed_path)
        else:
            raw_seeds = _load_bundled_seeds()

        if not raw_seeds:
            raise ValueError(
                "No valid seed templates found. Each seed must contain "
                f"'{QUESTION_PLACEHOLDER}'."
            )

        # Build seed pool with PromptNode-style tree structure.
        # Mirrors original: prompt_nodes = [PromptNode(self, prompt) for prompt in initial_seed]
        # Then sets index via property setter (which does NOT trigger parent.child.append
        # for initial seeds since parent=None).
        self._seed_pool = []
        for i, template in enumerate(raw_seeds):
            node = SeedNode(prompt=template, parent=None)
            node.index = i
            self._seed_pool.append(node)

        # Keep a copy of initial nodes (original: self.initial_prompts_nodes = self.prompt_nodes.copy())
        self._initial_prompt_nodes = list(self._seed_pool)

        # Initialize selection policy
        # Original GPTFuzzer defaults vary by usage; MCTSExploreSelectPolicy is the primary strategy.
        if self._select_policy_arg is not None:
            self._select_policy = self._select_policy_arg
        else:
            self._select_policy = MCTSExploreSelectPolicy()
        self._select_policy.set_seed_pool(self._seed_pool, self._initial_prompt_nodes)

        # Reset iteration state
        self._current_iteration = 0
        self._current_jailbreak = 0
        self._current_reject = 0
        self._current_query = 0
        self._energy_remaining = 0
        self._budget_exhausted = False

        logger.info(
            "GPTFuzzerOptimizer initialized with %d seeds, max_iterations=%d, energy=%d",
            len(self._seed_pool),
            self._max_iterations,
            self._energy,
        )

    async def on_event(self, event: Event) -> EventResponse:  # noqa: C901
        """Handle events from the controller.

        Event flow per run:
        1. RunStartEvent: Select seed, mutate, prepare attack prompt.
        2. ControllablePreCallEvent (first): Inject the attack prompt.
        3. ControllablePostCallEvent: Observe target response.
        4. ControllablePreCallEvent (second): No injection (single-turn).
        5. RunEndEvent: Update statistics, decide whether to continue.
        """
        if isinstance(event, RunStartEvent):
            return await self._handle_run_start(event)

        if isinstance(event, ControllablePreCallEvent):
            return self._handle_pre_call(event)

        if isinstance(event, ControllablePostCallEvent):
            return self._handle_post_call(event)

        if isinstance(event, RunEndEvent):
            return self._handle_run_end(event)

        # Default: acknowledge unknown events
        return EventResponse(event=event)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    async def _handle_run_start(self, event: RunStartEvent) -> EventResponse:
        """Select a seed, mutate it, and prepare the attack prompt."""
        self._injected_this_run = False
        self._mutation_failed = False
        self._current_attack_prompt = None
        self._current_mutated_template = None

        assert self._select_policy is not None
        assert self._goal is not None

        # Select a seed (new selection or continue energy budget)
        if self._energy_remaining <= 0:
            self._current_seed = self._select_policy.select()
            self._energy_remaining = self._energy

        self._energy_remaining -= 1
        assert self._current_seed is not None

        # Mutate the selected seed using the LLM
        try:
            mutator = random_mutator()
            all_seed_prompts = [node.prompt for node in self._seed_pool]
            mutated = await mutator.mutate(
                self._current_seed.prompt, all_seed_prompts, self.llm
            )

            if mutated is None:
                # Mutation returned empty -- fall back to the raw seed
                logger.warning("Mutation returned None, using raw seed template")
                mutated = self._current_seed.prompt
                self._mutation_failed = True

            # Original GPTFuzzer concatenation behavior:
            # MutateRandomSinglePolicy with concatentate=True:
            #   results = [result + prompt_node.prompt for result in results]
            # This appends the original seed prompt after the mutation output.
            if self._concatenate_mutation and not self._mutation_failed:
                mutated = mutated + self._current_seed.prompt

            # Store the template (with placeholder) before goal insertion
            self._current_mutated_template = mutated

            # Insert the goal into the template (original: synthesis_message)
            if QUESTION_PLACEHOLDER in mutated:
                attack_prompt = mutated.replace(
                    QUESTION_PLACEHOLDER, self._goal.description
                )
            else:
                # Original synthesis_message returns None if no placeholder.
                # Original evaluate() then sets results=[] (skip).
                # We still inject but note the template is invalid.
                attack_prompt = mutated + "\n\n" + self._goal.description

            self._current_attack_prompt = attack_prompt

        except BudgetExhaustedError:
            logger.info("LLM budget exhausted during mutation")
            self._budget_exhausted = True
            # Fall back to raw seed with goal inserted
            template = self._current_seed.prompt
            self._current_mutated_template = template
            if QUESTION_PLACEHOLDER in template:
                self._current_attack_prompt = template.replace(
                    QUESTION_PLACEHOLDER, self._goal.description
                )
            else:
                self._current_attack_prompt = template + "\n\n" + self._goal.description

        return EventResponse(event=event)

    def _handle_pre_call(self, event: ControllablePreCallEvent) -> EventResponse:
        """Inject the attack prompt into user_message only."""
        if (
            event.controllable.name == "user_message"
            and not self._injected_this_run
            and self._current_attack_prompt is not None
        ):
            self._injected_this_run = True
            return ControllableInjection(
                event=event,
                controllable=event.controllable,
                value=self._current_attack_prompt,
            )
        # Not user_message, already injected, or no prompt.
        return ControllableNoInjection(
            event=event,
            controllable=event.controllable,
        )

    def _handle_post_call(self, event: ControllablePostCallEvent) -> EventResponse:
        """Observe the target's response (stored on trajectory by framework)."""
        return ControllableNoInjection(
            event=event,
            controllable=event.controllable,
        )

    def _handle_run_end(self, event: RunEndEvent) -> EventResponse:
        """Update seed pool statistics and decide whether to continue.

        Mirrors original GPTFuzzer.update():
        - Increment iteration counter.
        - For each result: if jailbreak, add node to pool with parent link.
        - Accumulate jailbreak/reject/query counts.
        - Call select_policy.update() with the prompt nodes.
        """
        self._current_iteration += 1

        # Determine success from evaluation
        jailbreak_success = False
        if event.evaluation is not None:
            jailbreak_success = (
                event.evaluation.success
                or event.evaluation.primary_score.value >= self._success_threshold
            )

        # Create a SeedNode for this run's result (mirrors original PromptNode per result).
        # In the original, results is a list of per-question outcomes.
        # Here we have 1 question, so results is [1] or [0].
        run_node: SeedNode | None = None
        if self._current_seed is not None and not self._mutation_failed:
            run_node = SeedNode(
                prompt=self._current_mutated_template or self._current_seed.prompt,
                parent=self._current_seed,
            )
            run_node.results = [1] if jailbreak_success else [0]

            # Original update(): if num_jailbreak > 0, add to pool with index and parent link.
            if jailbreak_success:
                run_node.index = len(self._seed_pool)
                self._seed_pool.append(run_node)

        # Accumulate counters (mirrors original)
        if run_node is not None:
            self._current_jailbreak += run_node.num_jailbreak
            self._current_query += run_node.num_query
            self._current_reject += run_node.num_reject

        # Update selection policy with the list of prompt nodes from this iteration.
        # Original: self.select_policy.update(prompt_nodes)
        # prompt_nodes is the list of all PromptNodes from this iteration's mutate_single().
        # In superred we have one per run.
        prompt_nodes_for_update = [run_node] if run_node is not None else []
        if prompt_nodes_for_update:
            self._select_policy.update(prompt_nodes_for_update)  # type: ignore[union-attr]

        logger.info(
            "Iteration %d: %d jailbreaks, %d rejects, %d queries",
            self._current_iteration,
            self._current_jailbreak,
            self._current_reject,
            self._current_query,
        )

        # Decide whether to stop (mirrors original is_stop())
        done = self._budget_exhausted
        if (
            self._max_iterations != -1
            and self._current_iteration >= self._max_iterations
        ):
            done = True
        if self._stop_on_success and jailbreak_success:
            done = True

        return RunEndResponse(event=event, done=done)

    async def teardown(self) -> None:
        """Release resources."""
        self._seed_pool.clear()
        self._initial_prompt_nodes.clear()
        self._current_seed = None
        self._current_attack_prompt = None
        self._current_mutated_template = None
