"""BestOfNOptimizer: resample the goal under random augmentations.

Ports the text arm of Best-of-N jailbreaking
(``jplhughes/bon-jailbreaking``, MIT). Each run sends the goal rewritten by a
fresh random augmentation — scrambled word interiors, flipped capitalisation
and +/-1 ASCII shifts — and the search stops as soon as one lands. No LLM is
needed to build the variants.

Paper: Hughes et al., "Best-of-N Jailbreaking" (arXiv:2412.03556).
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

from best_of_n_optimizer.augmentations import DEFAULT_SIGMA, augment
from best_of_n_optimizer.surface_llm import classify_controllables
from best_of_n_optimizer.surfaces import (
    ROLE_CATEGORIES,
    preferred_surface_names,
    should_inject,
)

#: Upstream's ``n_steps`` (4) x ``num_concurrent_k`` (5): the sampling budget
#: its text run uses by default, i.e. the N in Best-of-N.
DEFAULT_N_SAMPLES = 20


class BestOfNOptimizer(Optimizer):
    """Send one randomly augmented restatement of the goal per run.

    Args:
        n_samples: How many augmented variants to try (upstream's
            ``n_steps * num_concurrent_k``).
        sigma: Proportion of characters to augment (upstream default 0.4).
        word_scrambling: Scramble word interiors.
        random_capitalization: Flip letter case.
        ascii_perturbation: Shift characters by +/-1.
        random_prefix_length: Random tokens to prepend (needs the ``tokens``
            extra; upstream default 0).
        random_suffix_length: Random tokens to append (same).
        seed_start: First seed; variant *i* uses ``seed_start + i``.
    """

    def __init__(
        self,
        n_samples: int = DEFAULT_N_SAMPLES,
        sigma: float = DEFAULT_SIGMA,
        word_scrambling: bool = True,
        random_capitalization: bool = True,
        ascii_perturbation: bool = True,
        random_prefix_length: int = 0,
        random_suffix_length: int = 0,
        seed_start: int = 0,
    ) -> None:
        super().__init__()
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples!r}")
        if not 0.0 <= sigma <= 1.0:
            raise ValueError(f"sigma must be in [0, 1], got {sigma!r}")
        if not (word_scrambling or random_capitalization or ascii_perturbation):
            raise ValueError("at least one augmentation must be enabled")
        if random_prefix_length < 0 or random_suffix_length < 0:
            raise ValueError("random prefix/suffix lengths must be >= 0")

        self._n_samples = n_samples
        self._sigma = sigma
        self._word_scrambling = word_scrambling
        self._random_capitalization = random_capitalization
        self._ascii_perturbation = ascii_perturbation
        self._random_prefix_length = random_prefix_length
        self._random_suffix_length = random_suffix_length
        self._seed_start = seed_start

        self._goal: Goal | None = None
        self._index = 0
        self._injected = False
        self._surface_roles: dict[str, str] = {}
        self._preferred: frozenset[str] = frozenset()

    def variant(self, index: int) -> str:
        """The augmented prompt for sample *index* (0-based)."""
        if self._goal is None:
            raise RuntimeError("optimizer not initialized")
        return augment(
            self._goal.description,
            sigma=self._sigma,
            seed=self._seed_start + index,
            word_scrambling=self._word_scrambling,
            random_capitalization=self._random_capitalization,
            ascii_perturbation=self._ascii_perturbation,
            random_prefix_length=self._random_prefix_length,
            random_suffix_length=self._random_suffix_length,
        )

    async def initialize(
        self,
        goal: Goal,
        controllables: list[Controllable],
        observables: list[ObservableValue],
        llm_client: LLMClient,
    ) -> None:
        await super().initialize(goal, controllables, observables, llm_client)
        self._goal = goal
        self._index = 0
        self._injected = False
        # One LLM pass reads each surface's own description and names the
        # user's prompt -- the channel an augmented request has to arrive on.
        # Returns {} on any failure, so the name/value-type backstop decides.
        self._surface_roles = await classify_controllables(
            self.llm, controllables, ROLE_CATEGORIES, goal=goal.description
        )
        self._preferred = preferred_surface_names(controllables, self._surface_roles)

    async def on_event(self, event: Event) -> EventResponse:
        if isinstance(event, RunStartEvent):
            self._injected = False
            return EventResponse(event=event)

        if isinstance(event, ControllablePreCallEvent):
            return self._handle_pre_call(event)

        if isinstance(event, RunEndEvent):
            # Upstream stops as soon as a sample lands; the framework already
            # halts on the claim's success verdict, so this only reports the
            # sampling budget being spent.
            self._index += 1
            return RunEndResponse(event=event, done=self._index >= self._n_samples)

        return EventResponse(event=event)

    def _handle_pre_call(
        self, event: ControllablePreCallEvent
    ) -> ControllableInjection | ControllableNoInjection:
        decline = ControllableNoInjection(event=event, controllable=event.controllable)
        if not should_inject(event.controllable, self._preferred, self._surface_roles):
            return decline
        if self._injected or self._index >= self._n_samples:
            return decline
        self._injected = True
        return ControllableInjection(
            event=event, controllable=event.controllable, value=self.variant(self._index)
        )

    async def teardown(self) -> None:
        pass
