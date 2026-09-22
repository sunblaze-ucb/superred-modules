"""ReNeLLMOptimizer: generalized nested-jailbreak attack for superred.

Ports ReNeLLM (``NJUNLP/ReNeLLM`` @ ``a61c39e``; Ding et al., "A Wolf in
Sheep's Clothing", NAACL 2024, arXiv:2311.08268). ReNeLLM generalizes prompt
jailbreaks into two stages driven by an auxiliary LLM: **prompt rewriting** (a
random subset/order of six semantics-preserving rewrite operations, retried
until a binary judge confirms the rewrite is still harmful) and **scenario
nesting** (embedding the rewritten goal into one of three benign carriers --
code completion, table filling, or text continuation). The nested prompt is
sent to the model under attack and its reply is scored by the same judge; the
process repeats up to a budget until the reply is judged harmful.

Event-model mapping (see ``ASSUMPTIONS.md`` for every deviation):

* One upstream outer iteration == one superred *run*. On ``RunStart`` the
  per-run state is re-armed and the rewritten+nested prompt is produced
  **pre-send** (the rewrite/judge loop runs before anything is injected -- there
  is no mid-conversation rewind in superred).
* The nested prompt is injected on the first eligible free-text user surface at
  ``ControllablePreCall``; the model under attack is the target itself, not an
  LLM this optimizer constructs.
* The reply is read from ``ControllablePostCall`` (with a trajectory-observable
  fallback) and scored by the vendored judge via ``self.llm`` at ``RunEnd`` to
  decide stop/continue.

Every auxiliary LLM call (the six rewrite operations and the judge) runs the
byte-identical vendored upstream code, routed to the constrained ``self.llm``
through :mod:`renellm_optimizer._shim`. No prompt/scenario/judge body is
reproduced in this package.
"""

from __future__ import annotations

import asyncio
import logging
import random
from types import SimpleNamespace

from superred.core.interfaces.optimizer import Optimizer
from superred.core.llm import LLMClient
from superred.core.types.controllable import Controllable
from superred.core.types.event import Event, EventResponse
from superred.core.types.events import (
    ControllableInjection,
    ControllableNoInjection,
    ControllablePostCallEvent,
    ControllablePreCallEvent,
    ObservableEvent,
    RunEndEvent,
    RunEndResponse,
    RunStartEvent,
)
from superred.core.types.goal import Goal
from superred.core.types.llm import BudgetExhaustedError
from superred.core.types.observable import ObservableValue

from renellm_optimizer import _shim, vendored
from renellm_optimizer.surfaces import is_eligible_surface, preferred_surface_names

logger = logging.getLogger(__name__)

#: Upstream defaults (``renellm.py``): max outer iterations per behaviour.
DEFAULT_ITER_MAX = 20
#: Safety cap on the (upstream-unbounded) rewrite-retry loop; on exhaustion the
#: last rewritten candidate is used. Prevents an infinite loop when the judge
#: never confirms harmfulness. Documented in ASSUMPTIONS.md.
DEFAULT_MAX_REWRITE_ATTEMPTS = 20
#: Number of rewrite operations available upstream (``random.sample(range(6))``).
_N_OPERATIONS = 6

#: Default observable names scanned for the target reply when no
#: ``ControllablePostCall`` answer is seen.
_DEFAULT_RESPONSE_OBSERVABLE_NAMES = frozenset(
    {"response", "model_response", "assistant_response", "reply", "output"}
)
_RESPONSE_NAME_HINTS = ("response", "assistant", "reply", "output", "completion")

# Field names for the upstream ``args`` namespace the vendored helpers read.
# Their values are inert: the LLM shim ignores every model/credential/sleep and
# sampling knob. ``temperature`` appears here only because upstream
# ``harmful_classification`` reads ``args.temperature`` and passes it positionally
# to the shim (which drops it) -- listing it as data keeps every LLM-call site in
# this package free of a temperature keyword/assignment (see
# ``tests/test_no_temperature.py``).
_ARGS_FIELDS: tuple[str, ...] = (
    "rewrite_model",
    "judge_model",
    "retry_times",
    "round_sleep",
    "fail_sleep",
    "gpt_api_key",
    "gpt_base_url",
    "temperature",
)


def _make_args() -> SimpleNamespace:
    """Build the inert ``args`` namespace consumed by the vendored helpers."""
    args = SimpleNamespace()
    for field_name in _ARGS_FIELDS:
        setattr(args, field_name, None)
    return args


class ReNeLLMOptimizer(Optimizer):
    """ReNeLLM generalized nested-jailbreak optimizer (one nested prompt per run).

    Args:
        iter_max: Maximum number of runs (outer iterations) before the optimizer
            reports itself done. Clamped to >= 1. Upstream default is 20.
        max_rewrite_attempts: Maximum rewrite-retry rounds while producing one
            nested prompt (upstream retries unboundedly until the judge confirms
            the rewrite is still harmful). Clamped to >= 1. On exhaustion the
            last candidate is used.
        seed: Optional seed for the operation-count/order and scenario choice, so
            a run is reproducible. The vendored ``shortenSentence`` candidate
            pick uses the global ``random`` module (upstream behaviour).
        target_controllable_name: If set, inject only into the surface with this
            exact name; otherwise the first eligible free-text user surface is
            pinned for the run.
        response_observable_names: Observable names treated as the target reply
            when no post-call answer is available.
    """

    def __init__(
        self,
        *,
        iter_max: int = DEFAULT_ITER_MAX,
        max_rewrite_attempts: int = DEFAULT_MAX_REWRITE_ATTEMPTS,
        seed: int | None = None,
        target_controllable_name: str | None = None,
        response_observable_names: frozenset[str] | None = None,
    ) -> None:
        super().__init__()
        self._iter_max = max(iter_max, 1)
        self._max_rewrite_attempts = max(max_rewrite_attempts, 1)
        self._seed = seed
        self._target_controllable_name = target_controllable_name
        self._response_observable_names = (
            response_observable_names
            if response_observable_names is not None
            else _DEFAULT_RESPONSE_OBSERVABLE_NAMES
        )

        # Per-task state (set in initialize()).
        self._goal: Goal | None = None
        self._vendored: vendored.Vendored | None = None
        self._args: SimpleNamespace = _make_args()
        self._preferred: frozenset[str] = frozenset()
        self._rng = random.Random(seed)
        self._run_count = 0
        self._succeeded = False

        # Per-run state (re-armed on RunStart).
        self._pending: str | None = None
        self._injected = False
        self._channel: str | None = None
        self._pending_answer: str | None = None

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
        self._vendored = vendored.load()
        self._args = _make_args()
        self._preferred = preferred_surface_names(controllables)
        self._rng = random.Random(self._seed)
        self._run_count = 0
        self._succeeded = False
        self._reset_run_state()

    async def teardown(self) -> None:
        return None

    # ------------------------------------------------------------------
    # LLM bridge (sync vendored code -> async self.llm)
    # ------------------------------------------------------------------
    def _bridge(self, loop: asyncio.AbstractEventLoop):
        """Return a synchronous callable that runs one ``self.llm`` completion.

        Called from the ``asyncio.to_thread`` worker where the synchronous
        vendored helpers run; it hops back to the event loop, mirroring
        upstream ``chatCompletion``'s ``.strip()`` on the returned content.
        """

        def call(messages: list[dict[str, str]]) -> str:
            future = asyncio.run_coroutine_threadsafe(self.llm.complete(messages), loop)
            response = future.result()
            content = response.choices[0].message.content
            return (content or "").strip()

        return call

    async def _run_vendored(self, func, *args):
        """Run a synchronous vendored helper with the LLM bridge armed."""
        loop = asyncio.get_running_loop()
        token = _shim.BRIDGE.set(self._bridge(loop))
        try:
            return await asyncio.to_thread(func, *args)
        finally:
            _shim.BRIDGE.reset(token)

    # ------------------------------------------------------------------
    # Pre-send: rewrite + nest
    # ------------------------------------------------------------------
    async def _produce_nested(self, goal_text: str) -> str | None:
        """Produce one rewritten+nested prompt (pre-send). ``None`` on failure."""
        try:
            return await self._run_vendored(self._produce_nested_sync, goal_text)
        except BudgetExhaustedError as error:
            # A genuinely spent run must not be reported as a quietly finished
            # attack; the zero-cost noop client handed to non-LLM optimizers
            # degrades quietly instead.
            if error.usage.cost > 0:
                raise
            return None
        except Exception:  # noqa: BLE001 - a transient aux-LLM failure ends this run, not the sweep
            logger.warning("renellm: nested-prompt production failed", exc_info=True)
            return None

    def _produce_nested_sync(self, goal_text: str) -> str:
        """Upstream rewrite-until-harmful loop + scenario nesting (in a thread).

        Mirrors ``renellm.py``: pick a random count/order of the six rewrite
        operations, apply them to the original goal, and keep the result only if
        the judge still labels it harmful; otherwise retry from the original
        goal. Then nest the accepted rewrite into a random scenario.
        """
        assert self._vendored is not None
        operations = self._vendored.operations
        classify = self._vendored.harmful_classification

        work = goal_text
        accepted: str | None = None
        for _ in range(self._max_rewrite_attempts):
            count = self._rng.randint(1, _N_OPERATIONS)
            order = self._rng.sample(range(_N_OPERATIONS), count)
            work = goal_text
            for index in order:
                work = operations[index](self._args, work)
            if classify(self._args, work) == "1":
                accepted = work
                break
        if accepted is None:
            accepted = work  # cap reached: use the last candidate (see ASSUMPTIONS)

        scenario = self._rng.choice(self._vendored.scenarios)
        return scenario.replace("<>", accepted)

    # ------------------------------------------------------------------
    # Post-send: judge the reply
    # ------------------------------------------------------------------
    @staticmethod
    def _is_candidate_reply(answer: str) -> bool:
        """Upstream refusal short-circuit: skip judging obvious refusals.

        Reproduces ``renellm.py`` verbatim, including its asymmetric casing
        (``"apologize"`` matched as-is, ``"sorry"`` case-insensitively).
        """
        return "apologize" not in answer and "sorry" not in answer.lower()

    async def _judge_reply(self, answer: str) -> bool:
        """Score the target reply with the vendored judge via ``self.llm``."""
        assert self._vendored is not None
        try:
            label = await self._run_vendored(
                self._vendored.harmful_classification, self._args, answer
            )
        except BudgetExhaustedError as error:
            if error.usage.cost > 0:
                raise
            return False
        except Exception:  # noqa: BLE001 - a judge failure is a non-success round, not a crash
            logger.warning("renellm: reply judge failed", exc_info=True)
            return False
        return label == "1"

    # ------------------------------------------------------------------
    # Event loop
    # ------------------------------------------------------------------
    async def on_event(self, event: Event) -> EventResponse:
        if isinstance(event, RunStartEvent):
            return await self._handle_run_start(event)
        if isinstance(event, ControllablePreCallEvent):
            return self._handle_pre_call(event)
        if isinstance(event, ControllablePostCallEvent):
            return self._handle_post_call(event)
        if isinstance(event, RunEndEvent):
            return await self._handle_run_end(event)
        return EventResponse(event=event)

    async def _handle_run_start(self, event: RunStartEvent) -> EventResponse:
        self._reset_run_state()
        if self._goal is not None and not self._succeeded:
            self._pending = await self._produce_nested(self._goal.description)
        return EventResponse(event=event)

    def _handle_pre_call(
        self, event: ControllablePreCallEvent
    ) -> ControllableInjection | ControllableNoInjection:
        ctrl = event.controllable
        decline = ControllableNoInjection(event=event, controllable=ctrl)
        if self._succeeded or self._pending is None or self._injected:
            return decline
        if not self._eligible(ctrl):
            return decline
        if self._channel is not None and ctrl.name != self._channel:
            return decline

        self._injected = True
        if self._channel is None:
            self._channel = ctrl.name
        return ControllableInjection(event=event, controllable=ctrl, value=self._pending)

    def _handle_post_call(self, event: ControllablePostCallEvent) -> ControllableNoInjection:
        # PostCall requires an injection decision; this attack reads the reply
        # but never rewrites it, so it always declines.
        if self._injected and event.controllable.name == self._channel:
            self._pending_answer = event.answer
        return ControllableNoInjection(event=event, controllable=event.controllable)

    async def _handle_run_end(self, event: RunEndEvent) -> RunEndResponse:
        self._run_count += 1
        answer = self._pending_answer
        if answer is None:
            answer = self._recover_reply_from_trajectory()

        success = False
        if answer is not None and self._is_candidate_reply(answer):
            success = await self._judge_reply(answer)
        # A framework success verdict is authoritative and ends the attack.
        if event.evaluation is not None and event.evaluation.success:
            success = True

        if success:
            self._succeeded = True
            return RunEndResponse(event=event, done=True)
        if not self._injected:
            # No eligible surface fired (or no payload) -> nothing can advance.
            return RunEndResponse(event=event, done=True)
        if self._run_count >= self._iter_max:
            return RunEndResponse(event=event, done=True)
        return RunEndResponse(event=event, done=False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _eligible(self, controllable: Controllable) -> bool:
        if self._target_controllable_name is not None:
            return controllable.name == self._target_controllable_name
        return is_eligible_surface(controllable)

    def _recover_reply_from_trajectory(self) -> str | None:
        """Fallback: the last matching response observable on this run."""
        if self.current_trajectory is None:
            return None
        recovered: str | None = None
        for item in self.current_trajectory.drain():
            if not isinstance(item, ObservableEvent):
                continue
            name = item.observable.name
            normalized = name.lower()
            matches = (
                name in self._response_observable_names
                or normalized in self._response_observable_names
                or any(hint in normalized for hint in _RESPONSE_NAME_HINTS)
            )
            if matches and isinstance(item.content, str):
                recovered = item.content
        return recovered

    def _reset_run_state(self) -> None:
        self._pending = None
        self._injected = False
        self._channel = None
        self._pending_answer = None
