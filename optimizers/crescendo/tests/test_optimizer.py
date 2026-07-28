"""Tests for the CrescendoOptimizer event-driven state machine."""

from unittest.mock import AsyncMock, patch, MagicMock
import pytest

from superred.core.interfaces.target import Target
from superred.core.types.controllable import Controllable
from superred.core.types.event import EventResponse
from superred.core.types.events import (
    ControllableInjection,
    ControllableNoInjection,
    ControllablePreCallEvent,
    ControllablePostCallEvent,
    ObservableEvent,
    RunEndEvent,
    RunStartEvent,
)
from superred.core.types.goal import Goal
from superred.core.types.llm import BudgetExhaustedError, LLMUsage
from superred.core.types.observable import Observable, ObservableValue
from superred.core.types.security_domain import SecurityDomain, SecurityDomainTag
from superred.core.types.state import ConfigSpec, QuerySpec

from crescendo_optimizer import retry
from crescendo_optimizer.attacker import AttackerOutput
from crescendo_optimizer.optimizer import CrescendoOptimizer
from crescendo_optimizer.retry import DEFAULT_ATTEMPTS, HelperLLMUnavailableError


class _NoSleepAsyncio:
    """Stand-in for the ``asyncio`` module inside ``retry``: skips backoff."""

    @staticmethod
    async def sleep(_delay: float) -> None:
        return None


def _make_controllable(name: str = "input", tag: str = "user"):
    return Controllable(name=name, security_domain=SecurityDomainTag(name=tag))


def _make_run_start():
    traj = MagicMock()
    return RunStartEvent(trajectory=traj)


def _make_run_end():
    return RunEndEvent(evaluation=None)


def _make_pre_call(ctrl):
    return ControllablePreCallEvent(controllable=ctrl, request="user input")


def _make_post_call(ctrl, answer):
    return ControllablePostCallEvent(controllable=ctrl, request="user input", answer=answer)


def _make_pre_call_with_request(ctrl, request):
    return ControllablePreCallEvent(controllable=ctrl, request=request)


def _make_post_call_with_request(ctrl, answer, request):
    return ControllablePostCallEvent(controllable=ctrl, request=request, answer=answer)


class _FakeReadableTrajectory:
    def __init__(self) -> None:
        self._items: list[object] = []

    def push(self, item: object) -> None:
        self._items.append(item)

    def snapshot(self) -> list[object]:
        return list(self._items)

    def drain(self) -> list[object]:
        items = list(self._items)
        self._items = []
        return items


async def _init_optimizer(**kwargs) -> CrescendoOptimizer:
    # Default max_attempts_per_variant=1 keeps these tests' rotation
    # semantics aligned with the original (one attempt per variant).
    defaults = dict(
        max_turns=3,
        max_backtracks=2,
        success_threshold=0.8,
        max_variants=2,
        max_attempts_per_variant=1,
    )
    defaults.update(kwargs)
    opt = CrescendoOptimizer(**defaults)

    mock_llm = AsyncMock()
    await opt.initialize(
        goal=Goal(description="test objective"),
        controllables=[_make_controllable()],
        observables=[],
        llm_client=mock_llm,
    )
    return opt


@pytest.mark.asyncio
async def test_run_start_returns_event_response():
    opt = await _init_optimizer()
    resp = await opt.on_event(_make_run_start())
    assert isinstance(resp, EventResponse)


@pytest.mark.asyncio
async def test_pre_call_injects_question():
    """ControllablePreCallEvent should trigger attacker and inject a question."""
    opt = await _init_optimizer()
    ctrl = _make_controllable()

    await opt.on_event(_make_run_start())

    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = AttackerOutput(question='What is X?', summary='', rationale='Starting innocuous')
        resp = await opt.on_event(_make_pre_call(ctrl))

    assert isinstance(resp, ControllableInjection)
    assert resp.value == "What is X?"


@pytest.mark.asyncio
async def test_post_call_evaluates_and_returns_no_injection():
    """ControllablePostCallEvent should evaluate the response."""
    opt = await _init_optimizer()
    ctrl = _make_controllable()

    await opt.on_event(_make_run_start())

    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = AttackerOutput(question='Q?', summary='', rationale='reason')
        await opt.on_event(_make_pre_call(ctrl))

    with patch.object(opt._evaluator, "is_refusal", new_callable=AsyncMock) as mock_ref, \
         patch.object(opt._evaluator, "score_response", new_callable=AsyncMock) as mock_score:
        mock_ref.return_value = False
        mock_score.return_value = (0.3, "Low progress")
        resp = await opt.on_event(_make_post_call(ctrl, "Here is some info about X"))

    assert isinstance(resp, ControllableNoInjection)


@pytest.mark.asyncio
async def test_post_call_with_different_controllable_is_processed():
    """PostCall should still be processed when target uses a separate response controllable."""
    opt = await _init_optimizer()
    pre_ctrl = _make_controllable(name="user_message", tag="user")
    post_ctrl = _make_controllable(name="response", tag="assistant")

    await opt.on_event(_make_run_start())

    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = AttackerOutput(question='Q?', summary='', rationale='reason')
        pre_resp = await opt.on_event(_make_pre_call(pre_ctrl))
    assert isinstance(pre_resp, ControllableInjection)

    post_resp = await opt.on_event(_make_post_call(post_ctrl, "Answer text"))

    assert isinstance(post_resp, ControllableNoInjection)
    assert opt._turn == 0

    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as mock_gen, \
         patch.object(opt._evaluator, "is_refusal", new_callable=AsyncMock) as mock_ref, \
         patch.object(opt._evaluator, "score_response", new_callable=AsyncMock) as mock_score:
        mock_gen.return_value = AttackerOutput(question='Q2?', summary='', rationale='reason')
        mock_ref.return_value = False
        mock_score.return_value = (0.4, "Progress")
        next_resp = await opt.on_event(_make_pre_call(pre_ctrl))

    assert isinstance(next_resp, ControllableInjection)
    assert opt._turn == 1
    assert opt._last_response == "Answer text"


@pytest.mark.asyncio
async def test_post_call_pairs_on_injected_value_when_requests_differ():
    """First PostCall can pair using the injected value as request marker."""
    opt = await _init_optimizer()
    pre_ctrl = _make_controllable(name="user_message", tag="user")
    post_ctrl = _make_controllable(name="response", tag="assistant")

    await opt.on_event(_make_run_start())

    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = AttackerOutput(question='Q-turn-1', summary='', rationale='reason')
        pre_resp = await opt.on_event(_make_pre_call_with_request(pre_ctrl, "user message"))
    assert isinstance(pre_resp, ControllableInjection)

    # ChatbotTarget-style shape: PostCall.request carries the injected text,
    # not the original PreCall.request.
    post_resp = await opt.on_event(
        _make_post_call_with_request(post_ctrl, "Answer text", "Q-turn-1"),
    )

    assert isinstance(post_resp, ControllableNoInjection)
    assert opt._turn == 0

    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as mock_gen, \
         patch.object(opt._evaluator, "is_refusal", new_callable=AsyncMock) as mock_ref, \
         patch.object(opt._evaluator, "score_response", new_callable=AsyncMock) as mock_score:
        mock_gen.return_value = AttackerOutput(question='Q-turn-2', summary='', rationale='reason')
        mock_ref.return_value = False
        mock_score.return_value = (0.4, "Progress")
        next_resp = await opt.on_event(_make_pre_call_with_request(pre_ctrl, "user message"))

    assert isinstance(next_resp, ControllableInjection)
    assert opt._turn == 1
    assert opt._last_response == "Answer text"


@pytest.mark.asyncio
async def test_noisy_post_call_before_real_post_call_is_ignored():
    """Out-of-band PostCall events should not hijack the active turn."""
    opt = await _init_optimizer()
    pre_ctrl = _make_controllable(name="user_message", tag="user")
    noisy_ctrl = _make_controllable(name="telemetry", tag="system")
    real_post_ctrl = _make_controllable(name="response", tag="assistant")

    await opt.on_event(_make_run_start())

    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = AttackerOutput(question='Q?', summary='', rationale='reason')
        pre_resp = await opt.on_event(_make_pre_call_with_request(pre_ctrl, "turn-1"))
    assert isinstance(pre_resp, ControllableInjection)

    # First PostCall is noise (wrong controllable + unrelated request) and should be ignored.
    noise_resp = await opt.on_event(
        _make_post_call_with_request(noisy_ctrl, "noise", "metrics-update"),
    )
    assert isinstance(noise_resp, ControllableNoInjection)
    assert opt._turn == 0

    # Real PostCall arrives next and should be accepted.
    real_resp = await opt.on_event(
        _make_post_call_with_request(real_post_ctrl, "real answer", "turn-1"),
    )
    assert isinstance(real_resp, ControllableNoInjection)
    assert opt._turn == 0

    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as mock_gen, \
         patch.object(opt._evaluator, "is_refusal", new_callable=AsyncMock) as mock_ref, \
         patch.object(opt._evaluator, "score_response", new_callable=AsyncMock) as mock_score:
        mock_gen.return_value = AttackerOutput(question='Q2?', summary='', rationale='reason')
        mock_ref.return_value = False
        mock_score.return_value = (0.5, "Mid progress")
        next_resp = await opt.on_event(_make_pre_call_with_request(pre_ctrl, "turn-2"))
    assert isinstance(next_resp, ControllableInjection)
    assert opt._turn == 1
    assert opt._last_response == "real answer"


@pytest.mark.asyncio
async def test_missing_post_call_advances_turn_and_terminates():
    """If target omits PostCall, optimizer should still advance turns and stop at max_turns."""
    opt = await _init_optimizer(max_turns=2, success_threshold=0.99)
    ctrl = _make_controllable()

    await opt.on_event(_make_run_start())

    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as mock_gen:
        mock_gen.side_effect = [
            AttackerOutput(question='Q1?', summary='', rationale='r1'),
            AttackerOutput(question='Q2?', summary='', rationale='r2'),
            AttackerOutput(question='Q3?', summary='', rationale='r3'),
        ]

        # Turn 1 injection, but no PostCall follows.
        resp1 = await opt.on_event(_make_pre_call(ctrl))
        assert isinstance(resp1, ControllableInjection)
        assert opt._turn == 0

        # Next PreCall should synthesize an unsuccessful turn and continue.
        resp2 = await opt.on_event(_make_pre_call(ctrl))
        assert isinstance(resp2, ControllableInjection)
        assert opt._turn == 1

        # Another missing PostCall should consume final turn and terminate.
        resp3 = await opt.on_event(_make_pre_call(ctrl))
        assert isinstance(resp3, ControllableNoInjection)
        assert opt._turn == 2
        assert opt._attempt_done is True
        assert mock_gen.call_count == 2


@pytest.mark.asyncio
async def test_missing_post_call_recovers_from_trajectory_response_observable():
    """If PostCall is filtered, Crescendo can recover answer from trajectory observables."""
    opt = await _init_optimizer(max_turns=3, success_threshold=0.99)
    ctrl = _make_controllable(name="user_message", tag="user")
    response_tag = SecurityDomainTag(name="response_readable")
    traj = _FakeReadableTrajectory()

    await opt.on_event(RunStartEvent(trajectory=traj))

    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as mock_gen, \
         patch.object(opt._evaluator, "is_refusal", new_callable=AsyncMock) as mock_ref, \
         patch.object(opt._evaluator, "score_response", new_callable=AsyncMock) as mock_score:
        mock_gen.side_effect = [
            AttackerOutput(question='Q1?', summary='', rationale='r1'),
            AttackerOutput(question='Q2?', summary='', rationale='r2'),
        ]
        mock_ref.return_value = False
        mock_score.return_value = (0.4, "Recovered from trajectory")

        # Turn 1 injection; target never sends PostCall.
        resp1 = await opt.on_event(_make_pre_call(ctrl))
        assert isinstance(resp1, ControllableInjection)
        assert opt._turn == 0

        # Target emitted response observable to trajectory (in-scope).
        traj.push(ObservableEvent(
            observable=Observable(name="response", security_domain=response_tag),
            content="Recovered answer",
        ))

        # Next PreCall should consume trajectory response and continue.
        resp2 = await opt.on_event(_make_pre_call(ctrl))
        assert isinstance(resp2, ControllableInjection)
        assert opt._turn == 1
        assert opt._last_response == "Recovered answer"
        assert opt._last_rationale == "Recovered from trajectory"
        mock_ref.assert_called_once_with("Recovered answer")
        mock_score.assert_called_once()


@pytest.mark.asyncio
async def test_when_post_call_and_trajectory_both_available_trajectory_wins():
    """Trajectory response should be preferred over post-call answer."""
    opt = await _init_optimizer(max_turns=3, success_threshold=0.99)
    pre_ctrl = _make_controllable(name="user_message", tag="user")
    post_ctrl = _make_controllable(name="response", tag="assistant")
    response_tag = SecurityDomainTag(name="response_readable")
    traj = _FakeReadableTrajectory()

    await opt.on_event(RunStartEvent(trajectory=traj))

    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = AttackerOutput(question='Q1?', summary='', rationale='r1')
        first = await opt.on_event(_make_pre_call(pre_ctrl))
    assert isinstance(first, ControllableInjection)

    post = await opt.on_event(_make_post_call_with_request(post_ctrl, "from-postcall", "Q1?"))
    assert isinstance(post, ControllableNoInjection)

    traj.push(ObservableEvent(
        observable=Observable(name="response", security_domain=response_tag),
        content="from-trajectory",
    ))

    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as mock_gen, \
         patch.object(opt._evaluator, "is_refusal", new_callable=AsyncMock) as mock_ref, \
         patch.object(opt._evaluator, "score_response", new_callable=AsyncMock) as mock_score:
        mock_gen.return_value = AttackerOutput(question='Q2?', summary='', rationale='r2')
        mock_ref.return_value = False
        mock_score.return_value = (0.6, "Trajectory preferred")
        second = await opt.on_event(_make_pre_call_with_request(pre_ctrl, "turn-2"))

    assert isinstance(second, ControllableInjection)
    assert opt._turn == 1
    assert opt._last_response == "from-trajectory"
    assert opt._last_rationale == "Trajectory preferred"


class _SplitPostCallTarget(Target):
    """A minimal multi-turn target with separate pre/post controllables."""

    def __init__(self, max_turns: int = 3) -> None:
        self._max_turns = max_turns
        self._last_response = ""
        self.injected_messages: list[str] = []

    @property
    def config_specs(self) -> list[ConfigSpec]:
        return []

    def set_config(self, name: str, value: str) -> None:
        return None

    @property
    def query_specs(self) -> list[QuerySpec]:
        return [QuerySpec(name="last_response", description="Last response text")]

    def query(self, name: str, **params: str) -> str:
        if name == "last_response":
            return self._last_response
        return ""

    @property
    def security_domain(self) -> SecurityDomain:
        root = SecurityDomainTag("root")
        user = SecurityDomainTag("user", parent=root)
        assistant = SecurityDomainTag("assistant", parent=root)
        return SecurityDomain([root, user, assistant])

    def get_controllables(self) -> list[Controllable]:
        return [
            _make_controllable(name="user_message", tag="user"),
            _make_controllable(name="response", tag="assistant"),
        ]

    def get_observables(self) -> list[ObservableValue]:
        return [
            ObservableValue(
                observable=Observable(
                    name="target_type",
                    security_domain=SecurityDomainTag("root"),
                ),
                content="split-post-target",
            )
        ]

    async def run(self, emit, send_event) -> None:
        user_ctrl = _make_controllable(name="user_message", tag="user")
        response_ctrl = _make_controllable(name="response", tag="assistant")

        for turn in range(self._max_turns):
            request = f"turn-{turn + 1}"
            pre = await send_event(
                ControllablePreCallEvent(controllable=user_ctrl, request=request),
            )
            if not isinstance(pre, ControllableInjection):
                break

            self.injected_messages.append(pre.value)
            answer = f"assistant-answer-{turn + 1}"
            self._last_response = answer
            emit(ObservableEvent(
                observable=Observable(
                    name="assistant_answer",
                    security_domain=SecurityDomainTag("assistant"),
                ),
                content=answer,
            ))
            await send_event(
                ControllablePostCallEvent(
                    controllable=response_ctrl,
                    request=request,
                    answer=answer,
                ),
            )

    async def reset_ephemeral_state(self) -> None:
        self._last_response = ""
        self.injected_messages = []

    async def teardown(self) -> None:
        pass


@pytest.mark.asyncio
async def test_split_post_call_target_run_end_to_end():
    """Target.run with split pre/post controllables should drive Crescendo turns correctly."""
    opt = await _init_optimizer(max_turns=2, success_threshold=0.99)
    target = _SplitPostCallTarget(max_turns=4)

    await opt.on_event(_make_run_start())

    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as mock_gen, \
         patch.object(opt._evaluator, "is_refusal", new_callable=AsyncMock) as mock_ref, \
         patch.object(opt._evaluator, "score_response", new_callable=AsyncMock) as mock_score:
        mock_gen.side_effect = [
            AttackerOutput(question='Q1?', summary='', rationale='r1'),
            AttackerOutput(question='Q2?', summary='', rationale='r2'),
        ]
        mock_ref.return_value = False
        mock_score.side_effect = [
            (0.2, "low"),
            (0.3, "low"),
        ]

        async def send_event(event):
            return await opt.on_event(event)

        target_events = []

        def emit(event):
            target_events.append(event)

        await target.run(emit, send_event)

    # Optimizer should stop the target after max_turns.
    assert opt._turn == 2
    assert opt._attempt_done is True
    assert target.injected_messages == ["Q1?", "Q2?"]
    assert len(target_events) == 2

@pytest.mark.asyncio
async def test_multi_turn_within_single_run():
    """Multiple PreCall/PostCall cycles happen within one run."""
    opt = await _init_optimizer(max_turns=3, success_threshold=0.99)
    ctrl = _make_controllable()

    await opt.on_event(_make_run_start())

    for turn in range(3):
        with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as mock_gen, \
             patch.object(opt._evaluator, "is_refusal", new_callable=AsyncMock) as mock_ref, \
             patch.object(opt._evaluator, "score_response", new_callable=AsyncMock) as mock_score:
            mock_gen.return_value = AttackerOutput(question=f'Q{turn + 1}?', summary='', rationale='r')
            mock_ref.return_value = False
            mock_score.return_value = (0.2 * max(1, turn), "Progress")
            resp = await opt.on_event(_make_pre_call(ctrl))
        assert isinstance(resp, ControllableInjection)
        assert resp.value == f"Q{turn + 1}?"

        await opt.on_event(_make_post_call(ctrl, f"Answer {turn + 1}"))

    # Last turn response is pending until the next PreCall.
    assert opt._turn == 2

    # After max_turns, next PreCall should get NoInjection
    with patch.object(opt._evaluator, "is_refusal", new_callable=AsyncMock) as mock_ref, \
         patch.object(opt._evaluator, "score_response", new_callable=AsyncMock) as mock_score:
        mock_ref.return_value = False
        mock_score.return_value = (0.6, "Progress")
        resp = await opt.on_event(_make_pre_call(ctrl))
    assert isinstance(resp, ControllableNoInjection)
    assert opt._turn == 3


@pytest.mark.asyncio
async def test_success_mid_conversation_stops_injection():
    """Score >= threshold should stop injecting on next PreCall."""
    opt = await _init_optimizer(max_turns=5, success_threshold=0.8)
    ctrl = _make_controllable()

    await opt.on_event(_make_run_start())

    # Turn 1: low score
    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = AttackerOutput(question='Q1?', summary='', rationale='r')
        await opt.on_event(_make_pre_call(ctrl))
    await opt.on_event(_make_post_call(ctrl, "partial"))

    # Turn 2 generation happens while processing turn 1 feedback.
    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as mock_gen, \
         patch.object(opt._evaluator, "is_refusal", new_callable=AsyncMock) as mock_ref, \
         patch.object(opt._evaluator, "score_response", new_callable=AsyncMock) as mock_score:
        mock_gen.return_value = AttackerOutput(question='Q2?', summary='', rationale='r')
        mock_ref.return_value = False
        mock_score.return_value = (0.3, "Low")
        await opt.on_event(_make_pre_call(ctrl))
    await opt.on_event(_make_post_call(ctrl, "harmful"))

    # Success from turn 2 is applied at the next PreCall.
    with patch.object(opt._evaluator, "is_refusal", new_callable=AsyncMock) as mock_ref, \
         patch.object(opt._evaluator, "score_response", new_callable=AsyncMock) as mock_score:
        mock_ref.return_value = False
        mock_score.return_value = (0.9, "High")
        resp = await opt.on_event(_make_pre_call(ctrl))

    assert isinstance(resp, ControllableNoInjection)
    assert opt._succeeded is True

    # RunEnd should signal done
    resp = await opt.on_event(_make_run_end())
    assert resp.done is True


@pytest.mark.asyncio
async def test_backtracking_on_refusal():
    """Refusal should trigger backtracking without incrementing turn."""
    opt = await _init_optimizer()
    ctrl = _make_controllable()

    await opt.on_event(_make_run_start())

    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = AttackerOutput(question='Direct harmful Q?', summary='', rationale='r')
        await opt.on_event(_make_pre_call(ctrl))

    await opt.on_event(_make_post_call(ctrl, "I can't help with that"))

    with patch.object(opt._evaluator, "is_refusal", new_callable=AsyncMock) as mock_ref, \
         patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as mock_gen2, \
         patch.object(opt._attacker, "notify_refusal") as mock_notify:
        mock_ref.return_value = True
        mock_gen2.return_value = AttackerOutput(question='Retry question?', summary='', rationale='r')
        resp = await opt.on_event(_make_pre_call(ctrl))
        mock_notify.assert_called_once_with("Direct harmful Q?")

    assert isinstance(resp, ControllableInjection)
    # Turn should not have incremented, attempt not done
    assert opt._turn == 0
    assert opt._attempt_done is False


@pytest.mark.asyncio
async def test_max_backtracks_stops_backtracking():
    """After max_backtracks, refusals are treated as normal responses."""
    opt = await _init_optimizer(max_backtracks=1, max_turns=5)
    ctrl = _make_controllable()

    await opt.on_event(_make_run_start())

    # First refusal — should backtrack
    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = AttackerOutput(question='Q1?', summary='', rationale='r')
        await opt.on_event(_make_pre_call(ctrl))
    await opt.on_event(_make_post_call(ctrl, "I can't"))
    with patch.object(opt._evaluator, "is_refusal", new_callable=AsyncMock) as mock_ref, \
         patch.object(opt._attacker, "notify_refusal"), \
         patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as mock_gen2:
        mock_ref.return_value = True
        mock_gen2.return_value = AttackerOutput(question='Q2?', summary='', rationale='r')
        resp = await opt.on_event(_make_pre_call(ctrl))
    assert isinstance(resp, ControllableInjection)
    assert opt._backtrack_count == 1
    assert opt._turn == 0

    # Second refusal — backtracking exhausted, treat as normal response
    await opt.on_event(_make_post_call(ctrl, "I can't again"))
    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as mock_gen3, \
         patch.object(opt._evaluator, "is_refusal", new_callable=AsyncMock) as mock_ref, \
         patch.object(opt._evaluator, "score_response", new_callable=AsyncMock) as mock_score:
        mock_gen3.return_value = AttackerOutput(question='Q3?', summary='', rationale='r')
        mock_ref.return_value = True
        mock_score.return_value = (0.0, "Refusal")
        resp2 = await opt.on_event(_make_pre_call(ctrl))
    assert isinstance(resp2, ControllableInjection)

    assert opt._turn == 1
    assert opt._backtrack_count == 1  # didn't increase


@pytest.mark.asyncio
async def test_variant_rotation_across_runs():
    """Each run uses a different variant. Failed attempt → next variant."""
    opt = await _init_optimizer(max_turns=1, max_variants=2)
    ctrl = _make_controllable()

    # Run 1 (attempt 0): exhaust max_turns
    await opt.on_event(_make_run_start())
    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = AttackerOutput(question='Q?', summary='', rationale='r')
        await opt.on_event(_make_pre_call(ctrl))
    await opt.on_event(_make_post_call(ctrl, "response"))
    # NoInjection to terminate target loop
    with patch.object(opt._evaluator, "is_refusal", new_callable=AsyncMock) as mock_ref, \
         patch.object(opt._evaluator, "score_response", new_callable=AsyncMock) as mock_score:
        mock_ref.return_value = False
        mock_score.return_value = (0.2, "Low")
        resp = await opt.on_event(_make_pre_call(ctrl))
    assert isinstance(resp, ControllableNoInjection)

    resp = await opt.on_event(_make_run_end())
    assert resp.done is False  # more attempts available

    # Run 2 (variant 1): should reset state and rotate variant
    await opt.on_event(_make_run_start())
    assert opt._variant_index == 1
    assert opt._variant_attempt == 0
    assert opt._turn == 0


@pytest.mark.asyncio
async def test_all_attempts_exhausted():
    """When all attempts exhausted, RunEnd signals done=True."""
    opt = await _init_optimizer(max_turns=1, max_variants=1)
    ctrl = _make_controllable()

    await opt.on_event(_make_run_start())
    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = AttackerOutput(question='Q?', summary='', rationale='r')
        await opt.on_event(_make_pre_call(ctrl))
    await opt.on_event(_make_post_call(ctrl, "response"))

    # Terminate target loop
    with patch.object(opt._evaluator, "is_refusal", new_callable=AsyncMock) as mock_ref, \
         patch.object(opt._evaluator, "score_response", new_callable=AsyncMock) as mock_score:
        mock_ref.return_value = False
        mock_score.return_value = (0.3, "Low")
        resp = await opt.on_event(_make_pre_call(ctrl))
    assert isinstance(resp, ControllableNoInjection)

    resp = await opt.on_event(_make_run_end())
    assert resp.done is True


@pytest.mark.asyncio
async def test_secondary_controllable_ignored():
    """Non-primary controllable should get NoInjection."""
    opt = await _init_optimizer()
    ctrl1 = _make_controllable()
    ctrl2 = Controllable(name="other", security_domain=SecurityDomainTag(name="sys"), description="secondary")

    await opt.on_event(_make_run_start())

    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = AttackerOutput(question='Q?', summary='', rationale='r')
        resp1 = await opt.on_event(_make_pre_call(ctrl1))
    assert isinstance(resp1, ControllableInjection)

    resp2 = await opt.on_event(_make_pre_call(ctrl2))
    assert isinstance(resp2, ControllableNoInjection)


@pytest.mark.asyncio
async def test_attacker_failure_escapes_instead_of_injecting_filler():
    """A dead attacker must not be scored as a weak one.

    The optimizer used to substitute a generic benign question here. That
    question was then sent to the target and scored, so a broken attacker
    produced a legitimate-looking score of 0.
    """
    opt = await _init_optimizer()
    ctrl = _make_controllable()

    await opt.on_event(_make_run_start())

    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as mock_gen:
        mock_gen.side_effect = HelperLLMUnavailableError(
            "Crescendo attacker generation (turn 1) failed on all 3 attempts"
        )
        with pytest.raises(HelperLLMUnavailableError):
            await opt.on_event(_make_pre_call(ctrl))

    assert opt._current_question is None
    assert not hasattr(opt, "_build_fallback_question")


@pytest.mark.asyncio
async def test_attacker_budget_exhaustion_escapes_on_event():
    """BudgetExhaustedError reaches the controller, which maps it to
    stop_reason="budget_exhausted".

    Swallowing it is why the first archive shows 2,299 Crescendo tasks that
    had spent their entire $0.75 cap yet reported zero budget_exhausted tasks.
    """
    opt = await _init_optimizer()
    ctrl = _make_controllable()

    await opt.on_event(_make_run_start())

    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as mock_gen:
        mock_gen.side_effect = BudgetExhaustedError(
            "Cost cap reached: $0.750131/$0.750000",
            usage=LLMUsage(calls=214, cost=0.750131),
        )
        with pytest.raises(BudgetExhaustedError):
            await opt.on_event(_make_pre_call(ctrl))


@pytest.mark.asyncio
async def test_refusal_check_budget_exhaustion_escapes_on_event():
    """Same for the internal refusal check, which runs on every turn."""
    opt = await _init_optimizer()
    ctrl = _make_controllable()
    await opt.on_event(_make_run_start())

    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = AttackerOutput(question="Q?", summary="", rationale="r")
        await opt.on_event(_make_pre_call(ctrl))
    await opt.on_event(_make_post_call(ctrl, "Some answer"))

    with patch.object(opt._evaluator, "is_refusal", new_callable=AsyncMock) as mock_ref:
        mock_ref.side_effect = BudgetExhaustedError(
            "Cost cap reached: $0.750131/$0.750000",
            usage=LLMUsage(calls=214, cost=0.750131),
        )
        with pytest.raises(BudgetExhaustedError):
            await opt.on_event(_make_pre_call(ctrl))
    assert mock_ref.await_count == 1  # never retried


@pytest.mark.asyncio
async def test_scoring_budget_exhaustion_escapes_on_event():
    """Same for the internal scorer."""
    opt = await _init_optimizer()
    ctrl = _make_controllable()
    await opt.on_event(_make_run_start())

    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = AttackerOutput(question="Q?", summary="", rationale="r")
        await opt.on_event(_make_pre_call(ctrl))
    await opt.on_event(_make_post_call(ctrl, "Some answer"))

    with patch.object(opt._evaluator, "is_refusal", new_callable=AsyncMock) as mock_ref, \
         patch.object(opt._evaluator, "score_response", new_callable=AsyncMock) as mock_score:
        mock_ref.return_value = False
        mock_score.side_effect = BudgetExhaustedError(
            "Cost cap reached: $0.750131/$0.750000",
            usage=LLMUsage(calls=214, cost=0.750131),
        )
        with pytest.raises(BudgetExhaustedError):
            await opt.on_event(_make_pre_call(ctrl))


@pytest.mark.asyncio
async def test_transient_evaluator_failure_is_retried_then_degraded(monkeypatch):
    """A transient evaluator failure is retried; a persistent one degrades
    but is counted, because the internal evaluator cannot change the verdict
    the security claim's judge records."""
    monkeypatch.setattr(retry, "asyncio", _NoSleepAsyncio())
    opt = await _init_optimizer()
    ctrl = _make_controllable()
    await opt.on_event(_make_run_start())

    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = AttackerOutput(question="Q?", summary="", rationale="r")
        await opt.on_event(_make_pre_call(ctrl))
    await opt.on_event(_make_post_call(ctrl, "Some answer"))

    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as mock_gen, \
         patch.object(opt._evaluator, "is_refusal", new_callable=AsyncMock) as mock_ref, \
         patch.object(opt._evaluator, "score_response", new_callable=AsyncMock) as mock_score:
        mock_gen.return_value = AttackerOutput(question="Q2?", summary="", rationale="r")
        # Archive shape: litellm.APIConnectionError from the attacker proxy.
        mock_ref.side_effect = [
            ConnectionError("litellm.APIConnectionError: connection error"),
            False,
        ]
        mock_score.side_effect = ConnectionError(
            "litellm.APIConnectionError: connection error"
        )
        resp = await opt.on_event(_make_pre_call(ctrl))

    assert isinstance(resp, ControllableInjection)
    assert mock_ref.await_count == 2  # retried once, then succeeded
    assert mock_score.await_count == DEFAULT_ATTEMPTS
    assert opt._degraded == {"turn_scoring": 1}
    assert opt._last_score == 0.0
