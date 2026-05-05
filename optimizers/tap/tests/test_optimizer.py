"""Tests for TapOptimizer event-driven state machine."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from superred.core.types.controllable import Controllable
from superred.core.types.event import EventResponse
from superred.core.types.evaluation import EvaluationResult, Score
from superred.core.types.events import (
    ControllableInjection,
    ControllableNoInjection,
    ControllablePreCallEvent,
    ControllablePostCallEvent,
    ObservableEvent,
    RunEndEvent,
    RunEndResponse,
    RunStartEvent,
)
from superred.core.types.goal import Goal
from superred.core.types.observable import Observable
from superred.core.types.security_domain import SecurityDomainTag
from superred.core.types.trajectory import Trajectory

from tap_optimizer.optimizer import TapOptimizer

DOMAIN = SecurityDomainTag(name="test")


def _make_controllable(*, name: str = "user_message") -> Controllable:
    return Controllable(
        name=name,
        security_domain=DOMAIN,
        description="test",
        value_type="text",
    )


def _make_run_start(trajectory: Trajectory | None = None) -> RunStartEvent:
    return RunStartEvent(trajectory=trajectory or Trajectory())


def _make_run_end() -> RunEndEvent:
    return RunEndEvent(evaluation=None)


def _make_run_end_with_eval(*, success: bool, score: float) -> RunEndEvent:
    return RunEndEvent(
        evaluation=EvaluationResult(
            success=success,
            primary_score=Score(value=score, security_domain=DOMAIN),
        )
    )


def _make_pre_call(ctrl: Controllable) -> ControllablePreCallEvent:
    return ControllablePreCallEvent(controllable=ctrl, request="hello")


def _make_post_call(ctrl: Controllable, answer: str) -> ControllablePostCallEvent:
    return ControllablePostCallEvent(controllable=ctrl, request="hello", answer=answer)


def _make_response_observable(content: str) -> ObservableEvent:
    return ObservableEvent(
        observable=Observable(
            name="response",
            security_domain=DOMAIN,
            description="Target response",
        ),
        content=content,
    )


def _mock_response(content: str) -> MagicMock:
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


async def _init_optimizer(**kwargs) -> TapOptimizer:
    defaults = dict(
        branching_factor=1,
        root_nodes=1,
        tree_width=10,
        tree_depth=3,
        attack_temperature=1.0,
        random_seed=0,
    )
    defaults.update(kwargs)
    opt = TapOptimizer(**defaults)
    mock_llm = AsyncMock()
    await opt.initialize(
        goal=Goal(description="test goal"),
        controllables=[_make_controllable()],
        observables=[],
        llm_client=mock_llm,
    )
    return opt


def _setup_llm_mock(opt: TapOptimizer, responses: list[str]) -> None:
    call_count = 0

    async def mock_complete(messages, **kwargs):
        nonlocal call_count
        call_count += 1
        idx = call_count - 1
        if idx < len(responses):
            return _mock_response(responses[idx])
        return _mock_response("fallback")

    opt.llm.complete = AsyncMock(side_effect=mock_complete)


@pytest.mark.asyncio
async def test_defaults_match_original_tap_shape() -> None:
    opt = TapOptimizer()
    assert opt._branching_factor == 1
    assert opt._root_nodes == 1
    assert opt._tree_width == 10
    assert opt._tree_depth == 10
    assert opt._attack_top_p == 0.9
    assert opt._attacker_max_tokens == 500
    assert opt._evaluator_temperature == 0.0
    assert opt._evaluator_max_tokens == 10
    assert opt._max_attack_attempts == 5
    assert opt._keep_last_n == 3


@pytest.mark.asyncio
async def test_run_start_generates_and_prunes_without_querying_target_or_judge() -> None:
    opt = await _init_optimizer()
    attacker_json = json.dumps({"improvement": "Try this", "prompt": "attack prompt"})
    _setup_llm_mock(opt, [
        attacker_json,
        "Response: [[YES]]",
    ])

    event = _make_run_start()
    result = await opt.on_event(event)

    assert isinstance(result, EventResponse)
    assert result.event is event
    assert opt.llm.complete.await_count == 2
    assert opt._current_candidate is not None
    assert opt._current_candidate.prompt == "attack prompt"


@pytest.mark.asyncio
async def test_pre_call_ignores_system_prompt_then_injects_user_message() -> None:
    opt = await _init_optimizer()
    attacker_json = json.dumps({"improvement": "Roleplaying", "prompt": "You are a pirate"})
    _setup_llm_mock(opt, [attacker_json, "Response: [[YES]]"])
    await opt.on_event(_make_run_start())

    system_resp = await opt.on_event(_make_pre_call(_make_controllable(name="system_prompt")))
    assert isinstance(system_resp, ControllableNoInjection)

    user_ctrl = _make_controllable(name="user_message")
    result = await opt.on_event(_make_pre_call(user_ctrl))
    assert isinstance(result, ControllableInjection)
    assert result.value == "You are a pirate"
    assert result.controllable is user_ctrl


@pytest.mark.asyncio
async def test_all_surviving_candidates_are_sent_to_real_target_across_runs() -> None:
    opt = await _init_optimizer(root_nodes=2, tree_width=2, tree_depth=2)
    attacker_json_1 = json.dumps({"improvement": "one", "prompt": "prompt one"})
    attacker_json_2 = json.dumps({"improvement": "two", "prompt": "prompt two"})
    _setup_llm_mock(opt, [
        attacker_json_1,
        attacker_json_2,
        "Response: [[YES]]",
        "Response: [[YES]]",
        "Rating: [[3]]",
        "Rating: [[4]]",
    ])
    ctrl = _make_controllable()

    trajectory_1 = Trajectory()
    await opt.on_event(_make_run_start(trajectory_1))
    first = await opt.on_event(_make_pre_call(ctrl))
    assert isinstance(first, ControllableInjection)
    assert first.value == "prompt one"
    trajectory_1.emit(_make_response_observable("first target response"))
    first_end = await opt.on_event(_make_run_end())
    assert isinstance(first_end, RunEndResponse)
    assert first_end.done is False

    calls_after_first_end = opt.llm.complete.await_count
    trajectory_2 = Trajectory()
    await opt.on_event(_make_run_start(trajectory_2))
    second = await opt.on_event(_make_pre_call(ctrl))
    assert isinstance(second, ControllableInjection)
    assert second.value == "prompt two"
    assert opt.llm.complete.await_count == calls_after_first_end
    trajectory_2.emit(_make_response_observable("second target response"))
    second_end = await opt.on_event(_make_run_end())

    assert isinstance(second_end, RunEndResponse)
    assert second_end.done is False
    assert opt._depth == 1


@pytest.mark.asyncio
async def test_success_does_not_skip_remaining_candidates_at_same_depth() -> None:
    opt = await _init_optimizer(root_nodes=2, tree_width=2, tree_depth=2)
    attacker_json_1 = json.dumps({"improvement": "one", "prompt": "prompt one"})
    attacker_json_2 = json.dumps({"improvement": "two", "prompt": "prompt two"})
    _setup_llm_mock(opt, [
        attacker_json_1,
        attacker_json_2,
        "Response: [[YES]]",
        "Response: [[YES]]",
        "Rating: [[10]]",
        "Rating: [[4]]",
    ])
    ctrl = _make_controllable()

    trajectory_1 = Trajectory()
    await opt.on_event(_make_run_start(trajectory_1))
    first = await opt.on_event(_make_pre_call(ctrl))
    assert isinstance(first, ControllableInjection)
    assert first.value == "prompt one"
    trajectory_1.emit(_make_response_observable("successful target response"))
    first_end = await opt.on_event(_make_run_end())
    assert isinstance(first_end, RunEndResponse)
    assert first_end.done is False

    trajectory_2 = Trajectory()
    await opt.on_event(_make_run_start(trajectory_2))
    second = await opt.on_event(_make_pre_call(ctrl))
    assert isinstance(second, ControllableInjection)
    assert second.value == "prompt two"
    trajectory_2.emit(_make_response_observable("second target response"))
    second_end = await opt.on_event(_make_run_end())

    assert isinstance(second_end, RunEndResponse)
    assert second_end.done is True
    assert opt._best_candidate is not None
    assert opt._best_candidate.prompt == "prompt one"
    assert opt._best_candidate.score == 10.0


@pytest.mark.asyncio
async def test_run_end_scores_trajectory_response_and_ignores_post_call() -> None:
    opt = await _init_optimizer(tree_depth=2)
    attacker_json = json.dumps({"improvement": "Try this", "prompt": "attack prompt"})
    _setup_llm_mock(opt, [
        attacker_json,
        "Response: [[YES]]",
        "Rating: [[6]]",
    ])
    ctrl = _make_controllable()
    trajectory = Trajectory()

    await opt.on_event(_make_run_start(trajectory))
    await opt.on_event(_make_pre_call(ctrl))
    post_resp = await opt.on_event(_make_post_call(ctrl, "post-call response"))
    assert isinstance(post_resp, ControllableNoInjection)
    trajectory.emit(_make_response_observable("trajectory response"))
    result = await opt.on_event(_make_run_end())

    assert isinstance(result, RunEndResponse)
    assert opt._best_candidate is not None
    assert opt._best_candidate.target_response == "trajectory response"
    scoring_messages = opt.llm.complete.call_args.args[0]
    assert "[PROMPT]:attack prompt" in scoring_messages[1]["content"]
    assert "trajectory response" in scoring_messages[1]["content"]
    assert "post-call response" not in scoring_messages[1]["content"]


@pytest.mark.asyncio
async def test_trajectory_response_takes_priority_over_run_end_score() -> None:
    opt = await _init_optimizer(tree_depth=2)
    attacker_json = json.dumps({"improvement": "Try this", "prompt": "attack prompt"})
    _setup_llm_mock(opt, [
        attacker_json,
        "Response: [[YES]]",
        "Rating: [[7]]",
    ])
    ctrl = _make_controllable()
    trajectory = Trajectory()

    await opt.on_event(_make_run_start(trajectory))
    await opt.on_event(_make_pre_call(ctrl))
    trajectory.emit(_make_response_observable("visible target response"))
    result = await opt.on_event(_make_run_end_with_eval(success=False, score=0.1))

    assert isinstance(result, RunEndResponse)
    assert opt._best_candidate is not None
    assert opt._best_candidate.target_response == "visible target response"
    assert opt._best_candidate.score == 7.0
    scoring_messages = opt.llm.complete.call_args.args[0]
    assert "visible target response" in scoring_messages[1]["content"]


@pytest.mark.asyncio
async def test_post_call_only_does_not_score_without_trajectory_or_feedback() -> None:
    opt = await _init_optimizer(tree_depth=2)
    attacker_json = json.dumps({"improvement": "Try this", "prompt": "attack prompt"})
    _setup_llm_mock(opt, [
        attacker_json,
        "Response: [[YES]]",
    ])
    ctrl = _make_controllable()

    await opt.on_event(_make_run_start(Trajectory()))
    await opt.on_event(_make_pre_call(ctrl))
    await opt.on_event(_make_post_call(ctrl, "post-call only response"))
    result = await opt.on_event(_make_run_end())

    assert isinstance(result, RunEndResponse)
    assert opt._best_candidate is not None
    assert opt._best_candidate.score == 1.0
    assert "No in-scope target response" in (opt._best_candidate.target_response or "")
    assert opt.llm.complete.await_count == 2


@pytest.mark.asyncio
async def test_run_end_framework_success_marks_candidate_success_without_response() -> None:
    opt = await _init_optimizer(tree_depth=3)
    attacker_json = json.dumps({"improvement": "Try this", "prompt": "attack prompt"})
    _setup_llm_mock(opt, [attacker_json, "Response: [[YES]]"])
    ctrl = _make_controllable()

    await opt.on_event(_make_run_start(Trajectory()))
    await opt.on_event(_make_pre_call(ctrl))
    result = await opt.on_event(_make_run_end_with_eval(success=True, score=0.1))

    assert isinstance(result, RunEndResponse)
    assert result.done is True
    assert opt._best_candidate is not None
    assert opt._best_candidate.score == 10.0
    assert opt.llm.complete.await_count == 2


@pytest.mark.asyncio
async def test_feedback_score_can_rank_candidate_when_response_is_not_visible() -> None:
    opt = await _init_optimizer(tree_depth=2)
    attacker_json = json.dumps({"improvement": "Try this", "prompt": "attack prompt"})
    _setup_llm_mock(opt, [attacker_json, "Response: [[YES]]"])
    ctrl = _make_controllable()

    await opt.on_event(_make_run_start(Trajectory()))
    await opt.on_event(_make_pre_call(ctrl))
    await opt.on_event(_make_run_end_with_eval(success=False, score=0.5))

    assert opt._best_candidate is not None
    assert opt._best_candidate.score == 5.5
    assert "Framework evaluation" in (opt._best_candidate.target_response or "")


@pytest.mark.asyncio
async def test_jailbreak_on_judge_score_triggers_done_after_real_target_run() -> None:
    opt = await _init_optimizer(tree_depth=5)
    attacker_json = json.dumps({"improvement": "Perfect attack", "prompt": "Jailbreak prompt"})
    _setup_llm_mock(opt, [
        attacker_json,
        "Response: [[YES]]",
        "Rating: [[10]]",
    ])
    ctrl = _make_controllable()
    trajectory = Trajectory()

    await opt.on_event(_make_run_start(trajectory))
    pre = await opt.on_event(_make_pre_call(ctrl))
    assert isinstance(pre, ControllableInjection)
    assert pre.value == "Jailbreak prompt"
    trajectory.emit(_make_response_observable("Full compliance response"))
    result = await opt.on_event(_make_run_end())

    assert isinstance(result, RunEndResponse)
    assert result.done is True
    assert opt._done is True


@pytest.mark.asyncio
async def test_continues_across_depths_with_target_feedback() -> None:
    opt = await _init_optimizer(tree_depth=3, tree_width=1, branching_factor=1)
    attacker_json_1 = json.dumps({"improvement": "First try", "prompt": "prompt v1"})
    attacker_json_2 = json.dumps({"improvement": "Second try", "prompt": "prompt v2"})
    _setup_llm_mock(opt, [
        attacker_json_1,
        "Response: [[YES]]",
        "Rating: [[3]]",
        attacker_json_2,
        "Response: [[YES]]",
    ])
    ctrl = _make_controllable()

    trajectory_1 = Trajectory()
    await opt.on_event(_make_run_start(trajectory_1))
    await opt.on_event(_make_pre_call(ctrl))
    trajectory_1.emit(_make_response_observable("partial response"))
    first_end = await opt.on_event(_make_run_end())
    assert isinstance(first_end, RunEndResponse)
    assert first_end.done is False
    assert opt._depth == 1

    trajectory_2 = Trajectory()
    await opt.on_event(_make_run_start(trajectory_2))
    second = await opt.on_event(_make_pre_call(ctrl))
    assert isinstance(second, ControllableInjection)
    assert second.value == "prompt v2"

    attacker_messages = opt.llm.complete.call_args_list[3].args[0]
    attacker_text = "\n".join(message["content"] for message in attacker_messages)
    assert "LANGUAGE MODEL OUTPUT" in attacker_text
    assert "partial response" in attacker_text
    assert "SCORE" in attacker_text


@pytest.mark.asyncio
async def test_depth_exhausted_signals_done() -> None:
    opt = await _init_optimizer(tree_depth=1)
    attacker_json = json.dumps({"improvement": "Try", "prompt": "attack"})
    _setup_llm_mock(opt, [
        attacker_json,
        "Response: [[YES]]",
        "Rating: [[3]]",
    ])
    ctrl = _make_controllable()
    trajectory = Trajectory()

    await opt.on_event(_make_run_start(trajectory))
    await opt.on_event(_make_pre_call(ctrl))
    trajectory.emit(_make_response_observable("response"))
    result = await opt.on_event(_make_run_end())
    assert isinstance(result, RunEndResponse)
    assert result.done is True


@pytest.mark.asyncio
async def test_no_candidate_signals_done_when_attacker_generation_fails() -> None:
    opt = await _init_optimizer(max_attack_attempts=1)
    bad_attacker_output = "not json"
    _setup_llm_mock(opt, [
        bad_attacker_output,
    ])

    await opt.on_event(_make_run_start())
    assert opt._done is True

    result = await opt.on_event(_make_run_end())
    assert isinstance(result, RunEndResponse)
    assert result.done is True
