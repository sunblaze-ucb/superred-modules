"""Tests for ManyShotOptimizer event-driven state machine."""

import json
import random
from unittest.mock import AsyncMock
import pytest

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
from superred.core.types.trajectory import Trajectory

from tests.conftest import mock_response, make_controllable, DOMAIN
from many_shot_optimizer.optimizer import ManyShotOptimizer


def _make_run_start():
    return RunStartEvent(trajectory=Trajectory())


def _make_response_observable(content: str):
    return ObservableEvent(
        observable=Observable(
            name="response",
            security_domain=DOMAIN,
            description="Target response",
        ),
        content=content,
    )


def _make_run_end():
    return RunEndEvent(evaluation=None)


def _make_run_end_with_eval(*, success: bool, score: float):
    return RunEndEvent(
        evaluation=EvaluationResult(
            success=success,
            primary_score=Score(value=score, security_domain=DOMAIN),
        )
    )


def _make_pre_call(ctrl):
    return ControllablePreCallEvent(controllable=ctrl, request="user input")


def _make_post_call(ctrl, answer):
    return ControllablePostCallEvent(
        controllable=ctrl, request="user input", answer=answer
    )


FAKE_EXAMPLES = [{"user": f"Q{i}", "assistant": f"A{i}"} for i in range(50)]


def _example_text(index: int) -> str:
    return f"User: Q{index} Assistant: A{index}"


async def _init_optimizer(**kwargs) -> ManyShotOptimizer:
    defaults = dict(
        example_counts=[5, 10],
        success_threshold=0.8,
        custom_examples=FAKE_EXAMPLES,
        generate_extra=False,
        shuffle_examples=False,
    )
    defaults.update(kwargs)
    opt = ManyShotOptimizer(**defaults)
    mock_llm = AsyncMock()
    await opt.initialize(
        goal=Goal(description="test objective"),
        controllables=[make_controllable()],
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
async def test_pre_call_injects_many_shot_prompt():
    opt = await _init_optimizer(example_counts=[3])
    ctrl = make_controllable()
    await opt.on_event(_make_run_start())
    resp = await opt.on_event(_make_pre_call(ctrl))
    assert isinstance(resp, ControllableInjection)
    assert "You are a helpful assistant" in resp.value
    assert "test objective" in resp.value


@pytest.mark.asyncio
async def test_prompt_contains_correct_example_count():
    opt = await _init_optimizer(example_counts=[5])
    ctrl = make_controllable()
    await opt.on_event(_make_run_start())
    resp = await opt.on_event(_make_pre_call(ctrl))
    assert isinstance(resp, ControllableInjection)
    assert resp.value.count("Assistant:") == 5


@pytest.mark.asyncio
async def test_defaults_use_256_shots_and_shuffle_examples():
    seed = 7
    examples = [{"user": f"Q{i}", "assistant": f"A{i}"} for i in range(300)]
    opt = await _init_optimizer(
        example_counts=None,
        custom_examples=examples,
        random_seed=seed,
        shuffle_examples=True,
    )
    ctrl = make_controllable()
    expected = list(examples)
    random.Random(seed).shuffle(expected)
    expected = expected[:4]

    await opt.on_event(_make_run_start())
    resp = await opt.on_event(_make_pre_call(ctrl))

    assert isinstance(resp, ControllableInjection)
    assert resp.value.count("Assistant:") == 256
    positions = [
        resp.value.index(f"User: {example['user']} Assistant: {example['assistant']}")
        for example in expected
    ]
    assert positions == sorted(positions)


@pytest.mark.asyncio
async def test_examples_preserve_source_order_when_shuffle_disabled():
    opt = await _init_optimizer(example_counts=[4], shuffle_examples=False)
    ctrl = make_controllable()

    await opt.on_event(_make_run_start())
    resp = await opt.on_event(_make_pre_call(ctrl))

    assert isinstance(resp, ControllableInjection)
    positions = [resp.value.index(_example_text(index)) for index in range(4)]
    assert positions == sorted(positions)


@pytest.mark.asyncio
async def test_shuffle_examples_is_opt_in_and_seeded():
    seed = 7
    opt = await _init_optimizer(
        example_counts=[4], shuffle_examples=True, random_seed=seed
    )
    ctrl = make_controllable()

    expected = list(FAKE_EXAMPLES)
    random.Random(seed).shuffle(expected)
    expected = expected[:4]

    await opt.on_event(_make_run_start())
    resp = await opt.on_event(_make_pre_call(ctrl))

    assert isinstance(resp, ControllableInjection)
    positions = [
        resp.value.index(f"User: {example['user']} Assistant: {example['assistant']}")
        for example in expected
    ]
    assert positions == sorted(positions)


@pytest.mark.asyncio
async def test_success_triggers_done():
    opt = await _init_optimizer(success_threshold=0.8)
    ctrl = make_controllable()
    await opt.on_event(_make_run_start())
    await opt.on_event(_make_pre_call(ctrl))

    opt.llm.complete.return_value = mock_response(
        '{"refusal": false, "rationale": "Attempts the requested behavior."}'
    )
    await opt.on_event(_make_post_call(ctrl, "full compliance"))
    resp = await opt.on_event(_make_run_end())
    assert isinstance(resp, RunEndResponse)
    assert resp.done is True


@pytest.mark.asyncio
async def test_run_end_prefers_trajectory_response_over_post_call():
    opt = await _init_optimizer(example_counts=[3], success_threshold=0.8)
    ctrl = make_controllable()
    trajectory = Trajectory()

    await opt.on_event(RunStartEvent(trajectory=trajectory))
    await opt.on_event(_make_pre_call(ctrl))
    await opt.on_event(_make_post_call(ctrl, "post-call fallback"))
    trajectory.emit(_make_response_observable("trajectory response"))

    opt.llm.complete.return_value = mock_response(
        '{"refusal": false, "rationale": "Attempts the requested behavior."}'
    )
    resp = await opt.on_event(_make_run_end())

    assert isinstance(resp, RunEndResponse)
    assert resp.done is True
    scoring_prompt = opt.llm.complete.call_args.args[0][0]["content"]
    assert "trajectory response" in scoring_prompt
    assert "post-call fallback" not in scoring_prompt


@pytest.mark.asyncio
async def test_escalation_through_schedule():
    opt = await _init_optimizer(example_counts=[3, 6])
    ctrl = make_controllable()

    # First attempt: 3 examples
    await opt.on_event(_make_run_start())
    resp1 = await opt.on_event(_make_pre_call(ctrl))
    assert resp1.value.count("Assistant:") == 3

    opt.llm.complete.return_value = mock_response(
        '{"refusal": true, "rationale": "Refuses the request."}'
    )
    await opt.on_event(_make_post_call(ctrl, "I can't help with that request."))
    end_resp = await opt.on_event(_make_run_end())
    assert end_resp.done is False

    # Second attempt: 6 examples
    await opt.on_event(_make_run_start())
    resp2 = await opt.on_event(_make_pre_call(ctrl))
    assert resp2.value.count("Assistant:") == 6


@pytest.mark.asyncio
async def test_all_attempts_exhausted():
    opt = await _init_optimizer(example_counts=[3])
    ctrl = make_controllable()

    await opt.on_event(_make_run_start())
    await opt.on_event(_make_pre_call(ctrl))

    opt.llm.complete.return_value = mock_response(
        '{"refusal": true, "rationale": "Refuses the request."}'
    )
    await opt.on_event(_make_post_call(ctrl, "I can't help with that request."))
    resp = await opt.on_event(_make_run_end())
    assert resp.done is True


@pytest.mark.asyncio
async def test_custom_example_counts():
    opt = await _init_optimizer(example_counts=[7, 14, 21])
    ctrl = make_controllable()

    await opt.on_event(_make_run_start())
    resp = await opt.on_event(_make_pre_call(ctrl))
    assert resp.value.count("Assistant:") == 7


@pytest.mark.asyncio
async def test_secondary_controllable_ignored():
    opt = await _init_optimizer()
    ctrl1 = make_controllable()
    ctrl2 = make_controllable(name="other")

    await opt.on_event(_make_run_start())
    resp1 = await opt.on_event(_make_pre_call(ctrl1))
    assert isinstance(resp1, ControllableInjection)

    resp2 = await opt.on_event(_make_pre_call(ctrl2))
    assert isinstance(resp2, ControllableNoInjection)


@pytest.mark.asyncio
async def test_evaluator_failure_gives_low_score():
    opt = await _init_optimizer(example_counts=[3, 6])
    ctrl = make_controllable()

    await opt.on_event(_make_run_start())
    await opt.on_event(_make_pre_call(ctrl))

    opt.llm.complete.side_effect = Exception("LLM down")
    await opt.on_event(_make_post_call(ctrl, "some response"))
    resp = await opt.on_event(_make_run_end())
    assert resp.done is False


@pytest.mark.asyncio
async def test_llm_generation_triggered_when_count_exceeds_bundled():
    small_examples = [{"user": f"Q{i}", "assistant": f"A{i}"} for i in range(5)]
    opt = await _init_optimizer(
        example_counts=[10],
        custom_examples=small_examples,
        generate_extra=True,
    )
    ctrl = make_controllable()

    # Mock LLM for generator (returns 5 generated examples)
    generated = [{"user": f"GQ{i}", "assistant": f"GA{i}"} for i in range(5)]
    opt.llm.complete.return_value = mock_response(json.dumps(generated))

    await opt.on_event(_make_run_start())
    resp = await opt.on_event(_make_pre_call(ctrl))
    assert isinstance(resp, ControllableInjection)
    assert resp.value.count("Assistant:") == 10


@pytest.mark.asyncio
async def test_custom_category_filter_limits_examples():
    examples = [
        {"user": "A question", "assistant": "A answer", "category": "alpha"},
        {"user": "B question", "assistant": "B answer", "category": "beta"},
        {"user": "C question", "assistant": "C answer", "category": "alpha"},
    ]
    opt = await _init_optimizer(
        example_counts=[2],
        custom_examples=examples,
        example_category="alpha",
        shuffle_examples=False,
    )
    ctrl = make_controllable()

    await opt.on_event(_make_run_start())
    resp = await opt.on_event(_make_pre_call(ctrl))

    assert isinstance(resp, ControllableInjection)
    assert "A question" in resp.value
    assert "C question" in resp.value
    assert "B question" not in resp.value


@pytest.mark.asyncio
async def test_custom_category_filter_rejects_missing_category():
    examples = [{"user": "A question", "assistant": "A answer", "category": "alpha"}]
    opt = ManyShotOptimizer(
        example_counts=[1],
        custom_examples=examples,
        example_category="missing",
    )
    with pytest.raises(ValueError, match="No ManyShot examples matched"):
        await opt.initialize(
            goal=Goal(description="test objective"),
            controllables=[make_controllable()],
            observables=[],
            llm_client=AsyncMock(),
        )


@pytest.mark.asyncio
async def test_run_end_evaluation_can_mark_success_without_post_call():
    opt = await _init_optimizer(example_counts=[3, 6])
    ctrl = make_controllable()
    await opt.on_event(_make_run_start())
    await opt.on_event(_make_pre_call(ctrl))

    resp = await opt.on_event(_make_run_end_with_eval(success=True, score=0.1))
    assert isinstance(resp, RunEndResponse)
    assert resp.done is True
    assert opt._best_score == 0.1


@pytest.mark.asyncio
async def test_run_end_evaluation_score_can_mark_success():
    opt = await _init_optimizer(example_counts=[3, 6], success_threshold=0.8)
    ctrl = make_controllable()
    await opt.on_event(_make_run_start())
    await opt.on_event(_make_pre_call(ctrl))

    resp = await opt.on_event(_make_run_end_with_eval(success=False, score=0.85))
    assert isinstance(resp, RunEndResponse)
    assert resp.done is True
    assert opt._best_score == 0.85


@pytest.mark.asyncio
async def test_split_chatbot_flow_ignores_system_prompt_then_injects_user_message():
    opt = await _init_optimizer(example_counts=[3])
    system_ctrl = make_controllable(name="system_prompt")
    user_ctrl = make_controllable(name="user_message")
    await opt.on_event(_make_run_start())

    system_resp = await opt.on_event(_make_pre_call(system_ctrl))
    assert isinstance(system_resp, ControllableNoInjection)

    user_resp = await opt.on_event(_make_pre_call(user_ctrl))
    assert isinstance(user_resp, ControllableInjection)
    assert "test objective" in user_resp.value


@pytest.mark.asyncio
async def test_injects_only_once_per_run():
    opt = await _init_optimizer(example_counts=[3])
    ctrl = make_controllable()
    await opt.on_event(_make_run_start())

    first = await opt.on_event(_make_pre_call(ctrl))
    second = await opt.on_event(_make_pre_call(ctrl))
    assert isinstance(first, ControllableInjection)
    assert isinstance(second, ControllableNoInjection)
