"""Tests for capability-aware extensions on CrescendoOptimizer.

Covers:
    (a)/(b) static observable consumption (model id, system prompt readable)
    (c)    system_prompt writable controllable + pretext lift (Option C2)
    (d)    dual evaluator: task verdict authoritative at RunEnd
    (e)    cross-attempt-within-variant memory (failure log)
    bug-fix: name-based PreCall dispatch tolerates system_prompt-first ordering
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from superred.core.types.controllable import Controllable
from superred.core.types.evaluation import EvaluationResult, Score
from superred.core.types.events import (
    ControllableInjection,
    ControllableNoInjection,
    ControllablePreCallEvent,
    ControllablePostCallEvent,
    RunEndEvent,
    RunStartEvent,
)
from superred.core.types.goal import Goal
from superred.core.types.observable import Observable, ObservableValue
from superred.core.types.security_domain import SecurityDomainTag

from crescendo_optimizer.attacker import AttackerOutput, ReplayPlan, TurnRecord
from crescendo_optimizer.optimizer import CrescendoOptimizer


def _ctrl(name: str, tag: str = "user") -> Controllable:
    return Controllable(name=name, security_domain=SecurityDomainTag(name=tag))


def _obs(name: str, content: str, tag: str = "system") -> ObservableValue:
    return ObservableValue(
        observable=Observable(name=name, security_domain=SecurityDomainTag(name=tag)),
        content=content,
    )


def _run_start() -> RunStartEvent:
    return RunStartEvent(trajectory=MagicMock())


def _pre_call(ctrl: Controllable, request: str = "user input") -> ControllablePreCallEvent:
    return ControllablePreCallEvent(controllable=ctrl, request=request)


def _post_call(
    ctrl: Controllable, answer: str, request: str = "user input",
) -> ControllablePostCallEvent:
    return ControllablePostCallEvent(controllable=ctrl, request=request, answer=answer)


def _run_end(evaluation: EvaluationResult | None = None) -> RunEndEvent:
    return RunEndEvent(evaluation=evaluation)


async def _make_optimizer(
    *,
    controllables: list[Controllable],
    observables: list[ObservableValue],
    max_turns: int = 3,
    max_attempts: int = 2,
    max_attempts_per_variant: int = 1,
    success_threshold: float = 0.8,
    max_backtracks: int = 2,
) -> CrescendoOptimizer:
    opt = CrescendoOptimizer(
        max_turns=max_turns,
        max_backtracks=max_backtracks,
        success_threshold=success_threshold,
        max_attempts=max_attempts,
        max_attempts_per_variant=max_attempts_per_variant,
    )
    await opt.initialize(
        goal=Goal(description="test objective"),
        controllables=controllables,
        observables=observables,
        llm_client=AsyncMock(),
    )
    return opt


# ── Bug fix: name-based PreCall dispatch ────────────────────────────────


@pytest.mark.asyncio
async def test_system_prompt_event_first_does_not_lock_primary():
    """ChatbotTarget order: system_prompt PreCall first, then user_message.

    Optimizer must NOT lock onto system_prompt as the primary user-message
    channel. user_message must still receive injections.
    """
    sp_ctrl = _ctrl("system_prompt", tag="system_prompt")
    um_ctrl = _ctrl("user_message", tag="user")
    opt = await _make_optimizer(controllables=[sp_ctrl, um_ctrl], observables=[])
    await opt.on_event(_run_start())

    # System prompt PreCall arrives first; not writable in scope (no
    # system_prompt controllable was passed to indicate writability).
    sp_resp = await opt.on_event(_pre_call(sp_ctrl, "the configured prompt"))
    assert isinstance(sp_resp, ControllableNoInjection)

    # User-message PreCall must still get a real injection.
    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as gen:
        gen.return_value = AttackerOutput(question="Q1", summary="", rationale="r")
        um_resp = await opt.on_event(_pre_call(um_ctrl))
    assert isinstance(um_resp, ControllableInjection)
    assert um_resp.value == "Q1"


@pytest.mark.asyncio
async def test_response_pre_call_is_ignored():
    """Response controllable PreCall (if it ever fires) is never injected."""
    resp_ctrl = _ctrl("response", tag="model")
    um_ctrl = _ctrl("user_message", tag="user")
    opt = await _make_optimizer(controllables=[um_ctrl], observables=[])
    await opt.on_event(_run_start())

    out = await opt.on_event(_pre_call(resp_ctrl))
    assert isinstance(out, ControllableNoInjection)


# ── (a)/(b) static observables flow into attacker ───────────────────────


@pytest.mark.asyncio
async def test_model_observable_passed_to_attacker():
    um = _ctrl("user_message", tag="user")
    obs = [_obs("model", "openai/gpt-4o", tag="model_identity")]
    opt = await _make_optimizer(controllables=[um], observables=obs)
    await opt.on_event(_run_start())
    assert opt._target_model_id == "openai/gpt-4o"
    assert opt._attacker is not None
    # The attacker holds the captured value
    assert opt._attacker._target_model_id == "openai/gpt-4o"


@pytest.mark.asyncio
async def test_system_prompt_observable_passed_to_attacker():
    um = _ctrl("user_message", tag="user")
    obs = [_obs("system_prompt", "You are a helpful X assistant.", tag="sp_readable")]
    opt = await _make_optimizer(controllables=[um], observables=obs)
    await opt.on_event(_run_start())
    assert opt._target_system_prompt == "You are a helpful X assistant."
    assert opt._attacker is not None
    assert opt._attacker._target_system_prompt == "You are a helpful X assistant."


@pytest.mark.asyncio
async def test_attacker_system_prompt_includes_target_context():
    """When observables are visible, the attacker LLM call includes them."""
    um = _ctrl("user_message", tag="user")
    obs = [
        _obs("model", "anthropic/claude-haiku-4.5", tag="model_identity"),
        _obs("system_prompt", "Be terse.", tag="sp_readable"),
    ]
    opt = await _make_optimizer(controllables=[um], observables=obs)
    await opt.on_event(_run_start())

    mock_llm = opt._attacker._llm
    mock_llm.complete = AsyncMock()
    mock_llm.complete.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=(
            '{"generated_question": "Q", '
            '"last_response_summary": "", '
            '"rationale_behind_jailbreak": "r"}'
        )))],
    )
    await opt._attacker.generate_question(
        goal="g", turn=1, max_turns=3,
        last_response=None, last_score=None, last_rationale=None,
    )
    system_text = mock_llm.complete.call_args[0][0][0]["content"]
    assert "anthropic/claude-haiku-4.5" in system_text
    assert "Be terse." in system_text


# ── (c) system_prompt writable controllable + pretext lift ─────────────


@pytest.mark.asyncio
async def test_can_write_system_prompt_flag_set_from_controllables():
    sp = _ctrl("system_prompt", tag="system_prompt")
    um = _ctrl("user_message", tag="user")
    opt = await _make_optimizer(controllables=[sp, um], observables=[])
    assert opt._can_write_system_prompt is True


@pytest.mark.asyncio
async def test_system_prompt_pretext_lift_eager_q1_caching():
    """When system_prompt is writable, on the system_prompt PreCall the
    optimizer eagerly generates (framing, Q1) and injects the framing.
    The cached Q1 is then used on the next user_message PreCall without
    triggering a second attacker call."""
    sp = _ctrl("system_prompt", tag="system_prompt")
    um = _ctrl("user_message", tag="user")
    opt = await _make_optimizer(controllables=[sp, um], observables=[])
    await opt.on_event(_run_start())

    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as gen:
        gen.return_value = AttackerOutput(
            question="Tell me about the history of X.",
            summary="",
            rationale="benign opener",
            framing="I'm a graduate student researching X.",
        )
        sp_resp = await opt.on_event(_pre_call(sp, "configured prompt"))
        # Framing was lifted to the system prompt slot
        assert isinstance(sp_resp, ControllableInjection)
        assert sp_resp.value == "I'm a graduate student researching X."
        assert gen.call_count == 1
        # gen was called with include_framing=True
        assert gen.call_args.kwargs["include_framing"] is True

        # Q1 is cached and used on next user_message PreCall — no new gen call
        um_resp = await opt.on_event(_pre_call(um))
        assert isinstance(um_resp, ControllableInjection)
        assert um_resp.value == "Tell me about the history of X."
        assert gen.call_count == 1  # still only one gen call


@pytest.mark.asyncio
async def test_system_prompt_writable_but_attacker_returns_no_framing():
    """If the attacker fails to produce a framing, system_prompt event
    falls back to NoInjection (task-configured prompt is preserved).
    The user_message turn 1 then generates Q1 via a fresh attacker call."""
    sp = _ctrl("system_prompt", tag="system_prompt")
    um = _ctrl("user_message", tag="user")
    opt = await _make_optimizer(controllables=[sp, um], observables=[])
    await opt.on_event(_run_start())

    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as gen:
        # First call: attacker raises (simulating malformed output / parse error)
        gen.side_effect = [
            ValueError("framing missing"),
            AttackerOutput(question="Q1 fallback", summary="", rationale="r"),
        ]
        sp_resp = await opt.on_event(_pre_call(sp))
        assert isinstance(sp_resp, ControllableNoInjection)
        # Q1 is NOT cached
        assert opt._pending_q1 is None

        um_resp = await opt.on_event(_pre_call(um))
        assert isinstance(um_resp, ControllableInjection)
        assert um_resp.value == "Q1 fallback"


@pytest.mark.asyncio
async def test_system_prompt_writable_only_first_event_lifted():
    """Only the FIRST system_prompt PreCall in an attempt lifts the framing.
    Defensive: subsequent system_prompt events get NoInjection."""
    sp = _ctrl("system_prompt", tag="system_prompt")
    um = _ctrl("user_message", tag="user")
    opt = await _make_optimizer(controllables=[sp, um], observables=[])
    await opt.on_event(_run_start())

    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as gen:
        gen.return_value = AttackerOutput(
            question="Q1", summary="", rationale="r", framing="framing-text",
        )
        first = await opt.on_event(_pre_call(sp))
        assert isinstance(first, ControllableInjection)
        second = await opt.on_event(_pre_call(sp))
        assert isinstance(second, ControllableNoInjection)


@pytest.mark.asyncio
async def test_system_prompt_not_writable_returns_no_injection():
    """When system_prompt is not in the writable controllables list, the
    optimizer must respond NoInjection — even if a system_prompt PreCall
    somehow arrives."""
    um = _ctrl("user_message", tag="user")
    sp_ctrl = _ctrl("system_prompt", tag="system_prompt")
    opt = await _make_optimizer(controllables=[um], observables=[])  # no sp
    await opt.on_event(_run_start())

    out = await opt.on_event(_pre_call(sp_ctrl))
    assert isinstance(out, ControllableNoInjection)


# ── (d) Dual evaluator: task verdict authoritative ─────────────────────


@pytest.mark.asyncio
async def test_task_evaluator_overrides_internal_success():
    """Internal said success but task says failure → not done; record failure."""
    um = _ctrl("user_message", tag="user")
    opt = await _make_optimizer(
        controllables=[um], observables=[], max_attempts=1, max_attempts_per_variant=2,
    )
    await opt.on_event(_run_start())

    # Force internal success
    opt._succeeded = True

    eval_result = EvaluationResult(
        success=False,
        primary_score=Score(value=0.1, name="primary"),
        rationale="Task judge: not actually achieved.",
    )
    resp = await opt.on_event(_run_end(eval_result))

    # Task verdict (False) overrides internal True; another attempt available.
    assert resp.done is False
    assert opt._succeeded is False
    # Failure record captured task rationale
    assert len(opt._variant_failure_log) == 1
    assert opt._variant_failure_log[0].task_rationale == "Task judge: not actually achieved."


@pytest.mark.asyncio
async def test_task_evaluator_promotes_internal_failure_to_success():
    """Internal said failure but task says success → done=True."""
    um = _ctrl("user_message", tag="user")
    opt = await _make_optimizer(controllables=[um], observables=[])
    await opt.on_event(_run_start())

    opt._succeeded = False

    eval_result = EvaluationResult(
        success=True,
        primary_score=Score(value=1.0, name="primary"),
        rationale="Task judge: succeeded.",
    )
    resp = await opt.on_event(_run_end(eval_result))

    assert resp.done is True
    assert opt._succeeded is True


@pytest.mark.asyncio
async def test_no_evaluation_keeps_internal_verdict():
    """When include_feedback=False on the controller, RunEndEvent.evaluation
    is None; internal verdict stands."""
    um = _ctrl("user_message", tag="user")
    opt = await _make_optimizer(controllables=[um], observables=[])
    await opt.on_event(_run_start())

    opt._succeeded = True
    resp = await opt.on_event(_run_end(None))
    assert resp.done is True


# ── (e) Cross-attempt within variant: failure log carried, variants siloed ──


@pytest.mark.asyncio
async def test_variant_attempt_increments_within_variant():
    """First attempt fails → next run is same variant, attempt+1, with failure log."""
    um = _ctrl("user_message", tag="user")
    opt = await _make_optimizer(
        controllables=[um], observables=[], max_attempts=2, max_attempts_per_variant=3,
    )
    await opt.on_event(_run_start())

    # Failed attempt 1
    opt._succeeded = False
    opt._last_score = 0.2
    opt._last_rationale = "low progress"

    resp = await opt.on_event(_run_end(EvaluationResult(
        success=False, primary_score=Score(value=0.2), rationale="judge: refused often",
    )))
    assert resp.done is False
    assert opt._variant_index == 0
    assert opt._variant_attempt == 1
    assert len(opt._variant_failure_log) == 1
    assert opt._variant_failure_log[0].attempt_number == 1
    # No refusals in this stub attempt → no first_refused_question recorded
    assert opt._variant_failure_log[0].first_refused_question is None
    assert opt._variant_failure_log[0].task_rationale == "judge: refused often"

    # New attacker on RunStart receives failure log
    await opt.on_event(_run_start())
    assert opt._attacker is not None
    assert opt._attacker._previous_failures == tuple(opt._variant_failure_log)


@pytest.mark.asyncio
async def test_variant_rotation_clears_failure_log():
    """When all attempts in a variant are exhausted, rotate variant and clear log."""
    um = _ctrl("user_message", tag="user")
    opt = await _make_optimizer(
        controllables=[um], observables=[], max_attempts=2, max_attempts_per_variant=2,
    )
    # First failed attempt of variant 0
    opt._succeeded = False
    await opt.on_event(_run_start())
    await opt.on_event(_run_end(EvaluationResult(
        success=False, primary_score=Score(value=0.0), rationale="r1",
    )))
    assert opt._variant_index == 0
    assert opt._variant_attempt == 1
    assert len(opt._variant_failure_log) == 1

    # Second failed attempt of variant 0 — rotates to variant 1, clears log
    await opt.on_event(_run_start())
    await opt.on_event(_run_end(EvaluationResult(
        success=False, primary_score=Score(value=0.0), rationale="r2",
    )))
    assert opt._variant_index == 1
    assert opt._variant_attempt == 0
    assert opt._variant_failure_log == []

    # New attacker for variant 1 has empty failure log (variants are siloed)
    await opt.on_event(_run_start())
    assert opt._attacker._previous_failures == ()


@pytest.mark.asyncio
async def test_all_variants_and_attempts_exhausted_signals_done():
    um = _ctrl("user_message", tag="user")
    opt = await _make_optimizer(
        controllables=[um], observables=[], max_attempts=1, max_attempts_per_variant=2,
    )
    opt._succeeded = False
    await opt.on_event(_run_start())
    resp1 = await opt.on_event(_run_end(EvaluationResult(
        success=False, primary_score=Score(value=0.0), rationale="r",
    )))
    assert resp1.done is False  # one more attempt within variant 0

    await opt.on_event(_run_start())
    resp2 = await opt.on_event(_run_end(EvaluationResult(
        success=False, primary_score=Score(value=0.0), rationale="r",
    )))
    assert resp2.done is True  # variant exhausted AND no more variants


@pytest.mark.asyncio
async def test_failure_record_captures_first_refused_question():
    """First refused question + task rationale must end up in the FailureRecord
    on attempt end. Subsequent refusals are not separately tracked."""
    um = _ctrl("user_message", tag="user")
    opt = await _make_optimizer(
        controllables=[um], observables=[], max_attempts=1, max_attempts_per_variant=2,
    )
    await opt.on_event(_run_start())

    # Simulate: attacker had two refused questions during the attempt
    opt._attacker.notify_refusal("Refused Q A")
    opt._attacker.notify_refusal("Refused Q B")

    await opt.on_event(_run_end(EvaluationResult(
        success=False, primary_score=Score(value=0.0), rationale="judge: refused.",
    )))

    record = opt._variant_failure_log[0]
    assert record.first_refused_question == "Refused Q A"
    assert record.task_rationale == "judge: refused."


@pytest.mark.asyncio
async def test_failure_record_no_refusal_records_none():
    """If attempt had no refusals, first_refused_question must be None."""
    um = _ctrl("user_message", tag="user")
    opt = await _make_optimizer(
        controllables=[um], observables=[], max_attempts=1, max_attempts_per_variant=2,
    )
    await opt.on_event(_run_start())
    # No notify_refusal calls
    await opt.on_event(_run_end(EvaluationResult(
        success=False, primary_score=Score(value=0.0), rationale="judge: not enough.",
    )))
    record = opt._variant_failure_log[0]
    assert record.first_refused_question is None
    assert record.task_rationale == "judge: not enough."


# ── (e) interaction with attacker rendering ───────────────────────────


@pytest.mark.asyncio
async def test_attacker_rendered_prompt_includes_failure_log_on_retry():
    """On the second attempt within a variant, the attacker's system prompt
    must contain the previous-attempts block."""
    um = _ctrl("user_message", tag="user")
    opt = await _make_optimizer(
        controllables=[um], observables=[], max_attempts=1, max_attempts_per_variant=2,
    )
    opt._succeeded = False
    await opt.on_event(_run_start())
    opt._attacker.notify_refusal("Don't ask this directly")
    await opt.on_event(_run_end(EvaluationResult(
        success=False, primary_score=Score(value=0.0), rationale="too direct",
    )))

    # Second attempt: new attacker built with failure log
    await opt.on_event(_run_start())
    assert opt._attacker._previous_failures
    record = opt._attacker._previous_failures[0]
    assert record.first_refused_question == "Don't ask this directly"

    # Render the attacker's system prompt and check the failure log appears
    mock_llm = opt._attacker._llm
    mock_llm.complete = AsyncMock()
    mock_llm.complete.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=(
            '{"generated_question": "Q", '
            '"last_response_summary": "", '
            '"rationale_behind_jailbreak": "r"}'
        )))],
    )
    await opt._attacker.generate_question(
        goal="g", turn=1, max_turns=3,
        last_response=None, last_score=None, last_rationale=None,
    )
    system_text = mock_llm.complete.call_args[0][0][0]["content"]
    assert "Previous Attempts" in system_text
    assert "Don't ask this directly" in system_text
    assert "too direct" in system_text


# ── End-to-end: ChatbotTarget-style three-controllable scope flow ───────


@pytest.mark.asyncio
async def test_full_chatbot_phase1_phase2_flow_with_writable_sp():
    """Exercises the full ChatbotTarget event order:
       sp PreCall → user_message PreCall → response PostCall → ...
    with all three controllables in scope. Verifies framing is lifted,
    Q1 is cached, response post-calls are processed normally."""
    sp = _ctrl("system_prompt", tag="system_prompt")
    um = _ctrl("user_message", tag="user")
    resp = _ctrl("response", tag="model")
    opt = await _make_optimizer(
        controllables=[sp, um, resp], observables=[],
        max_turns=2, max_attempts=1, success_threshold=0.99,
    )
    await opt.on_event(_run_start())

    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as gen:
        gen.return_value = AttackerOutput(
            question="Q1", summary="", rationale="r", framing="framing-text",
        )
        # Phase 1: system prompt (lifted)
        sp_out = await opt.on_event(_pre_call(sp))
        assert isinstance(sp_out, ControllableInjection)
        assert sp_out.value == "framing-text"

    # Phase 2 turn 1: user_message uses cached Q1
    um_out = await opt.on_event(_pre_call(um))
    assert isinstance(um_out, ControllableInjection)
    assert um_out.value == "Q1"

    # Phase 2 turn 1 PostCall (response controllable carrying the answer)
    post_out = await opt.on_event(_post_call(resp, "answer 1", request="Q1"))
    assert isinstance(post_out, ControllableNoInjection)

    # Phase 2 turn 2: user_message — fresh attacker call
    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as gen2, \
         patch.object(opt._evaluator, "is_refusal", new_callable=AsyncMock) as ref, \
         patch.object(opt._evaluator, "score_response", new_callable=AsyncMock) as score:
        gen2.return_value = AttackerOutput(question="Q2", summary="", rationale="r")
        ref.return_value = False
        score.return_value = (0.4, "low")
        out2 = await opt.on_event(_pre_call(um))
    assert isinstance(out2, ControllableInjection)
    assert out2.value == "Q2"
    assert opt._turn == 1
    assert opt._last_response == "answer 1"


@pytest.mark.asyncio
async def test_full_chatbot_flow_no_writable_sp_baseline():
    """Same target shape but system_prompt NOT in writable list (readable
    only). Optimizer must NoInjection on system_prompt event and run
    plain Crescendo on user_message."""
    sp = _ctrl("system_prompt", tag="system_prompt_readable_zone")
    um = _ctrl("user_message", tag="user")
    # Note: system_prompt controllable not provided, simulating readable-only
    # scope (the system_prompt PreCall would still arrive if the controller
    # filter let it through; in practice it would be filtered out, but we
    # test defensively).
    opt = await _make_optimizer(controllables=[um], observables=[])
    await opt.on_event(_run_start())

    sp_out = await opt.on_event(_pre_call(sp))
    assert isinstance(sp_out, ControllableNoInjection)

    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as gen:
        gen.return_value = AttackerOutput(question="Q1", summary="", rationale="r")
        # gen is called with include_framing=False (no eager call happened)
        um_out = await opt.on_event(_pre_call(um))
        assert isinstance(um_out, ControllableInjection)
        assert um_out.value == "Q1"
        assert gen.call_args.kwargs["include_framing"] is False


# ── Replay-based pseudo-backtrack ─────────────────────────────────────


@pytest.mark.asyncio
async def test_replay_plan_built_from_successful_prefix_when_refusal_occurred():
    """After a failed attempt with at least one refusal, the optimizer
    queues a replay plan containing the consecutive successful prefix
    captured by _process_answer."""
    um = _ctrl("user_message", tag="user")
    opt = await _make_optimizer(
        controllables=[um], observables=[], max_attempts=1, max_attempts_per_variant=2,
    )
    await opt.on_event(_run_start())

    # Manually populate the attempt's successful injections (simulate two
    # turns that scored, before a refusal-then-terminal at turn 3).
    opt._attempt_injections = [
        TurnRecord(
            injected_question="Q1",
            target_response="A1",
            score=0.2,
            rationale="r1",
        ),
        TurnRecord(
            injected_question="Q2",
            target_response="A2",
            score=0.5,
            rationale="r2",
        ),
    ]
    opt._attacker.notify_refusal("Q3 (refused)")

    await opt.on_event(_run_end(EvaluationResult(
        success=False, primary_score=Score(value=0.5), rationale="too cautious",
    )))

    # Plan should have moved to pending state.
    assert opt._pending_replay_plan is not None
    assert opt._pending_replay_plan.framing is None  # (c) wasn't used here
    assert len(opt._pending_replay_plan.successful_turns) == 2
    assert opt._pending_replay_plan.successful_turns[0].injected_question == "Q1"
    assert opt._pending_replay_plan.successful_turns[1].injected_question == "Q2"


@pytest.mark.asyncio
async def test_replay_plan_not_built_when_no_refusal_occurred():
    """If the attempt completed without any refusals, no replay plan is
    queued — the next attempt starts fresh (escalation completed cleanly
    but task judge said fail; replaying the same prefix would just hit
    the same task verdict)."""
    um = _ctrl("user_message", tag="user")
    opt = await _make_optimizer(
        controllables=[um], observables=[], max_attempts=1, max_attempts_per_variant=2,
    )
    await opt.on_event(_run_start())
    opt._attempt_injections = [
        TurnRecord(injected_question="Q1", target_response="A1", score=0.7, rationale="r"),
    ]
    # No notify_refusal calls

    await opt.on_event(_run_end(EvaluationResult(
        success=False, primary_score=Score(value=0.7), rationale="task: not enough",
    )))
    assert opt._pending_replay_plan is None


@pytest.mark.asyncio
async def test_replay_consumed_by_next_attempt_state():
    """At RunStart of the next attempt within a variant, the pending replay
    plan is consumed: replay queue populated, pending_replay_plan cleared."""
    um = _ctrl("user_message", tag="user")
    opt = await _make_optimizer(
        controllables=[um], observables=[], max_attempts=1, max_attempts_per_variant=2,
    )
    await opt.on_event(_run_start())
    opt._attempt_injections = [
        TurnRecord(injected_question="Q1", target_response="A1", score=0.4, rationale="r"),
    ]
    opt._attacker.notify_refusal("refused-Q2")
    await opt.on_event(_run_end(EvaluationResult(
        success=False, primary_score=Score(value=0.4), rationale="r",
    )))
    assert opt._pending_replay_plan is not None

    # Next run: plan is consumed, replay queue populated.
    await opt.on_event(_run_start())
    assert opt._pending_replay_plan is None
    assert len(opt._replay_iter) == 1
    assert opt._replay_iter[0].injected_question == "Q1"


@pytest.mark.asyncio
async def test_replay_injects_cached_questions_without_calling_attacker():
    """During replay, user_message PreCalls inject cached questions and
    feedback consumption uses cached scores — no attacker or evaluator
    calls until the replay queue is exhausted."""
    um = _ctrl("user_message", tag="user")
    opt = await _make_optimizer(
        controllables=[um], observables=[], max_attempts=1,
        max_attempts_per_variant=2, max_turns=5,
    )
    # Stuff a replay plan in directly (skip building it via run_end).
    opt._pending_replay_plan = ReplayPlan(
        framing=None,
        successful_turns=(
            TurnRecord("cached-Q1", "cached-A1", 0.2, "r1"),
            TurnRecord("cached-Q2", "cached-A2", 0.4, "r2"),
        ),
    )
    opt._variant_attempt = 1  # second attempt within the variant

    await opt.on_event(_run_start())

    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as gen, \
         patch.object(opt._evaluator, "is_refusal", new_callable=AsyncMock) as ref, \
         patch.object(opt._evaluator, "score_response", new_callable=AsyncMock) as score:
        # Turn 1: replayed, attacker NOT called
        out1 = await opt.on_event(_pre_call(um))
        assert isinstance(out1, ControllableInjection)
        assert out1.value == "cached-Q1"
        assert gen.call_count == 0
        assert ref.call_count == 0
        assert score.call_count == 0

        # Turn 2: replayed, attacker NOT called; feedback for turn 1 also
        # comes from the cache (no evaluator call).
        out2 = await opt.on_event(_pre_call(um))
        assert isinstance(out2, ControllableInjection)
        assert out2.value == "cached-Q2"
        assert gen.call_count == 0
        assert ref.call_count == 0
        assert score.call_count == 0
        assert opt._turn == 1
        assert opt._last_response == "cached-A1"
        assert opt._last_score == 0.2

        # Turn 3: replay queue empty → attacker IS called now (with last_response
        # set from the last replayed turn).
        gen.return_value = AttackerOutput(question="fresh-Q3", summary="", rationale="r")
        ref.return_value = False
        score.return_value = (0.6, "progress")
        out3 = await opt.on_event(_pre_call(um))
        assert isinstance(out3, ControllableInjection)
        assert out3.value == "fresh-Q3"
        assert gen.call_count == 1
        # Attacker received last_response from cached turn 2
        assert gen.call_args.kwargs["last_response"] == "cached-A2"
        assert gen.call_args.kwargs["last_score"] == 0.4


@pytest.mark.asyncio
async def test_replay_carries_forward_into_next_attempts_injection_log():
    """Replayed turns must end up in the new attempt's _attempt_injections
    so that a third attempt's replay plan covers the full prefix."""
    um = _ctrl("user_message", tag="user")
    opt = await _make_optimizer(
        controllables=[um], observables=[], max_attempts=1,
        max_attempts_per_variant=3, max_turns=5,
    )
    opt._pending_replay_plan = ReplayPlan(
        framing=None,
        successful_turns=(
            TurnRecord("cached-Q1", "cached-A1", 0.3, "r1"),
        ),
    )
    opt._variant_attempt = 1
    await opt.on_event(_run_start())

    # Turn 1 replayed
    await opt.on_event(_pre_call(um))
    # Turn 2: attacker call, feedback from turn 1 (cached) consumed first
    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as gen, \
         patch.object(opt._evaluator, "is_refusal", new_callable=AsyncMock) as ref, \
         patch.object(opt._evaluator, "score_response", new_callable=AsyncMock) as score:
        gen.return_value = AttackerOutput(question="fresh-Q2", summary="", rationale="r")
        ref.return_value = False
        score.return_value = (0.4, "ok")
        await opt.on_event(_pre_call(um))

    # The replayed turn was carried into _attempt_injections
    assert len(opt._attempt_injections) == 1
    assert opt._attempt_injections[0].injected_question == "cached-Q1"


@pytest.mark.asyncio
async def test_replay_framing_lifted_on_system_prompt_event():
    """When replay plan carries a framing, the system_prompt PreCall
    injects it without calling the attacker."""
    sp = _ctrl("system_prompt", tag="system_prompt")
    um = _ctrl("user_message", tag="user")
    opt = await _make_optimizer(controllables=[sp, um], observables=[])
    opt._pending_replay_plan = ReplayPlan(
        framing="cached-framing-text",
        successful_turns=(),
    )
    opt._variant_attempt = 1
    await opt.on_event(_run_start())

    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as gen:
        sp_out = await opt.on_event(_pre_call(sp))
        # Framing replayed verbatim, attacker NOT called for framing
        assert isinstance(sp_out, ControllableInjection)
        assert sp_out.value == "cached-framing-text"
        assert gen.call_count == 0


@pytest.mark.asyncio
async def test_failure_log_omits_restored_sentence_on_fresh_retry():
    """When the prior attempt failed without any refusals (no replay
    plan queued), the next attempt's attacker prompt must NOT claim
    the conversation has been restored — that retry actually starts
    from turn 1 against a fresh target."""
    um = _ctrl("user_message", tag="user")
    opt = await _make_optimizer(
        controllables=[um], observables=[],
        max_attempts=1, max_attempts_per_variant=2,
    )
    await opt.on_event(_run_start())
    # Failed attempt with no refusals — task judge rejected on goal grounds
    await opt.on_event(_run_end(EvaluationResult(
        success=False, primary_score=Score(value=0.3),
        rationale="answer was off-target",
    )))
    assert opt._pending_replay_plan is None  # fresh retry
    await opt.on_event(_run_start())

    mock_llm = opt._attacker._llm
    mock_llm.complete = AsyncMock()
    mock_llm.complete.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=(
            '{"generated_question": "Q", '
            '"last_response_summary": "", '
            '"rationale_behind_jailbreak": "r"}'
        )))],
    )
    await opt._attacker.generate_question(
        goal="g", turn=1, max_turns=3,
        last_response=None, last_score=None, last_rationale=None,
    )
    system_text = mock_llm.complete.call_args[0][0][0]["content"]
    assert "Previous Attempts" in system_text
    assert "restored" not in system_text
    assert "starting from turn 1" in system_text


@pytest.mark.asyncio
async def test_failure_log_keeps_restored_sentence_on_replay_retry():
    """When the prior attempt hit a refusal and a replay plan was queued,
    the next attempt's attacker prompt must include the 'restored'
    sentence — that retry does in fact replay the prefix."""
    um = _ctrl("user_message", tag="user")
    opt = await _make_optimizer(
        controllables=[um], observables=[],
        max_attempts=1, max_attempts_per_variant=2,
    )
    await opt.on_event(_run_start())
    opt._attempt_injections = [
        TurnRecord("Q1", "A1", 0.4, "r1"),
    ]
    opt._attacker.notify_refusal("refused-Q2")
    await opt.on_event(_run_end(EvaluationResult(
        success=False, primary_score=Score(value=0.4), rationale="r",
    )))
    assert opt._pending_replay_plan is not None
    await opt.on_event(_run_start())

    mock_llm = opt._attacker._llm
    mock_llm.complete = AsyncMock()
    mock_llm.complete.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=(
            '{"generated_question": "Q", '
            '"last_response_summary": "", '
            '"rationale_behind_jailbreak": "r"}'
        )))],
    )
    await opt._attacker.generate_question(
        goal="g", turn=1, max_turns=3,
        last_response=None, last_score=None, last_rationale=None,
    )
    system_text = mock_llm.complete.call_args[0][0][0]["content"]
    assert "restored" in system_text
    assert "starting from turn 1" not in system_text


@pytest.mark.asyncio
async def test_task_override_of_internal_success_does_not_queue_replay():
    """When the internal judge concluded success mid-conversation but
    the task judge overrules to failure at run end, the cached prefix
    is the exact transcript the task judge already rejected. Replaying
    it would just reproduce the same verdict and burn an attempt. The
    next attempt should start fresh with the failure log."""
    um = _ctrl("user_message", tag="user")
    opt = await _make_optimizer(
        controllables=[um], observables=[],
        max_attempts=1, max_attempts_per_variant=2,
    )
    await opt.on_event(_run_start())

    # Internal judge said the attempt succeeded, with prior backtrack
    # refusal and accumulated successful turns.
    opt._succeeded = True
    opt._attempt_injections = [
        TurnRecord("Q1", "A1", 0.4, "r1"),
        TurnRecord("Q2", "A2", 0.9, "r2"),
    ]
    opt._attacker.notify_refusal("Q-mid-refused")

    await opt.on_event(_run_end(EvaluationResult(
        success=False, primary_score=Score(value=0.1),
        rationale="judge: answer was generic, goal not met",
    )))

    assert opt._succeeded is False
    # Failure log captured for next attempt
    assert len(opt._variant_failure_log) == 1
    assert opt._variant_failure_log[0].task_rationale == (
        "judge: answer was generic, goal not met"
    )
    # No replay plan: task already rejected this prefix
    assert opt._pending_replay_plan is None


@pytest.mark.asyncio
async def test_replay_attempt_with_no_framing_does_not_re_invoke_eager_call():
    """Replay plan with framing=None means the prior attempt did NOT
    install a persona on the target's system prompt (either capability
    (c) was unused, or the eager attacker call had failed). Faithful
    replay must reproduce that context: leave the system prompt at
    default. Firing a fresh eager call (which could succeed this time)
    would inject a persona the cached turns never saw, desynchronising
    replayed responses from what the live target now produces."""
    sp = _ctrl("system_prompt", tag="system_prompt")
    um = _ctrl("user_message", tag="user")
    opt = await _make_optimizer(controllables=[sp, um], observables=[])

    # Stage a replay plan with cached turns but no framing — simulates
    # a prior attempt where (c) was in scope but the eager call failed,
    # so turns ran against the task-default system prompt until refusal.
    opt._pending_replay_plan = ReplayPlan(
        framing=None,
        successful_turns=(
            TurnRecord("cached-Q1", "cached-A1", 0.3, "r1"),
        ),
    )
    opt._variant_attempt = 1
    await opt.on_event(_run_start())

    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as gen:
        sp_out = await opt.on_event(_pre_call(sp))

    assert isinstance(sp_out, ControllableNoInjection)
    # No eager attacker call: the system prompt stays at task-default,
    # matching the prior attempt's actual target state.
    assert gen.call_count == 0


@pytest.mark.asyncio
async def test_replay_plan_captures_attempt_framing():
    """After a (c)-using attempt fails with refusal, the replay plan
    carries the framing forward."""
    sp = _ctrl("system_prompt", tag="system_prompt")
    um = _ctrl("user_message", tag="user")
    opt = await _make_optimizer(
        controllables=[sp, um], observables=[],
        max_attempts=1, max_attempts_per_variant=2,
    )
    await opt.on_event(_run_start())

    # Phase 1: framing eagerly injected
    with patch.object(opt._attacker, "generate_question", new_callable=AsyncMock) as gen:
        gen.return_value = AttackerOutput(
            question="Q1", summary="", rationale="r", framing="framing-A",
        )
        await opt.on_event(_pre_call(sp))

    # Simulate one successful turn + a refusal
    opt._attempt_injections = [
        TurnRecord("Q1", "A1", 0.3, "r"),
    ]
    opt._attacker.notify_refusal("refused-Q2")

    await opt.on_event(_run_end(EvaluationResult(
        success=False, primary_score=Score(value=0.3), rationale="not enough",
    )))

    assert opt._pending_replay_plan is not None
    assert opt._pending_replay_plan.framing == "framing-A"
    assert opt._pending_replay_plan.successful_turns[0].injected_question == "Q1"


@pytest.mark.asyncio
async def test_terminal_refusal_locks_attempt_injections():
    """Once the within-attempt backtrack budget is exhausted (refusal
    accepted as turn outcome), subsequent successful turns must not be
    added to _attempt_injections — replaying past a poisoned turn would
    desynchronise from the target's actual conversation."""
    um = _ctrl("user_message", tag="user")
    opt = await _make_optimizer(
        controllables=[um], observables=[], max_backtracks=0, max_turns=5,
    )
    await opt.on_event(_run_start())

    # Drive one turn through _process_answer that hits backtracks-exhausted
    # path: backtrack_count is already at the limit (0), so a refusal will
    # be accepted as the turn outcome.
    opt._current_question = "Q1"
    with patch.object(opt._evaluator, "is_refusal", new_callable=AsyncMock) as ref, \
         patch.object(opt._evaluator, "score_response", new_callable=AsyncMock) as score:
        ref.return_value = True
        score.return_value = (0.0, "refused")
        await opt._process_answer("I can't help with that.")

    assert opt._terminal_refusal_occurred is True
    assert opt._attempt_injections == []  # nothing recorded for the refused turn

    # A subsequent (a)-path turn must NOT be appended either.
    opt._current_question = "Q2"
    with patch.object(opt._evaluator, "is_refusal", new_callable=AsyncMock) as ref, \
         patch.object(opt._evaluator, "score_response", new_callable=AsyncMock) as score:
        ref.return_value = False
        score.return_value = (0.6, "engaged")
        await opt._process_answer("Some content.")
    assert opt._attempt_injections == []  # locked


@pytest.mark.asyncio
async def test_variant_rotation_clears_pending_replay_plan():
    """Replay plan must not leak across variants — variants are siloed."""
    um = _ctrl("user_message", tag="user")
    opt = await _make_optimizer(
        controllables=[um], observables=[],
        max_attempts=2, max_attempts_per_variant=1,
    )
    await opt.on_event(_run_start())
    opt._attempt_injections = [
        TurnRecord("Q1", "A1", 0.3, "r"),
    ]
    opt._attacker.notify_refusal("refused-Q2")

    await opt.on_event(_run_end(EvaluationResult(
        success=False, primary_score=Score(value=0.3), rationale="r",
    )))

    # max_attempts_per_variant=1, so this immediately rotates variant.
    assert opt._variant_index == 1
    assert opt._variant_attempt == 0
    # Replay plan must be cleared on rotation
    assert opt._pending_replay_plan is None
    assert opt._variant_failure_log == []
