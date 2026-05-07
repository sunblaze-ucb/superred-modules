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

from crescendo_optimizer.attacker import AttackerOutput
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
) -> CrescendoOptimizer:
    opt = CrescendoOptimizer(
        max_turns=max_turns,
        max_backtracks=2,
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
    assert opt._target_model_id == "openai/gpt-4o"
    assert opt._attacker is not None
    # The attacker holds the captured value
    assert opt._attacker._target_model_id == "openai/gpt-4o"


@pytest.mark.asyncio
async def test_system_prompt_observable_passed_to_attacker():
    um = _ctrl("user_message", tag="user")
    obs = [_obs("system_prompt", "You are a helpful X assistant.", tag="sp_readable")]
    opt = await _make_optimizer(controllables=[um], observables=obs)
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
    assert opt._variant_failure_log[0].final_score == 0.2

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
async def test_failure_record_captures_refused_questions():
    """Refused questions accumulated by attacker.notify_refusal must end up
    in the FailureRecord on attempt end."""
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
    assert record.refused_questions == ("Refused Q A", "Refused Q B")
    assert record.task_rationale == "judge: refused."


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
    assert "Don't ask this directly" in record.refused_questions

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
