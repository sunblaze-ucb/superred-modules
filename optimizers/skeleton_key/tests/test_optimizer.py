"""Contract tests for SkeletonKeyOptimizer against superred's event model.

Skeleton Key is prompt-only and deterministic: no model calls, no payload text
in fixtures. The vendored preamble/acceptance bodies are loaded through the
module's own accessors and checked *by position* against a synthetic goal marker;
they are never printed or asserted by content.
"""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock

from superred.core.types.controllable import Controllable
from superred.core.types.events import (
    ControllableInjection,
    ControllableNoInjection,
    ControllablePostCallEvent,
    ControllablePreCallEvent,
    RunEndEvent,
    RunStartEvent,
)
from superred.core.types.goal import Goal
from superred.core.types.security_domain import SecurityDomainTag

from skeleton_key_optimizer import SkeletonKeyOptimizer, render

USER = SecurityDomainTag("user")
GOAL = "SYNTHETIC_GOAL_MARKER_42"


def _ctrl(name="user", value_type="text"):
    return Controllable(name=name, security_domain=USER, value_type=value_type)


def _mock_llm() -> MagicMock:
    llm = MagicMock()
    llm.complete = AsyncMock(side_effect=AssertionError("Skeleton Key must not call the LLM"))
    return llm


def _make(llm=None, controllables=None, **kw):
    opt = SkeletonKeyOptimizer(**kw)
    asyncio.run(
        opt.initialize(
            goal=Goal(description=GOAL),
            controllables=controllables if controllables is not None else [_ctrl()],
            observables=[],
            llm_client=llm if llm is not None else _mock_llm(),
        )
    )
    return opt


def _start(o):
    asyncio.run(o.on_event(RunStartEvent(trajectory=MagicMock())))


def _pre(o, n="user", value_type="text"):
    return asyncio.run(
        o.on_event(ControllablePreCallEvent(controllable=_ctrl(n, value_type), request=""))
    )


def _post(o, request, answer, n="user"):
    return asyncio.run(
        o.on_event(ControllablePostCallEvent(controllable=_ctrl(n), request=request, answer=answer))
    )


def _end(o):
    return asyncio.run(o.on_event(RunEndEvent(evaluation=None)))


# --- imports / packaging ---------------------------------------------------


def test_module_imports_without_heavy_deps() -> None:
    for mod in ("torch", "transformers", "nltk", "pandas", "pyrit", "sentence_transformers"):
        assert mod not in sys.modules


# --- render() structure (synthetic strings only) ---------------------------


def test_render_orders_preamble_then_acceptance_then_goal() -> None:
    out = render("OBJECTIVE_X", "PREAMBLE_X", "ACCEPT_X")
    assert out.index("PREAMBLE_X") < out.index("ACCEPT_X") < out.index("OBJECTIVE_X")


def test_render_uses_turn_transcript_framing() -> None:
    out = render("OBJECTIVE_X", "PREAMBLE_X", "ACCEPT_X")
    # PyRIT ConversationContextNormalizer no-editable-history transcript.
    assert "Turn 1:" in out and "Turn 2:" in out
    assert out.index("Turn 1:") < out.index("PREAMBLE_X")
    assert out.index("ACCEPT_X") < out.index("Turn 2:") < out.index("OBJECTIVE_X")
    assert "user: PREAMBLE_X" in out
    assert "assistant: ACCEPT_X" in out
    assert "user: OBJECTIVE_X" in out


# --- faithfulness: vendored preamble precedes the goal ---------------------


def test_injection_places_skeleton_key_preamble_before_goal() -> None:
    opt = _make()
    _start(opt)
    resp = _pre(opt)
    assert isinstance(resp, ControllableInjection)
    payload = resp.value
    # Loaded from the vendored seed files (never printed here).
    preamble = opt.skeleton_key_prompt
    acceptance = opt.skeleton_key_acceptance
    assert preamble and acceptance and preamble != acceptance
    i_pre = payload.index(preamble)
    i_acc = payload.index(acceptance)
    i_goal = payload.index(GOAL)
    assert i_pre < i_acc < i_goal  # preamble, then acceptance, then objective


def test_payload_property_matches_injected_value() -> None:
    opt = _make()
    _start(opt)
    resp = _pre(opt)
    assert isinstance(resp, ControllableInjection)
    assert resp.value == opt.payload


def test_vendored_bodies_are_loaded_and_distinct() -> None:
    opt = _make()
    assert opt.skeleton_key_prompt.strip()
    assert opt.skeleton_key_acceptance.strip()
    assert opt.skeleton_key_prompt != opt.skeleton_key_acceptance


# --- event contract --------------------------------------------------------


def test_first_precall_injects_then_declines() -> None:
    opt = _make()
    _start(opt)
    assert isinstance(_pre(opt), ControllableInjection)
    # A second surface in the same run gets nothing (one self-contained turn).
    assert isinstance(_pre(opt), ControllableNoInjection)


def test_post_call_returns_no_injection() -> None:
    opt = _make()
    _start(opt)
    _pre(opt)
    resp = _post(opt, "req", "ANSWER")
    assert isinstance(resp, ControllableNoInjection)


def test_never_injects_system_prompt() -> None:
    opt = _make()
    _start(opt)
    assert isinstance(_pre(opt, "system_prompt"), ControllableNoInjection)


def test_declines_non_free_text_surface() -> None:
    opt = _make(controllables=[_ctrl("payload", value_type="json")])
    _start(opt)
    assert isinstance(_pre(opt, "payload", value_type="json"), ControllableNoInjection)


# --- run lifecycle / re-arm ------------------------------------------------


def test_single_attempt_is_done_after_one_run() -> None:
    opt = _make()  # default max_attempts_on_failure=0 -> one send
    _start(opt)
    _pre(opt)
    assert _end(opt).done is True


def test_runstart_rearms_state_across_two_runs() -> None:
    opt = _make(max_attempts_on_failure=1)  # 2 sends across 2 runs
    _start(opt)
    assert isinstance(_pre(opt), ControllableInjection)
    assert opt._injected is True
    assert _end(opt).done is False  # retry budget remains
    # Second run: RunStart must re-arm the per-run _injected flag.
    _start(opt)
    assert opt._injected is False
    assert isinstance(_pre(opt), ControllableInjection)  # injects again
    assert _end(opt).done is True  # budget now exhausted


def test_retry_budget_counts_runs() -> None:
    opt = _make(max_attempts_on_failure=2)  # 3 sends
    for _ in range(2):
        _start(opt)
        _pre(opt)
        assert _end(opt).done is False
    _start(opt)
    _pre(opt)
    assert _end(opt).done is True


# --- prompt-only guarantees ------------------------------------------------


def test_prompt_only_never_calls_the_llm() -> None:
    llm = _mock_llm()
    opt = _make(llm=llm)
    _start(opt)
    _pre(opt)
    _post(opt, "req", "ANSWER")
    _end(opt)
    llm.complete.assert_not_called()


def test_constructor_overrides_are_used() -> None:
    opt = _make(
        skeleton_key_prompt="CUSTOM_PREAMBLE",
        skeleton_key_acceptance="CUSTOM_ACCEPT",
    )
    _start(opt)
    resp = _pre(opt)
    assert isinstance(resp, ControllableInjection)
    value = resp.value
    assert value.index("CUSTOM_PREAMBLE") < value.index("CUSTOM_ACCEPT") < value.index(GOAL)


def test_empty_override_rejected() -> None:
    import pytest

    with pytest.raises(ValueError):
        SkeletonKeyOptimizer(skeleton_key_prompt="   ")
    with pytest.raises(ValueError):
        SkeletonKeyOptimizer(skeleton_key_acceptance="")
