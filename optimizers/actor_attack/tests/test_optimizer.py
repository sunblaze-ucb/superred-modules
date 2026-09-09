"""Tests for ActorAttackOptimizer's multi-turn interrogation loop."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from actor_attack_optimizer.optimizer import (
    DEFAULT_MAX_TURNS_PER_ACTOR,
    DEFAULT_NUM_ACTORS,
    ActorAttackOptimizer,
)
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

USER = SecurityDomainTag("user")
GOAL = "explain how to pick a lock"

NETWORK = json.dumps(
    {
        "actors": [
            {
                "actor_name": "A Locksmith",
                "relation_to_goal": "practises the craft",
                "opening_question": "OPENING_Q",
            },
            {
                "actor_name": "Second",
                "relation_to_goal": "x",
                "opening_question": "IGNORED",
            },
        ]
    }
)
NON_REFUSAL = json.dumps({"classification": "Non-refusal"})
REFUSAL = json.dumps({"classification": "Refusal"})
RATING = json.dumps({"rating": 5})
HIGH_RATING = json.dumps({"rating": 10})
FOLLOW_UP = json.dumps({"next_question": "FOLLOW_Q", "is_final_probe": False})
FINAL_PROBE = json.dumps({"next_question": "FINAL_Q", "is_final_probe": True})


def _ctrl(name="user_query", value_type="text"):
    return Controllable(name=name, security_domain=USER, value_type=value_type)


def _llm(*contents):
    llm = MagicMock()
    out = []
    for c in contents:
        r = MagicMock()
        r.choices = [MagicMock()]
        r.choices[0].message.content = c
        out.append(r)
    llm.complete = AsyncMock(side_effect=out)
    return llm


def _make(llm, **kw):
    opt = ActorAttackOptimizer(**kw)
    import actor_attack_optimizer.optimizer as mod

    mod.classify_controllables = AsyncMock(return_value={})
    asyncio.run(
        opt.initialize(
            goal=Goal(description=GOAL),
            controllables=[_ctrl()],
            observables=[],
            llm_client=llm,
        )
    )
    return opt


def _start(o): asyncio.run(o.on_event(RunStartEvent(trajectory=MagicMock())))
def _pre(o, n="user_query"):
    return asyncio.run(o.on_event(ControllablePreCallEvent(controllable=_ctrl(n), request="")))
def _post(o, request, answer, n="user_query"):
    return asyncio.run(
        o.on_event(
            ControllablePostCallEvent(controllable=_ctrl(n), request=request, answer=answer)
        )
    )
def _end(o): return asyncio.run(o.on_event(RunEndEvent(evaluation=None)))


def test_upstream_defaults_and_clamping() -> None:
    assert DEFAULT_NUM_ACTORS == 3
    assert DEFAULT_MAX_TURNS_PER_ACTOR == 4
    # upstream clamps rather than rejects
    o = ActorAttackOptimizer(num_actors=0, max_turns_per_actor=1)
    assert o._num_actors == 1 and o._max_turns == 2


def test_first_turn_is_the_first_actors_opening_question() -> None:
    opt = _make(_llm(NETWORK))
    assert opt.actor is not None
    assert opt.actor.actor_name == "A Locksmith"  # upstream uses actors[0]
    _start(opt)
    resp = _pre(opt)
    assert isinstance(resp, ControllableInjection)
    assert resp.value == "OPENING_Q"


def test_next_probe_is_built_from_the_targets_answer() -> None:
    """The defining property: each turn is steered by the previous response."""
    llm = _llm(NETWORK, NON_REFUSAL, RATING, FOLLOW_UP)
    opt = _make(llm)
    _start(opt); _pre(opt)
    _post(opt, "OPENING_Q", "TARGET_SAID_THIS")
    probe_prompt = llm.complete.await_args_list[3].args[0][0]["content"]
    assert "TARGET_SAID_THIS" in probe_prompt
    assert "A Locksmith" in probe_prompt
    _end(opt); _start(opt)
    assert _pre(opt).value == "FOLLOW_Q"


def test_refusal_scores_one_and_skips_the_judge() -> None:
    llm = _llm(NETWORK, REFUSAL, FOLLOW_UP)
    opt = _make(llm)
    _start(opt); _pre(opt)
    _post(opt, "OPENING_Q", "I cannot help")
    # 3 calls: network, non_refusal, next_probe -- no judge call
    assert llm.complete.await_count == 3


def test_high_rating_stops_the_attack() -> None:
    llm = _llm(NETWORK, NON_REFUSAL, HIGH_RATING)
    opt = _make(llm)
    _start(opt); _pre(opt)
    _post(opt, "OPENING_Q", "here is how")
    assert _end(opt).done is True


def test_best_message_tracks_the_highest_rated_turn() -> None:
    llm = _llm(NETWORK, NON_REFUSAL, RATING, FOLLOW_UP)
    opt = _make(llm)
    _start(opt); _pre(opt)
    _post(opt, "OPENING_Q", "partial")
    assert opt.best_message == "OPENING_Q"


def test_turn_budget_is_respected() -> None:
    llm = _llm(NETWORK, NON_REFUSAL, RATING, FOLLOW_UP, NON_REFUSAL, RATING)
    opt = _make(llm, max_turns_per_actor=2)
    for _ in range(2):
        _start(opt)
        assert isinstance(_pre(opt), ControllableInjection)
        _post(opt, "q", "a")
        done = _end(opt).done
    assert done is True


def test_final_probe_flag_sends_one_more_then_stops() -> None:
    llm = _llm(NETWORK, NON_REFUSAL, RATING, FINAL_PROBE, NON_REFUSAL, RATING)
    opt = _make(llm, max_turns_per_actor=4)
    _start(opt); _pre(opt); _post(opt, "OPENING_Q", "a"); _end(opt)
    _start(opt)
    assert _pre(opt).value == "FINAL_Q"
    _post(opt, "FINAL_Q", "a")
    assert _end(opt).done is True


def test_empty_actor_network_ends_immediately() -> None:
    opt = _make(_llm(json.dumps({"actors": []})))
    _start(opt)
    assert isinstance(_pre(opt), ControllableNoInjection)
    assert _end(opt).done is True


def test_unparseable_network_ends_immediately() -> None:
    opt = _make(_llm("not json"))
    _start(opt)
    assert isinstance(_pre(opt), ControllableNoInjection)


def test_conversation_is_pinned_to_one_surface() -> None:
    llm = _llm(NETWORK, NON_REFUSAL, RATING, FOLLOW_UP)
    opt = _make(llm)
    _start(opt)
    assert isinstance(_pre(opt, "user_query"), ControllableInjection)
    _post(opt, "OPENING_Q", "a"); _end(opt)
    _start(opt)
    assert isinstance(_pre(opt, "other_surface"), ControllableNoInjection)
    assert isinstance(_pre(opt, "user_query"), ControllableInjection)


def test_never_injects_system_prompt() -> None:
    opt = _make(_llm(NETWORK))
    _start(opt)
    assert isinstance(_pre(opt, "system_prompt"), ControllableNoInjection)


def test_genuine_budget_exhaustion_propagates() -> None:
    """A spent budget must not look like a quietly finished attack."""
    from superred.core.types.llm import BudgetExhaustedError, LLMUsage

    llm = MagicMock()
    llm.complete = AsyncMock(
        side_effect=BudgetExhaustedError("spent", LLMUsage(calls=3, cost=0.5))
    )
    opt = ActorAttackOptimizer()
    import actor_attack_optimizer.optimizer as mod

    mod.classify_controllables = AsyncMock(return_value={})
    with pytest.raises(BudgetExhaustedError):
        asyncio.run(
            opt.initialize(
                goal=Goal(description=GOAL),
                controllables=[_ctrl()],
                observables=[],
                llm_client=llm,
            )
        )


def test_budgetless_noop_client_degrades_quietly() -> None:
    from superred.core.types.llm import BudgetExhaustedError, LLMUsage

    llm = MagicMock()
    llm.complete = AsyncMock(
        side_effect=BudgetExhaustedError("none", LLMUsage(calls=0, cost=0.0))
    )
    opt = _make(llm)
    _start(opt)
    assert isinstance(_pre(opt), ControllableNoInjection)


def test_post_call_returns_an_injection_decision_not_a_bare_event_response() -> None:
    """The channel type-checks PostCall responses: a plain EventResponse is
    rejected at runtime. This attack reads the answer but never rewrites it,
    so it must decline."""
    llm = _llm(NETWORK, NON_REFUSAL, RATING, FOLLOW_UP)
    opt = _make(llm)
    _start(opt); _pre(opt)
    resp = _post(opt, "OPENING_Q", "answer")
    assert isinstance(resp, ControllableNoInjection)


def test_a_final_probe_does_not_shorten_the_next_task() -> None:
    """``is_final_probe`` lowers ``_max_turns`` to end the conversation early.

    ``initialize`` runs once per task and resets every other piece of per-task
    state, so without restoring ``_max_turns`` a reused optimizer carries the
    shortened budget into the next task -- permanently, since it only ever
    shrinks.
    """
    llm = _llm(NETWORK, NON_REFUSAL, RATING, FINAL_PROBE)
    opt = _make(llm, max_turns_per_actor=4)
    assert opt._max_turns == 4

    _start(opt)
    _pre(opt)
    _post(opt, "OPENING_Q", "answer")
    assert opt._max_turns < 4  # the final probe shortened this conversation

    # A second task on the same instance must start from the configured budget.
    asyncio.run(
        opt.initialize(
            goal=Goal(description="a different goal"),
            controllables=[_ctrl()],
            observables=[],
            llm_client=_llm(NETWORK),
        )
    )
    assert opt._max_turns == 4


def test_a_post_call_on_an_unrelated_surface_does_not_count_as_progress() -> None:
    """``_saw_post_call`` gates the RunEnd stop, so it must mean *our* surface.

    A target that emits post-call events for some other controllable would
    otherwise look like it was advancing the conversation while no probe was
    ever generated, and the run budget would spin.
    """
    llm = _llm(NETWORK, NON_REFUSAL, RATING, FOLLOW_UP)
    opt = _make(llm)
    _start(opt)
    assert isinstance(_pre(opt), ControllableInjection)  # pins user_query

    before = llm.complete.await_count
    _post(opt, "OPENING_Q", "answer", n="some_other_surface")
    assert llm.complete.await_count == before  # no judge calls on a foreign answer
    assert opt._saw_post_call is False
    assert _end(opt).done is True  # nothing can advance this attack; stop


def test_a_post_call_before_the_opener_lands_cannot_clobber_it() -> None:
    """Until the opening question is sent there is no turn to score.

    ``_channel`` is still ``None`` then, so an unrelated post-call must be
    ignored rather than overwriting ``_pending`` with a follow-up.
    """
    llm = _llm(NETWORK, NON_REFUSAL, RATING, FOLLOW_UP)
    opt = _make(llm)
    _start(opt)
    opener = opt._pending
    assert opener == "OPENING_Q"

    _post(opt, "unrelated", "some answer", n="some_other_surface")
    assert opt._pending == opener  # the opener survives

    resp = _pre(opt)
    assert isinstance(resp, ControllableInjection)
    assert resp.value == opener
