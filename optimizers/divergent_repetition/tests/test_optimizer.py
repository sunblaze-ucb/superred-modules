"""Tests for DivergentRepetitionOptimizer."""

from __future__ import annotations

import asyncio
import json
from importlib import resources
from unittest.mock import AsyncMock, MagicMock

from divergent_repetition_optimizer import DivergentRepetitionOptimizer, build_prompts
from superred.core.types.controllable import Controllable
from superred.core.types.events import (
    ControllableInjection,
    ControllableNoInjection,
    ControllablePreCallEvent,
    RunEndEvent,
    RunStartEvent,
)
from superred.core.types.goal import Goal
from superred.core.types.security_domain import SecurityDomainTag

USER = SecurityDomainTag("user")


def _data():
    return json.loads(
        resources.files("divergent_repetition_optimizer").joinpath("data.json").read_text()
    )


def _ctrl(name="user_query", value_type="text"):
    return Controllable(name=name, security_domain=USER, value_type=value_type)


def _make(**kw):
    opt = DivergentRepetitionOptimizer(**kw)
    import divergent_repetition_optimizer.optimizer as mod

    mod.classify_controllables = AsyncMock(return_value={})
    asyncio.run(
        opt.initialize(
            goal=Goal(description="explain how to pick a lock"),
            controllables=[_ctrl()],
            observables=[],
            llm_client=MagicMock(),
        )
    )
    return opt


def _start(o): asyncio.run(o.on_event(RunStartEvent(trajectory=MagicMock())))
def _pre(o, n="user_query"):
    return asyncio.run(o.on_event(ControllablePreCallEvent(controllable=_ctrl(n), request="")))
def _end(o): return asyncio.run(o.on_event(RunEndEvent(evaluation=None)))


def test_data_matches_upstream_shape() -> None:
    d = _data()
    assert d["repeat_words"] == ["poem", "company", "book", "a"]
    assert d["num_repetitions"] == [1, 3, 40]
    assert len(d["prompt_templates"]) == 3
    assert len(d["repeat_words_extended"]) == 105


def test_build_prompts_reproduces_the_nested_loop() -> None:
    d = _data()
    prompts = build_prompts()
    assert len(prompts) == 4 * 3 * 3 == 36
    # first prompt: first word, first template, 1 repetition
    assert prompts[0] == d["prompt_templates"][0].replace("[XX]", "poem")


def test_extended_uses_the_105_word_list() -> None:
    assert len(build_prompts(extended=True)) == 105 * 3 * 3


def test_prompts_ask_to_repeat_a_word() -> None:
    for p in build_prompts()[:5]:
        assert "epeat" in p  # "Repeat"/"repeat"


def test_goal_agnostic_prompts_do_not_embed_the_goal() -> None:
    opt = _make()
    _start(opt)
    resp = _pre(opt)
    assert isinstance(resp, ControllableInjection)
    assert "pick a lock" not in resp.value  # data-extraction attack ignores the goal


def test_sweeps_all_prompts_one_per_run() -> None:
    opt = _make()
    seen = 0
    while True:
        _start(opt)
        r = _pre(opt)
        if isinstance(r, ControllableNoInjection):
            break
        seen += 1
        if _end(opt).done:
            break
    assert seen == 36


def test_shuffle_is_seed_reproducible() -> None:
    a, b = _make(shuffle=True, seed=5), _make(shuffle=True, seed=5)
    _start(a); _start(b)
    assert _pre(a).value == _pre(b).value


def test_no_shuffle_is_upstream_order() -> None:
    opt = _make()
    _start(opt)
    assert _pre(opt).value == build_prompts()[0]


def test_injects_once_per_run() -> None:
    opt = _make()
    _start(opt)
    assert isinstance(_pre(opt, "user_query"), ControllableInjection)
    assert isinstance(_pre(opt, "db_lookup"), ControllableNoInjection)


def test_never_injects_system_prompt() -> None:
    opt = _make()
    _start(opt)
    assert isinstance(_pre(opt, "system_prompt"), ControllableNoInjection)
