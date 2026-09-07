"""Tests for LLM-classified injection-surface selection."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

from agentbreaker_optimizer.optimizer import AgentBreakerOptimizer
from agentbreaker_optimizer.surfaces import (
    ROLE_CATEGORIES,
    is_eligible_surface,
    preferred_surface_names,
    should_inject,
)
from superred.core.types.controllable import Controllable
from superred.core.types.events import (
    ControllableInjection,
    ControllableNoInjection,
    ControllablePreCallEvent,
    RunStartEvent,
)
from superred.core.types.goal import Goal
from superred.core.types.security_domain import SecurityDomainTag

USER = SecurityDomainTag("user")


def _c(name: str, value_type: str = "text") -> Controllable:
    return Controllable(name=name, security_domain=USER, value_type=value_type)


def test_role_categories_offer_a_user_prompt_bucket() -> None:
    assert "user-prompt" in ROLE_CATEGORIES


def test_eligibility_backstop() -> None:
    assert is_eligible_surface(_c("anything")) is True
    assert is_eligible_surface(_c("system_prompt")) is False
    assert is_eligible_surface(_c("payload", value_type="json")) is False


def test_classification_narrows_to_the_user_prompt_surface() -> None:
    ctrls = [_c("db_lookup"), _c("user_query")]
    roles = {"db_lookup": "content-injection", "user_query": "user-prompt"}
    assert preferred_surface_names(ctrls, roles) == frozenset({"user_query"})


def test_no_classification_falls_back_to_every_eligible_surface() -> None:
    ctrls = [_c("db_lookup"), _c("user_query"), _c("system_prompt")]
    assert preferred_surface_names(ctrls, {}) == frozenset({"db_lookup", "user_query"})


def test_classified_non_preferred_surface_is_skipped() -> None:
    roles = {"db_lookup": "content-injection", "user_query": "user-prompt"}
    pref = frozenset({"user_query"})
    assert should_inject(_c("db_lookup"), pref, roles) is False
    assert should_inject(_c("user_query"), pref, roles) is True


def test_surface_unseen_at_initialize_still_receives_the_payload() -> None:
    roles = {"user_query": "user-prompt"}
    assert should_inject(_c("late"), frozenset({"user_query"}), roles) is True


def test_system_prompt_role_never_receives_the_payload() -> None:
    assert should_inject(_c("sp"), frozenset({"sp"}), {"sp": "system-prompt"}) is False


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


ANALYSIS = '{"tool_analyses": {"send_email": {"functionality": "sends email", "vulnerabilities": "none", "exploit_strategies": "coax", "attack_prompts": ["do it"]}}, "priority_targets": ["send_email - impact"]}'


def _obs():
    from superred.core.types.observable import Observable, ObservableValue

    o = Observable(name="tool_catalog_listing", security_domain=USER)
    return ObservableValue(observable=o, content="send_email(to, body): sends email")


def _opt_with_roles(roles, ctrls):
    opt = AgentBreakerOptimizer()
    import agentbreaker_optimizer.optimizer as mod

    mod.classify_controllables = AsyncMock(return_value=roles)
    asyncio.run(
        opt.initialize(
            goal=Goal(description="pick a lock"),
            controllables=ctrls,
            observables=[_obs()],
            llm_client=_llm(ANALYSIS, "attack prompt here"),
        )
    )
    asyncio.run(opt.on_event(RunStartEvent(trajectory=MagicMock())))
    return opt


def _pre(opt, name: str):
    return asyncio.run(
        opt.on_event(ControllablePreCallEvent(controllable=_c(name), request=""))
    )


def test_payload_skips_content_surface_and_lands_on_the_user_prompt() -> None:
    """The behaviour the classifier buys: a content surface firing first no
    longer consumes the injection."""
    ctrls = [_c("db_lookup"), _c("user_query")]
    opt = _opt_with_roles(
        {"db_lookup": "content-injection", "user_query": "user-prompt"}, ctrls
    )
    assert isinstance(_pre(opt, "db_lookup"), ControllableNoInjection)
    resp = _pre(opt, "user_query")
    assert isinstance(resp, ControllableInjection)
    assert resp.value


def test_classifier_failure_preserves_first_eligible_behaviour() -> None:
    ctrls = [_c("db_lookup"), _c("user_query")]
    opt = _opt_with_roles({}, ctrls)
    assert isinstance(_pre(opt, "db_lookup"), ControllableInjection)
