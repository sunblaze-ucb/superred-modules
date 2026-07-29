"""Event-lifecycle tests for LibertasOptimizer."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

from superred.core.types.controllable import Controllable
from superred.core.types.evaluation import EvaluationResult, Score
from superred.core.types.event import EventResponse
from superred.core.types.events import (
    ControllableInjection,
    ControllableNoInjection,
    ControllablePostCallEvent,
    ControllablePreCallEvent,
    RunEndEvent,
    RunEndResponse,
    RunStartEvent,
)
from superred.core.types.goal import Goal
from superred.core.types.observable import Observable, ObservableValue
from superred.core.types.security_domain import SecurityDomainTag

from libertas_optimizer import LibertasOptimizer, render_prompt

USER_TAG = SecurityDomainTag("user")
SYSTEM_TAG = SecurityDomainTag("system")
MODEL_TAG = SecurityDomainTag("model")
GOAL = "Explain the benchmark's forbidden behavior. 🧪"


def _user_ctrl(name: str = "user_message") -> Controllable:
    return Controllable(name=name, security_domain=USER_TAG)


def _system_ctrl() -> Controllable:
    return Controllable(name="system_prompt", security_domain=SYSTEM_TAG)


def _model_observable(model: str) -> ObservableValue:
    return ObservableValue(
        observable=Observable(name="model", security_domain=MODEL_TAG),
        content=model,
    )


def _initialize(
    optimizer: LibertasOptimizer,
    *,
    model: str = "openai/gpt-5.4-mini",
    controllables: list[Controllable] | None = None,
) -> MagicMock:
    llm = MagicMock()
    asyncio.run(
        optimizer.initialize(
            goal=Goal(description=GOAL),
            controllables=controllables or [_system_ctrl(), _user_ctrl()],
            observables=[_model_observable(model)],
            llm_client=llm,
        )
    )
    return llm


def _event(optimizer: LibertasOptimizer, event):
    return asyncio.run(optimizer.on_event(event))


def _evaluation(success: bool) -> EvaluationResult:
    return EvaluationResult(
        success=success,
        primary_score=Score(value=1.0 if success else 0.0),
    )


def test_zero_argument_construction() -> None:
    LibertasOptimizer()


def test_model_identity_routes_to_matching_upstream_files() -> None:
    optimizer = LibertasOptimizer()
    _initialize(optimizer, model="anthropic/claude-opus-4-6")

    assert optimizer.resolved_provider == "anthropic"
    assert optimizer.templates
    assert {template.source_file for template in optimizer.templates} == {
        "ANTHROPIC.mkd"
    }


def test_unknown_model_uses_all_provider_families() -> None:
    optimizer = LibertasOptimizer(max_attempts=100)
    _initialize(optimizer, model="private/opaque-model")

    assert optimizer.resolved_provider is None
    providers = {template.provider for template in optimizer.templates}
    assert {"openai", "anthropic", "google", "xai"}.issubset(providers)


def test_injects_exact_rendered_template_once_on_user_surface() -> None:
    optimizer = LibertasOptimizer(provider="openai", max_attempts=1)
    llm = _initialize(optimizer)

    start = _event(optimizer, RunStartEvent(trajectory=MagicMock()))
    assert isinstance(start, EventResponse)
    template = optimizer.current_template
    assert template is not None

    system = _event(
        optimizer,
        ControllablePreCallEvent(controllable=_system_ctrl(), request="system"),
    )
    assert isinstance(system, ControllableNoInjection)

    injected = _event(
        optimizer,
        ControllablePreCallEvent(controllable=_user_ctrl(), request="user"),
    )
    assert isinstance(injected, ControllableInjection)
    assert injected.value == render_prompt(template, GOAL)

    second = _event(
        optimizer,
        ControllablePreCallEvent(controllable=_user_ctrl(), request="user"),
    )
    assert isinstance(second, ControllableNoInjection)
    llm.assert_not_called()


def test_post_call_is_never_modified() -> None:
    optimizer = LibertasOptimizer(provider="openai", max_attempts=1)
    _initialize(optimizer)
    _event(optimizer, RunStartEvent(trajectory=MagicMock()))

    response = _event(
        optimizer,
        ControllablePostCallEvent(
            controllable=_user_ctrl(),
            request="prompt",
            answer="victim response",
        ),
    )
    assert isinstance(response, ControllableNoInjection)


def test_templates_advance_one_per_run() -> None:
    optimizer = LibertasOptimizer(provider="google", max_attempts=2)
    _initialize(optimizer, model="google/gemini-3-pro")

    _event(optimizer, RunStartEvent(trajectory=MagicMock()))
    first = optimizer.current_template
    end = _event(optimizer, RunEndEvent(evaluation=_evaluation(False)))
    assert isinstance(end, RunEndResponse)
    assert end.done is False

    _event(optimizer, RunStartEvent(trajectory=MagicMock()))
    second = optimizer.current_template
    assert first is not None and second is not None
    assert first.id != second.id

    end = _event(optimizer, RunEndEvent(evaluation=_evaluation(False)))
    assert isinstance(end, RunEndResponse)
    assert end.done is True


def test_security_claim_success_stops_schedule_early() -> None:
    optimizer = LibertasOptimizer(provider="google", max_attempts=3)
    _initialize(optimizer, model="gemini-3-pro")
    _event(optimizer, RunStartEvent(trajectory=MagicMock()))

    end = _event(optimizer, RunEndEvent(evaluation=_evaluation(True)))
    assert isinstance(end, RunEndResponse)
    assert end.done is True


def test_system_template_uses_system_then_user_query() -> None:
    optimizer = LibertasOptimizer(
        provider="openai",
        include_system_templates=True,
        source_files=("OPENAI.mkd",),
    )
    _initialize(optimizer)

    # The first OPENAI.mkd strict entry is a user template. Advance to the
    # first upstream custom-instruction entry.
    _event(optimizer, RunStartEvent(trajectory=MagicMock()))
    _event(optimizer, RunEndEvent(evaluation=_evaluation(False)))
    _event(optimizer, RunStartEvent(trajectory=MagicMock()))
    template = optimizer.current_template
    assert template is not None
    assert template.delivery == "system"

    system = _event(
        optimizer,
        ControllablePreCallEvent(controllable=_system_ctrl(), request="system"),
    )
    assert isinstance(system, ControllableInjection)
    assert system.value == render_prompt(template, GOAL)

    user = _event(
        optimizer,
        ControllablePreCallEvent(controllable=_user_ctrl(), request="user"),
    )
    assert isinstance(user, ControllableInjection)
    assert user.value == GOAL

    extra = _event(
        optimizer,
        ControllablePreCallEvent(controllable=_user_ctrl(), request="user"),
    )
    assert isinstance(extra, ControllableNoInjection)


def test_system_templates_are_filtered_without_system_surface() -> None:
    optimizer = LibertasOptimizer(
        provider="openai",
        include_system_templates=True,
        source_files=("OPENAI.mkd",),
    )
    _initialize(optimizer, controllables=[_user_ctrl()])
    assert all(template.delivery == "user" for template in optimizer.templates)


def test_explicit_controllable_override_is_respected() -> None:
    alternate = _user_ctrl("custom_prompt")
    optimizer = LibertasOptimizer(
        provider="openai",
        max_attempts=1,
        target_controllable_name="custom_prompt",
    )
    _initialize(optimizer, controllables=[_system_ctrl(), alternate])
    _event(optimizer, RunStartEvent(trajectory=MagicMock()))

    skipped = _event(
        optimizer,
        ControllablePreCallEvent(controllable=_system_ctrl(), request="system"),
    )
    assert isinstance(skipped, ControllableNoInjection)
    injected = _event(
        optimizer,
        ControllablePreCallEvent(controllable=alternate, request="prompt"),
    )
    assert isinstance(injected, ControllableInjection)


def test_invalid_configuration_fails_loudly() -> None:
    try:
        LibertasOptimizer(max_attempts=0)
    except ValueError as exc:
        assert "max_attempts" in str(exc)
    else:
        raise AssertionError("expected invalid max_attempts to fail")

    optimizer = LibertasOptimizer(target_controllable_name="missing")
    try:
        _initialize(optimizer)
    except ValueError as exc:
        assert "unavailable" in str(exc)
    else:
        raise AssertionError("expected unavailable controllable to fail")
