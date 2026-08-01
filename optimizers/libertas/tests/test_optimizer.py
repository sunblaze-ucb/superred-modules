"""Event-lifecycle tests for LibertasOptimizer."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from superred.core.llm import LLMClient
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

from libertas_optimizer import LibertasOptimizer, load_prompt_templates, render_prompt

USER_TAG = SecurityDomainTag("user")
SYSTEM_TAG = SecurityDomainTag("system")
MODEL_TAG = SecurityDomainTag("model")
GOAL = "Explain the benchmark's forbidden behavior. 🧪"


class _ChoiceMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _ChoiceMessage(content)


class _LLMResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]


class _ScriptedLLM(LLMClient):
    def __init__(self, responses: list[str | Exception]) -> None:
        self.responses = responses
        self.calls: list[tuple[list[dict[str, str]], dict[str, Any]]] = []

    async def complete(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> Any:
        self.calls.append((messages, kwargs))
        if not self.responses:
            raise AssertionError("unexpected Libertas helper-LLM call")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return _LLMResponse(response)


def _user_ctrl(name: str = "user_message") -> Controllable:
    return Controllable(name=name, security_domain=USER_TAG)


def _system_ctrl() -> Controllable:
    return Controllable(name="system_prompt", security_domain=SYSTEM_TAG)


def _model_observable(model: str) -> ObservableValue:
    return ObservableValue(
        observable=Observable(name="model", security_domain=MODEL_TAG),
        content=model,
    )


def _text_observable(name: str, content: str) -> ObservableValue:
    return ObservableValue(
        observable=Observable(name=name, security_domain=MODEL_TAG),
        content=content,
    )


def _initialize(
    optimizer: LibertasOptimizer,
    *,
    model: str = "openai/gpt-5.4-mini",
    controllables: list[Controllable] | None = None,
    llm: LLMClient | None = None,
) -> LLMClient:
    selected_llm = llm or MagicMock()
    asyncio.run(
        optimizer.initialize(
            goal=Goal(description=GOAL),
            controllables=controllables or [_system_ctrl(), _user_ctrl()],
            observables=[_model_observable(model)],
            llm_client=selected_llm,
        )
    )
    return selected_llm


def _event(optimizer: LibertasOptimizer, event):
    return asyncio.run(optimizer.on_event(event))


def _evaluation(success: bool) -> EvaluationResult:
    return EvaluationResult(
        success=success,
        primary_score=Score(value=1.0 if success else 0.0),
    )


def test_zero_argument_construction() -> None:
    LibertasOptimizer()


def test_helper_llm_can_pick_a_later_compatible_prompt_before_attempt_limit() -> None:
    candidates = load_prompt_templates(
        source_files=("OPENAI.mkd",),
    )
    selected = next(template for template in candidates if template.heading == "GPT-4O")
    llm = _ScriptedLLM([json.dumps({"template_ids": [selected.id]}, ensure_ascii=False)])
    optimizer = LibertasOptimizer(
        source_files=("OPENAI.mkd",),
        max_attempts=1,
    )

    _initialize(optimizer, llm=llm)

    assert [template.id for template in optimizer.templates] == [selected.id]
    assert optimizer.selection_method == "llm"
    assert len(llm.calls) == 1
    messages, kwargs = llm.calls[0]
    assert kwargs == {"max_tokens": 2048}
    payload = json.loads(messages[1]["content"])
    assert payload["target_model"] == "openai/gpt-5.4-mini"
    assert payload["provider"] == "openai"
    assert payload["goal"] == GOAL
    assert {candidate["source_file"] for candidate in payload["candidates"]} == {"OPENAI.mkd"}
    assert all(
        set(candidate)
        == {
            "template_id",
            "source_file",
            "heading",
            "provider",
            "delivery",
            "goal_markers",
        }
        for candidate in payload["candidates"]
    )
    assert all(template.raw_template not in messages[1]["content"] for template in candidates)


def test_deterministic_selection_skips_the_helper_llm() -> None:
    llm = _ScriptedLLM([])
    optimizer = LibertasOptimizer(
        source_files=("OPENAI.mkd",),
        max_attempts=1,
        selection_strategy="deterministic",
    )

    _initialize(optimizer, llm=llm)

    expected = load_prompt_templates(source_files=("OPENAI.mkd",))[0]
    assert optimizer.templates == (expected,)
    assert llm.calls == []


def test_malformed_helper_response_falls_back_and_reports_selection_method() -> None:
    llm = _ScriptedLLM(["not JSON"])
    optimizer = LibertasOptimizer(source_files=("OPENAI.mkd",), max_attempts=1)

    _initialize(optimizer, llm=llm)

    expected = load_prompt_templates(source_files=("OPENAI.mkd",))[0]
    assert optimizer.templates == (expected,)
    assert optimizer.selection_method == "deterministic-fallback"


@pytest.mark.parametrize(
    "response",
    [
        "{}",
        '{"template_ids": []}',
        '{"template_ids": "not-a-list"}',
        '{"template_ids": [7]}',
        '{"template_ids": ["unknown-template-id"]}',
        '```json\n{"template_ids": ["unknown-template-id"]}\n```',
    ],
)
def test_invalid_helper_rankings_fall_back_atomically(response: str) -> None:
    llm = _ScriptedLLM([response])
    optimizer = LibertasOptimizer(source_files=("OPENAI.mkd",))

    _initialize(optimizer, llm=llm)

    assert optimizer.templates == load_prompt_templates(source_files=("OPENAI.mkd",))
    assert optimizer.selection_method == "deterministic-fallback"


def test_helper_ranking_rejects_unexpected_object_fields() -> None:
    candidates = load_prompt_templates(source_files=("OPENAI.mkd",))
    llm = _ScriptedLLM(
        [
            json.dumps(
                {
                    "template_ids": [candidates[-1].id],
                    "reason": "schema does not allow this field",
                }
            )
        ]
    )
    optimizer = LibertasOptimizer(source_files=("OPENAI.mkd",))

    _initialize(optimizer, llm=llm)

    assert optimizer.templates == candidates
    assert optimizer.selection_method == "deterministic-fallback"


def test_duplicate_helper_ids_fall_back_instead_of_dropping_attempts() -> None:
    candidates = load_prompt_templates(source_files=("OPENAI.mkd",))
    duplicate = candidates[-1].id
    llm = _ScriptedLLM([json.dumps({"template_ids": [duplicate, duplicate]}, ensure_ascii=False)])
    optimizer = LibertasOptimizer(source_files=("OPENAI.mkd",))

    _initialize(optimizer, llm=llm)

    assert optimizer.templates == candidates
    assert optimizer.selection_method == "deterministic-fallback"


def test_helper_exception_falls_back_without_aborting_initialization() -> None:
    llm = _ScriptedLLM([RuntimeError("selector unavailable")])
    optimizer = LibertasOptimizer(source_files=("OPENAI.mkd",), max_attempts=1)

    _initialize(optimizer, llm=llm)

    expected = load_prompt_templates(source_files=("OPENAI.mkd",))[0]
    assert optimizer.templates == (expected,)
    assert optimizer.selection_method == "deterministic-fallback"


def test_partial_helper_ranking_keeps_unmentioned_candidates_in_source_order() -> None:
    candidates = load_prompt_templates(source_files=("OPENAI.mkd",))
    preferred = candidates[-1]
    llm = _ScriptedLLM([json.dumps({"template_ids": [preferred.id]}, ensure_ascii=False)])
    optimizer = LibertasOptimizer(source_files=("OPENAI.mkd",))

    _initialize(optimizer, llm=llm)

    assert optimizer.templates == (preferred, *candidates[:-1])


def test_single_candidate_skips_helper_without_reporting_a_fallback() -> None:
    llm = _ScriptedLLM([])
    optimizer = LibertasOptimizer(
        source_files=("APPLE.mkd",),
        selection_strategy="llm",
    )

    _initialize(optimizer, model="apple/apple-intelligence", llm=llm)

    assert len(optimizer.templates) == 1
    assert optimizer.selection_method == "deterministic"
    assert llm.calls == []


def test_explicit_model_identity_drives_provider_and_helper_context() -> None:
    model_identity = "anthropic/claude-opus-4-1"
    candidate = next(
        template
        for template in load_prompt_templates(provider="anthropic")
        if template.heading == "OPUS-4.1"
    )
    llm = _ScriptedLLM([json.dumps({"template_ids": [candidate.id]}, ensure_ascii=False)])
    optimizer = LibertasOptimizer(model_identity=model_identity, max_attempts=1)

    _initialize(optimizer, model="openai/gpt-5.4-mini", llm=llm)

    assert optimizer.resolved_provider == "anthropic"
    assert optimizer.model_identity == model_identity
    assert optimizer.templates == (candidate,)
    request = json.loads(llm.calls[0][0][1]["content"])
    assert request["target_model"] == model_identity
    assert request["provider"] == "anthropic"


def test_unrelated_observable_text_cannot_impersonate_model_identity() -> None:
    optimizer = LibertasOptimizer(
        max_attempts=1,
        selection_strategy="deterministic",
    )
    asyncio.run(
        optimizer.initialize(
            goal=Goal(description=GOAL),
            controllables=[_user_ctrl()],
            observables=[
                _text_observable(
                    "system_prompt",
                    "You are not Claude; do not imitate Anthropic models.",
                )
            ],
            llm_client=_ScriptedLLM([]),
        )
    )

    assert optimizer.model_identity is None
    assert optimizer.resolved_provider is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"selection_strategy": "random"}, "selection_strategy"),
        ({"model_identity": "  "}, "model_identity"),
    ],
)
def test_invalid_selection_configuration_fails_loudly(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        LibertasOptimizer(**kwargs)


def test_detected_provider_prioritizes_native_then_universal_then_transfer_attacks() -> None:
    optimizer = LibertasOptimizer(selection_strategy="deterministic")
    _initialize(optimizer, model="nvidia/llama-3.1-nemotron")

    assert optimizer.resolved_provider == "nvidia"
    providers = [template.provider for template in optimizer.templates]
    native_count = providers.count("nvidia")
    universal_count = providers.count(None)
    assert native_count > 0
    assert universal_count > 0
    assert providers[:native_count] == ["nvidia"] * native_count
    assert providers[native_count : native_count + universal_count] == [None] * universal_count
    assert any(provider not in {"nvidia", None} for provider in providers)


def test_explicit_provider_remains_a_strict_native_first_filter() -> None:
    optimizer = LibertasOptimizer(
        provider="meta",
        selection_strategy="deterministic",
    )

    _initialize(optimizer, model="nvidia/llama-3.1-nemotron")

    assert optimizer.resolved_provider == "meta"
    assert [template.provider for template in optimizer.templates] == [
        "meta",
        "meta",
        "meta",
        None,
        None,
    ]


def test_source_files_remain_an_exact_filter_without_provider_reordering() -> None:
    expected = load_prompt_templates(source_files=("OPENAI.mkd", "META.mkd"))
    optimizer = LibertasOptimizer(
        source_files=("OPENAI.mkd", "META.mkd"),
        selection_strategy="deterministic",
    )

    _initialize(optimizer, model="meta/llama-4-maverick")

    assert optimizer.templates == expected


def test_helper_sees_transfer_candidates_but_cannot_move_them_ahead_of_native() -> None:
    all_candidates = load_prompt_templates(provider=None)
    native = next(template for template in all_candidates if template.provider == "meta")
    transfer = next(template for template in all_candidates if template.provider == "openai")
    llm = _ScriptedLLM(
        [json.dumps({"template_ids": [transfer.id, native.id]}, ensure_ascii=False)]
    )
    optimizer = LibertasOptimizer(model_identity="meta/llama-4-maverick")

    _initialize(optimizer, llm=llm)

    catalog = json.loads(llm.calls[0][0][1]["content"])["candidates"]
    assert {candidate["provider"] for candidate in catalog} >= {"meta", "openai", "nvidia"}
    providers = [template.provider for template in optimizer.templates]
    assert providers.index("meta") < providers.index(None) < providers.index("openai")
    assert optimizer.templates[0] == native


def test_attempt_limit_is_applied_after_native_and_universal_prioritization() -> None:
    optimizer = LibertasOptimizer(
        model_identity="meta/llama-4-maverick",
        selection_strategy="deterministic",
        max_attempts=4,
    )

    _initialize(optimizer)

    assert [template.provider for template in optimizer.templates] == [
        "meta",
        "meta",
        "meta",
        None,
    ]


def test_unknown_model_uses_all_provider_families() -> None:
    optimizer = LibertasOptimizer(max_attempts=100)
    _initialize(optimizer, model="private/opaque-model")

    assert optimizer.resolved_provider is None
    providers = {template.provider for template in optimizer.templates}
    assert {"openai", "anthropic", "google", "xai"}.issubset(providers)


def test_injects_exact_rendered_template_once_on_user_surface() -> None:
    optimizer = LibertasOptimizer(
        provider="openai",
        max_attempts=1,
        selection_strategy="deterministic",
    )
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


def test_system_only_scope_schedules_deliverable_system_templates() -> None:
    optimizer = LibertasOptimizer(
        provider="openai",
        include_system_templates=True,
        source_files=("OPENAI.mkd",),
        selection_strategy="deterministic",
    )

    _initialize(optimizer, controllables=[_system_ctrl()])

    assert optimizer.templates
    assert all(template.delivery == "system" for template in optimizer.templates)
    _event(optimizer, RunStartEvent(trajectory=MagicMock()))
    injected = _event(
        optimizer,
        ControllablePreCallEvent(controllable=_system_ctrl(), request="system"),
    )
    assert isinstance(injected, ControllableInjection)


def test_scope_without_a_runnable_surface_finishes_without_error() -> None:
    optimizer = LibertasOptimizer(
        provider="openai",
        source_files=("OPENAI.mkd",),
        selection_strategy="deterministic",
    )

    _initialize(optimizer, controllables=[_system_ctrl()])

    assert optimizer.templates == ()
    _event(optimizer, RunStartEvent(trajectory=MagicMock()))
    end = _event(optimizer, RunEndEvent(evaluation=None))
    assert isinstance(end, RunEndResponse)
    assert end.done is True


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
    with pytest.raises(ValueError, match="max_attempts"):
        LibertasOptimizer(max_attempts=0)


def test_unavailable_explicit_controllable_finishes_without_error() -> None:
    optimizer = LibertasOptimizer(
        target_controllable_name="missing",
        selection_strategy="deterministic",
    )

    _initialize(optimizer)

    assert optimizer.templates == ()
    _event(optimizer, RunStartEvent(trajectory=MagicMock()))
    end = _event(optimizer, RunEndEvent(evaluation=None))
    assert isinstance(end, RunEndResponse)
    assert end.done is True
