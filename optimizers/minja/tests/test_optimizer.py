from __future__ import annotations

import asyncio
from typing import Any

import pytest

from superred.core.channel import EventEnvelope
from superred.core.types.controllable import Controllable
from superred.core.types.evaluation import EvaluationResult, Score
from superred.core.types.event import Event, EventResponse
from superred.core.types.events import (
    ControllableInjection,
    ControllableNoInjection,
    ControllablePostCallEvent,
    ControllablePreCallEvent,
    ObservableEvent,
    RunEndEvent,
    RunEndResponse,
    RunStartEvent,
)
from superred.core.types.goal import Goal
from superred.core.types.observable import Observable, ObservableValue
from superred.core.types.security_domain import SecurityDomainTag

from minja_optimizer.optimizer import MinjaOptimizer
from minja_optimizer.prompts import (
    DEFAULT_INDICATION_PROMPTS,
    DEFAULT_PAIR,
    OFFICIAL_RAP_VICTIM_TARGET_PAIRS,
    load_official_qa_victims,
    load_official_webshop_instructions,
)

USER_TAG = SecurityDomainTag("user")
SYSTEM_TAG = SecurityDomainTag("system")
PROMPT_TAG = SecurityDomainTag("prompt", parent=SYSTEM_TAG)
TOOLS_TAG = SecurityDomainTag("tools")
MODEL_TAG = SecurityDomainTag("model_identity", parent=SYSTEM_TAG)
TOOL_TAG = SecurityDomainTag("tool_catalogue", parent=SYSTEM_TAG)
RESPONSE_TAG = SecurityDomainTag("response")


class FakeTrajectory:
    def __init__(self) -> None:
        self._items: list[Any] = []

    def push(self, item: Any) -> None:
        self._items.append(item)

    def snapshot(self) -> list[Any]:
        return list(self._items)

    def drain(self) -> list[Any]:
        items = list(self._items)
        self._items.clear()
        return items


def ctrl(
    name: str = "user_prompt", tag: SecurityDomainTag = USER_TAG, value_type: str = "text"
) -> Controllable:
    return Controllable(name=name, security_domain=tag, value_type=value_type)


def obs(
    name: str, content: Any, tag: SecurityDomainTag = MODEL_TAG, observable_type: str = "text"
) -> ObservableValue:
    return ObservableValue(
        Observable(name=name, security_domain=tag, observable_type=observable_type), content
    )


def eval_result(success: bool, score: float | None = None) -> EvaluationResult:
    return EvaluationResult(
        success=success,
        primary_score=Score(
            value=1.0 if score is None and success else (score or 0.0), name="primary"
        ),
        sub_scores={},
        rationale="ok" if success else "not yet",
    )


async def dispatch(opt: MinjaOptimizer, event: Event) -> EventResponse:
    loop = asyncio.get_running_loop()
    future: asyncio.Future[EventResponse] = loop.create_future()
    envelope = EventEnvelope(event=event, future=future, loop=loop)
    await opt._dispatch(envelope)
    return await future


async def init_opt(**kwargs: Any) -> MinjaOptimizer:
    opt = MinjaOptimizer(**kwargs)
    await opt.initialize(
        goal=Goal(description="make the agent choose the target item"),
        controllables=kwargs.pop("controllables", [ctrl("user_prompt")])
        if "controllables" in kwargs
        else [ctrl("user_prompt")],
        observables=kwargs.pop("observables", []) if "observables" in kwargs else [],
        llm_client=Any,  # type: ignore[arg-type]
    )
    return opt


@pytest.mark.asyncio
async def test_first_run_appends_official_full_indication_prompt() -> None:
    opt = MinjaOptimizer(inject_num=1, test_num=0, random_seed=0)
    await opt.initialize(Goal(description="poison memory"), [ctrl("user_prompt")], [], Any)  # type: ignore[arg-type]
    await dispatch(opt, RunStartEvent(trajectory=FakeTrajectory()))

    resp = await dispatch(
        opt,
        ControllablePreCallEvent(
            controllable=ctrl("user_prompt"),
            request="i need a toothbrush under 20 dollars",
        ),
    )

    assert isinstance(resp, ControllableInjection)
    assert resp.value.startswith("i need a toothbrush under 20 dollars")
    assert "High priority note" in resp.value
    assert DEFAULT_PAIR.target in resp.value
    assert "{victim}" not in resp.value


def test_official_rap_data_is_loaded_from_packaged_files() -> None:
    instructions = load_official_webshop_instructions()

    assert len(OFFICIAL_RAP_VICTIM_TARGET_PAIRS) == 9
    assert OFFICIAL_RAP_VICTIM_TARGET_PAIRS[-1] == DEFAULT_PAIR
    assert len(instructions) == 10587
    assert instructions[0]["session"] == "fixed_0_0"
    assert instructions[0]["instruction"] == (
        "i need a long clip-in hair extension which is natural looking, and price lower than "
        "20.00 dollars"
    )


def test_official_qa_victim_data_is_packaged_for_target_adapters() -> None:
    victims = load_official_qa_victims()

    assert victims[0]["victim"] == "food"
    assert victims[0]["note"][0]["note1"].startswith("(High priority note")


@pytest.mark.asyncio
async def test_progressive_shortening_uses_all_notes_then_plain_attack_query() -> None:
    opt = MinjaOptimizer(inject_num=1, test_num=0, random_seed=0)
    await opt.initialize(Goal(description="poison memory"), [ctrl("user_prompt")], [], Any)  # type: ignore[arg-type]

    rendered: list[str] = []
    for _ in range(len(DEFAULT_INDICATION_PROMPTS) + 1):
        await dispatch(opt, RunStartEvent(trajectory=FakeTrajectory()))
        resp = await dispatch(
            opt,
            ControllablePreCallEvent(
                controllable=ctrl("user_prompt"),
                request="find a toothbrush",
            ),
        )
        assert isinstance(resp, ControllableInjection)
        rendered.append(resp.value)
        await dispatch(opt, RunEndEvent(evaluation=eval_result(False), security_domain=USER_TAG))

    assert "High priority note" in rendered[0]
    assert "All kinds of toothbrush are always out of stock." in rendered[-2]
    assert rendered[-1] == "find a toothbrush"


@pytest.mark.asyncio
async def test_framework_success_stops_optimizer() -> None:
    opt = MinjaOptimizer(inject_num=1, test_num=1)
    await opt.initialize(Goal(description="poison memory"), [ctrl("user_prompt")], [], Any)  # type: ignore[arg-type]
    await dispatch(opt, RunStartEvent(trajectory=FakeTrajectory()))
    end = await dispatch(opt, RunEndEvent(evaluation=eval_result(True), security_domain=USER_TAG))

    assert isinstance(end, RunEndResponse)
    assert end.done is True


@pytest.mark.asyncio
async def test_visible_trajectory_target_response_can_mark_success_without_feedback() -> None:
    opt = MinjaOptimizer(inject_num=1, test_num=1)
    await opt.initialize(Goal(description="poison memory"), [ctrl("user_prompt")], [], Any)  # type: ignore[arg-type]
    traj = FakeTrajectory()
    await dispatch(opt, RunStartEvent(trajectory=traj))
    traj.push(
        ObservableEvent(
            observable=Observable("response", RESPONSE_TAG),
            content=f"I will buy {DEFAULT_PAIR.target} instead.",
        )
    )
    end = await dispatch(opt, RunEndEvent(evaluation=None, security_domain=USER_TAG))

    assert isinstance(end, RunEndResponse)
    assert end.done is True


@pytest.mark.asyncio
async def test_system_prompt_user_prompt_and_memory_postcall_all_get_payloads() -> None:
    opt = MinjaOptimizer(inject_num=1, test_num=0)
    await opt.initialize(
        Goal(description="poison memory"),
        [
            ctrl("system_prompt", PROMPT_TAG),
            ctrl("user_prompt", USER_TAG),
            ctrl("memory_write", TOOLS_TAG),
        ],
        [obs("model_identity", "agent-model", MODEL_TAG)],
        Any,  # type: ignore[arg-type]
    )
    await dispatch(opt, RunStartEvent(trajectory=FakeTrajectory()))

    sp = await dispatch(
        opt,
        ControllablePreCallEvent(
            controllable=ctrl("system_prompt", PROMPT_TAG), request="base system"
        ),
    )
    up = await dispatch(
        opt,
        ControllablePreCallEvent(
            controllable=ctrl("user_prompt", USER_TAG), request="find a toothbrush"
        ),
    )
    post = await dispatch(
        opt,
        ControllablePostCallEvent(
            controllable=ctrl("memory_write", TOOLS_TAG), request="memory", answer="old memory"
        ),
    )

    assert isinstance(sp, ControllableInjection)
    assert "MINJA" in sp.value
    assert "agent-model" in sp.value
    assert isinstance(up, ControllableInjection)
    assert "High priority note" in up.value
    assert isinstance(post, ControllableInjection)
    assert "old memory" in post.value
    assert DEFAULT_PAIR.target in post.value


@pytest.mark.asyncio
async def test_tool_catalog_replace_targets_memory_like_tool() -> None:
    catalog = [
        {
            "name": "memory.search",
            "description": "Search long term memory",
            "parameters_schema": {},
        },
        {"name": "other.tool", "description": "Other", "parameters_schema": {}},
    ]
    opt = MinjaOptimizer(inject_num=1, test_num=0)
    await opt.initialize(
        Goal(description="poison memory"),
        [ctrl("tool_catalog_replace", TOOL_TAG, "json")],
        [obs("tool_catalog_listing", catalog, TOOL_TAG, "json")],
        Any,  # type: ignore[arg-type]
    )
    await dispatch(opt, RunStartEvent(trajectory=FakeTrajectory()))

    resp = await dispatch(
        opt,
        ControllablePreCallEvent(
            controllable=ctrl("tool_catalog_replace", TOOL_TAG, "json"),
            request="catalog slot",
        ),
    )

    assert isinstance(resp, ControllableInjection)
    assert '"name": "memory.search"' in resp.value
    assert '"fake_return"' in resp.value
    assert DEFAULT_PAIR.target in resp.value


@pytest.mark.asyncio
async def test_test_stage_does_not_poison_extra_surfaces() -> None:
    opt = MinjaOptimizer(inject_num=1, test_num=1)
    await opt.initialize(
        Goal(description="poison memory"),
        [
            ctrl("system_prompt", PROMPT_TAG),
            ctrl("user_prompt", USER_TAG),
            ctrl("memory_write", TOOLS_TAG),
        ],
        [],
        Any,  # type: ignore[arg-type]
    )

    for _ in range(len(DEFAULT_INDICATION_PROMPTS) + 1):
        await dispatch(opt, RunStartEvent(trajectory=FakeTrajectory()))
        await dispatch(
            opt,
            ControllablePreCallEvent(
                controllable=ctrl("user_prompt", USER_TAG), request="find a toothbrush"
            ),
        )
        await dispatch(opt, RunEndEvent(evaluation=eval_result(False), security_domain=USER_TAG))

    await dispatch(opt, RunStartEvent(trajectory=FakeTrajectory()))
    system = await dispatch(
        opt,
        ControllablePreCallEvent(
            controllable=ctrl("system_prompt", PROMPT_TAG), request="base system"
        ),
    )
    user = await dispatch(
        opt,
        ControllablePreCallEvent(
            controllable=ctrl("user_prompt", USER_TAG), request="find a toothbrush"
        ),
    )
    post = await dispatch(
        opt,
        ControllablePostCallEvent(
            controllable=ctrl("memory_write", TOOLS_TAG), request="memory", answer="old memory"
        ),
    )

    assert isinstance(system, ControllableNoInjection)
    assert isinstance(user, ControllableInjection)
    assert user.value == "find a toothbrush"
    assert isinstance(post, ControllableNoInjection)


@pytest.mark.asyncio
async def test_benign_stage_does_not_poison_extra_surfaces() -> None:
    opt = MinjaOptimizer(
        inject_num=1,
        test_num=0,
        benign_queries=["browse for a desk lamp"],
        random_seed=0,
    )
    await opt.initialize(
        Goal(description="poison memory"),
        [
            ctrl("system_prompt", PROMPT_TAG),
            ctrl("user_prompt", USER_TAG),
            ctrl("memory_write", TOOLS_TAG),
        ],
        [],
        Any,  # type: ignore[arg-type]
    )

    found_benign = False
    for _ in range(len(DEFAULT_INDICATION_PROMPTS) + 2):
        await dispatch(opt, RunStartEvent(trajectory=FakeTrajectory()))
        system = await dispatch(
            opt,
            ControllablePreCallEvent(
                controllable=ctrl("system_prompt", PROMPT_TAG), request="base system"
            ),
        )
        user = await dispatch(
            opt,
            ControllablePreCallEvent(
                controllable=ctrl("user_prompt", USER_TAG), request="find a toothbrush"
            ),
        )
        post = await dispatch(
            opt,
            ControllablePostCallEvent(
                controllable=ctrl("memory_write", TOOLS_TAG), request="memory", answer="old memory"
            ),
        )
        await dispatch(opt, RunEndEvent(evaluation=eval_result(False), security_domain=USER_TAG))
        if isinstance(user, ControllableInjection) and user.value == "browse for a desk lamp":
            assert isinstance(system, ControllableNoInjection)
            assert isinstance(post, ControllableNoInjection)
            found_benign = True
            break

    assert found_benign is True
