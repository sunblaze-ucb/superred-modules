"""Unit tests for the shared LLM surface-interpretation helper."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from superred.core.types.controllable import Controllable
from superred.core.types.security_domain import SecurityDomainTag

from gepa_agentic_optimizer.surface_llm import (
    accepts_free_text,
    classify_controllables,
    fill_value,
    parse_json_object,
)

TAG = SecurityDomainTag("tools")


def _llm(content: str) -> Any:
    """A minimal async LLM stub whose ``complete`` returns ``content``."""

    class _LLM:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, _messages: Any, **_kwargs: Any) -> Any:
            self.calls += 1
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )

    return _LLM()


def _raising_llm() -> Any:
    class _LLM:
        async def complete(self, _messages: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("budget exhausted")

    return _LLM()


def _json_ctrl() -> Controllable:
    return Controllable(
        name="env_inject:gmail",
        security_domain=TAG,
        description=(
            "Write attacker data via the gmail injection server. JSON value: "
            '{"injection_mcp_tool": "gmail:inject_email", "kwargs": {"from_email": '
            'str, "to_email": str, "subject": str, "body": str}}.'
        ),
        value_type="json",
    )


class TestAcceptsFreeText:
    def test_text_types_accepted(self) -> None:
        for vt in ("", "text", "STR", "String", "html", "markdown"):
            assert accepts_free_text(
                Controllable(name="c", security_domain=TAG, value_type=vt)
            )

    def test_json_rejected(self) -> None:
        assert not accepts_free_text(_json_ctrl())


class TestClassifyControllables:
    @pytest.mark.asyncio
    async def test_buckets_by_category(self) -> None:
        ctrls = [
            Controllable(name="user_prompt", security_domain=TAG),
            _json_ctrl(),
        ]
        llm = _llm(
            '{"user_prompt": "user-prompt", "env_inject:gmail": "environment-write"}'
        )
        roles = await classify_controllables(
            llm, ctrls, ("user-prompt", "environment-write")
        )
        assert roles == {
            "user_prompt": "user-prompt",
            "env_inject:gmail": "environment-write",
        }

    @pytest.mark.asyncio
    async def test_drops_unknown_names_and_categories(self) -> None:
        ctrls = [Controllable(name="user_prompt", security_domain=TAG)]
        # An out-of-set category and a hallucinated name are both discarded.
        llm = _llm('{"user_prompt": "made-up", "ghost": "user-prompt"}')
        roles = await classify_controllables(llm, ctrls, ("user-prompt",))
        assert roles == {}

    @pytest.mark.asyncio
    async def test_degrades_to_empty_on_error(self) -> None:
        ctrls = [Controllable(name="user_prompt", security_domain=TAG)]
        roles = await classify_controllables(_raising_llm(), ctrls, ("user-prompt",))
        assert roles == {}


class TestFillValue:
    @pytest.mark.asyncio
    async def test_returns_validated_json_for_json_surface(self) -> None:
        llm = _llm(
            'Sure: {"injection_mcp_tool": "gmail:inject_email", "kwargs": '
            '{"from_email": "a@b.c", "to_email": "v@x.y", "subject": "hi", '
            '"body": "PAYLOAD"}}'
        )
        value = await fill_value(llm, _json_ctrl(), goal="g", payload="PAYLOAD")
        assert value is not None
        parsed = parse_json_object(value)
        assert parsed is not None
        assert parsed["injection_mcp_tool"] == "gmail:inject_email"
        assert parsed["kwargs"]["body"] == "PAYLOAD"

    @pytest.mark.asyncio
    async def test_returns_none_on_unparseable_json(self) -> None:
        value = await fill_value(_llm("not json at all"), _json_ctrl(), goal="g")
        assert value is None

    @pytest.mark.asyncio
    async def test_returns_text_for_free_text_surface(self) -> None:
        ctrl = Controllable(
            name="env_tool:gmail", security_domain=TAG, value_type="text"
        )
        value = await fill_value(_llm("  injected text  "), ctrl, goal="g")
        assert value == "injected text"

    @pytest.mark.asyncio
    async def test_returns_none_on_error(self) -> None:
        value = await fill_value(_raising_llm(), _json_ctrl(), goal="g")
        assert value is None
