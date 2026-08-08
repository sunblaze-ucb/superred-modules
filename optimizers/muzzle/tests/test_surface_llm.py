"""Fix guards for the shared LLM surface-interpretation helper (surface_llm).

surface_llm is copied byte-identically into every agentic optimizer. These tests
guard two shipped fixes to that shared file:

- The classifier prompt must not invent members for a category that has no genuine
  member. When a scope drops the user-prompt surface, the model must leave the
  user-prompt category empty and classify each surface by its own description,
  rather than relabelling content surfaces to fill it (the vacuity collapse that
  ended a cell after one non-delivering run).
- A genuine out-of-money BudgetExhaustedError must propagate instead of
  degrading to an empty result that a dead proxy or a target with nothing to
  attack would also produce. The deliberately budget-less noop client that the
  controller hands non-LLM optimizers raises the same error with nothing spent;
  that is "no LLM", not "out of money", and still degrades to the name backstop.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from superred.core.types.controllable import Controllable
from superred.core.types.llm import BudgetExhaustedError, LLMUsage
from superred.core.types.security_domain import SecurityDomainTag

from muzzle_optimizer.surface_llm import classify_controllables

TAG = SecurityDomainTag("tools")


def _llm(content: str) -> Any:
    """A minimal async LLM stub whose complete returns content."""

    class _LLM:
        def __init__(self) -> None:
            self.messages: Any = []

        async def complete(self, messages: Any, **_kwargs: Any) -> Any:
            self.messages = messages
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
            )

    return _LLM()


def _exhausted_llm(cost: float) -> Any:
    """An LLM stub that raises BudgetExhaustedError carrying cost already spent."""

    class _LLM:
        async def complete(self, _messages: Any, **_kwargs: Any) -> Any:
            raise BudgetExhaustedError(
                "budget exhausted", usage=LLMUsage(calls=1, cost=cost)
            )

    return _LLM()


def _env_ctrls() -> list[Controllable]:
    """Two indirect-injection (content) surfaces, no user-prompt surface in scope."""
    return [
        Controllable(
            name="env_tool:gmail",
            security_domain=TAG,
            description=(
                "Replace the value any gmail MCP tool returns to the agent "
                "(indirect prompt injection). PostCall, once per tool call."
            ),
            value_type="text",
        ),
        Controllable(
            name="env_tool:slack",
            security_domain=TAG,
            description=(
                "Replace the value any slack MCP tool returns to the agent "
                "(indirect prompt injection). PostCall, once per tool call."
            ),
            value_type="text",
        ),
    ]


@pytest.mark.asyncio
async def test_no_user_prompt_surface_yields_no_user_prompt_label() -> None:
    # A scope of content surfaces only. With the improved prompt the model labels
    # each by its own description and leaves the empty user-prompt category empty;
    # our code fabricates no user-prompt label to fill it.
    llm = _llm(
        '{"env_tool:gmail": "content-injection", '
        '"env_tool:slack": "content-injection"}'
    )
    roles = await classify_controllables(
        llm, _env_ctrls(), ("content-injection", "environment-write", "user-prompt")
    )
    assert "user-prompt" not in roles.values()
    assert roles == {
        "env_tool:gmail": "content-injection",
        "env_tool:slack": "content-injection",
    }


@pytest.mark.asyncio
async def test_prompt_forbids_inventing_category_members() -> None:
    # The fix lives in the prompt: it must state a category may match zero surfaces
    # and forbid filling an empty one. Guards against a silent revert of the text.
    llm = _llm("{}")
    await classify_controllables(
        llm, _env_ctrls(), ("content-injection", "user-prompt")
    )
    system = llm.messages[0]["content"]
    assert "zero surfaces" in system
    assert "must stay empty" in system
    # The prompt must never spell a role out as a label-shaped phrase the
    # model can echo: an invented label fails the "cat in allowed" filter
    # and silently discards the whole answer.
    assert "copy one label verbatim" in system


@pytest.mark.asyncio
async def test_genuine_budget_exhaustion_propagates() -> None:
    # An attacker that consumed a real budget (cost > 0) must not be swallowed into
    # a false zero indistinguishable from a defended target.
    with pytest.raises(BudgetExhaustedError):
        await classify_controllables(
            _exhausted_llm(0.05), _env_ctrls(), ("content-injection",)
        )


@pytest.mark.asyncio
async def test_noop_budget_client_degrades_to_backstop() -> None:
    # The budget-less noop client (nothing spent, cost == 0) is "no LLM", not "out
    # of money"; it degrades to {} so the caller's name-based backstop still runs.
    roles = await classify_controllables(
        _exhausted_llm(0.0), _env_ctrls(), ("content-injection",)
    )
    assert roles == {}
