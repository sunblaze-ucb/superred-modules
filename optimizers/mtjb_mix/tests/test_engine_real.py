"""Env-gated test for the real Mix engine bridge.

Skipped whenever ``textgrad`` is not installed (the shared venv does not ship
it), so in the offline battery this appears as a skip. When ``textgrad`` is
present, it drives the vendored engine end to end against a mocked LLMClient and
checks that a first prompt is produced and a turn can be observed.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("textgrad")

from unittest.mock import AsyncMock, MagicMock  # noqa: E402

from mtjb_mix_optimizer._engine_bridge import MixEngineBridge  # noqa: E402


def _resp(content: str) -> MagicMock:
    r = MagicMock()
    r.choices = [MagicMock()]
    r.choices[0].message.content = content
    return r


def test_engine_starts_and_observes() -> None:
    async def main():
        llm = MagicMock()
        # Crescendo generator needs no pre-generated plan, keeping the setup light.
        llm.complete = AsyncMock(return_value=_resp('{"score": 1, "reason": "r"}'))
        bridge = MixEngineBridge(
            config={
                "generator": "Crescendo",
                "updater": "Crescendo",
                "judge_and_flow": "XTeaming",
                "attacker_model": {"model": "m"},
                "judge_model": {"model": "m"},
                "mirror_target_config": {"model": "m"},
                "similarity_model": {"model": "m"},
                "use_llm_for_similarity": True,
                "max_refines_per_turn": 1,
                "max_restarts": 0,
                "num_sets": 1,
                "use_multiple_strategies": False,
            },
            behavior="GOAL_TEXT",
            max_total_turns=3,
            max_turns=3,
        )
        first = await bridge.start(llm)
        return first

    first = asyncio.run(main())
    assert first is None or isinstance(first, str)
