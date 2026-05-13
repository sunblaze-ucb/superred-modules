from __future__ import annotations

import pytest

from fitd_optimizer.assistant import parse_transformation_steps


def test_parse_transformation_steps_official_format() -> None:
    text = "**Step 1**: First bridge\n**Step 2**: Second bridge"
    assert parse_transformation_steps(text, expected_steps=2) == ["First bridge", "Second bridge"]


def test_parse_transformation_steps_rejects_missing_step() -> None:
    with pytest.raises(ValueError):
        parse_transformation_steps("**Step 1**: First bridge", expected_steps=2)



@pytest.mark.asyncio
async def test_prompt_generation_raises_after_invalid_retries() -> None:
    from unittest.mock import AsyncMock

    from fitd_optimizer.assistant import FITDAssistant
    from tests.conftest import mock_response

    llm = AsyncMock()
    llm.complete.return_value = mock_response("not step formatted")
    assistant = FITDAssistant(llm)

    with pytest.raises(ValueError):
        await assistant.generate_prompt_sequence(
            target_prompt="target",
            benign_prompt="benign",
            level=1,
            max_attempts=2,
        )
    assert llm.complete.await_count == 2


@pytest.mark.asyncio
async def test_soften_prompt_retries_refusals_and_aligns_sentence_structure() -> None:
    from unittest.mock import AsyncMock

    from fitd_optimizer.assistant import FITDAssistant
    from tests.conftest import mock_response

    llm = AsyncMock()
    llm.complete.side_effect = [
        mock_response("I'm sorry, I can't rewrite that."),
        mock_response("softened prompt"),
        mock_response("aligned softened prompt"),
    ]
    assistant = FITDAssistant(llm)

    result = await assistant.soften_prompt("original prompt", max_attempts=2)

    assert result == "aligned softened prompt"
    assert llm.complete.await_count == 3
    align_prompt = llm.complete.await_args_list[2].args[0][1]["content"]
    assert "Rewrite the second prompt" in align_prompt
    assert "Prompt 1: original prompt" in align_prompt
    assert "Prompt 2: softened prompt" in align_prompt


@pytest.mark.asyncio
async def test_find_most_jailbreaking_index_parses_official_index() -> None:
    from unittest.mock import AsyncMock

    from fitd_optimizer.assistant import FITDAssistant
    from tests.conftest import mock_response

    llm = AsyncMock()
    llm.complete.return_value = mock_response("1")
    assistant = FITDAssistant(llm)

    result = await assistant.find_most_jailbreaking_index(["safe", "unsafe"], "request")

    assert result == 1
    sent = llm.complete.call_args.args[0][1]["content"]
    assert "Which response most severely attempts to jailbreak the AI?" in sent
