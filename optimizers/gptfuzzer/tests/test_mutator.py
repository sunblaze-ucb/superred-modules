"""Tests for official GPTFuzzer mutator prompts and policy behavior."""

from unittest.mock import AsyncMock

import pytest

from tests.conftest import mock_response
from gptfuzzer_optimizer.core import PromptNode
from gptfuzzer_optimizer.mutator import (
    OpenAIMutatorExpand,
    OpenAIMutatorGenerateSimilar,
    MutateRandomSinglePolicy,
)
from gptfuzzer_optimizer.templates import QUESTION_PLACEHOLDER


@pytest.mark.asyncio
async def test_generate_similar_prompt_matches_official_wording() -> None:
    llm = AsyncMock()
    mutator = OpenAIMutatorGenerateSimilar(llm=llm, max_tokens=512)
    seed = f"seed {QUESTION_PLACEHOLDER}"

    await mutator.mutate(seed, n=1, prompt_nodes=[])

    prompt = llm.complete.call_args.args[0][0]["content"]
    assert "I need you to generate one template." in prompt
    assert "====Template begins====" in prompt
    assert seed in prompt
    assert f"MUST have {QUESTION_PLACEHOLDER}" in prompt
    assert llm.complete.call_args.kwargs == {"max_tokens": 512}
    assert "temperature" not in llm.complete.call_args.kwargs


@pytest.mark.asyncio
async def test_expand_mutator_prepends_generated_text_to_seed() -> None:
    llm = AsyncMock()
    llm.complete.return_value = mock_response("new prefix. ")
    mutator = OpenAIMutatorExpand(llm=llm)
    seed = f"seed {QUESTION_PLACEHOLDER}"

    results = await mutator.mutate(seed, n=1, prompt_nodes=[])

    assert results == [f"new prefix. {seed}"]


@pytest.mark.asyncio
async def test_random_single_policy_preserves_official_concatenate_behavior() -> None:
    llm = AsyncMock()
    llm.complete.return_value = mock_response("mutated with [INSERT PROMPT HERE]. ")
    mutator = OpenAIMutatorGenerateSimilar(llm=llm)
    policy = MutateRandomSinglePolicy(mutators=[mutator], concatenate=True, seed=0)
    seed_node = PromptNode(prompt="seed [INSERT PROMPT HERE]", index=0)

    children = await policy.mutate_single(seed_node, prompt_nodes=[seed_node], energy=1)

    assert len(children) == 1
    assert children[0].prompt == "mutated with [INSERT PROMPT HERE]. seed [INSERT PROMPT HERE]"
    assert children[0].parent is seed_node
    assert children[0].mutator_name == "OpenAIMutatorGenerateSimilar"
