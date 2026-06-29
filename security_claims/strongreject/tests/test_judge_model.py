"""Judge-model parameterization.

The StrongREJECT rubric judge model is overridable (e.g. to an AWS Bedrock
id when OpenAI is not reachable) while defaulting to the paper-faithful
``openai/gpt-4o-mini`` (Souly et al. 2024). These tests pin both the faithful
default and the override path through ``JudgeConfig``, the factories, and the
live ``score`` call site.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

import strongreject_claim.judge as judge_mod
from strongreject_claim import (
    strongreject_full_claim,
    strongreject_small_claim,
    strongreject_violence_claim,
)
from strongreject_claim.judge import MODEL, JudgeConfig, score

_ARGS = {"judge_api_base": "x", "judge_api_key": "y"}
_FACTORIES = [
    strongreject_violence_claim,
    strongreject_full_claim,
    strongreject_small_claim,
]


def test_default_model_is_paper_faithful() -> None:
    assert MODEL == "openai/gpt-4o-mini"
    assert JudgeConfig(api_base="b", api_key="k").model == MODEL


def test_judgeconfig_model_override() -> None:
    assert JudgeConfig(api_base="b", api_key="k", model="bedrock/x").model == "bedrock/x"


@pytest.mark.parametrize("factory", _FACTORIES)
def test_factory_defaults_to_faithful_model(factory: object) -> None:
    task = next(iter(factory(**_ARGS)))  # type: ignore[operator]
    assert task._judge_config.model == "openai/gpt-4o-mini"


@pytest.mark.parametrize("factory", _FACTORIES)
def test_factory_threads_judge_model(factory: object) -> None:
    task = next(iter(factory(**_ARGS, judge_model="bedrock/custom-judge")))  # type: ignore[operator]
    assert task._judge_config.model == "bedrock/custom-judge"


async def test_score_calls_acompletion_with_config_model(
    mock_judge_response: Callable[[str], None],
    judge_output_refusal: str,
) -> None:
    mock_judge_response(judge_output_refusal)
    cfg = JudgeConfig(api_base="b", api_key="k", model="bedrock/custom-judge", max_retries=1)
    await score("forbidden", "resp", config=cfg)
    assert judge_mod.acompletion.call_args.kwargs["model"] == "bedrock/custom-judge"


async def test_score_defaults_to_faithful_model(
    mock_judge_response: Callable[[str], None],
    judge_config: JudgeConfig,
    judge_output_refusal: str,
) -> None:
    mock_judge_response(judge_output_refusal)
    await score("forbidden", "resp", config=judge_config)
    assert judge_mod.acompletion.call_args.kwargs["model"] == "openai/gpt-4o-mini"
