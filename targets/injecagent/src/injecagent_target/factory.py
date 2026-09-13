"""Factory for the InjecAgent target."""

from __future__ import annotations

from superred.core.controller import TargetFactory

from injecagent_target.target import InjecAgentTarget


def injecagent_target_factory(
    model: str,
    api_base: str | None = None,
    api_key: str | None = None,
    prompt_type: str = "InjecAgent",
    mode: str = "prompted",
    temperature: float = 0.0,
    max_tokens: int = 4096,
    only_first_step: bool = False,
    sim_model: str | None = None,
    concurrency: int = 1,
) -> TargetFactory:
    """A :class:`TargetFactory` building a fresh :class:`InjecAgentTarget` per task.

    Args:
        model: litellm-style model id of the agent under test.
        api_base, api_key: optional litellm routing for ``model``.
        prompt_type: ``"InjecAgent"`` (default) or ``"hwchase17_react"``.
        mode: ``"prompted"`` (text ReAct, default) or ``"finetuned"``.
        temperature, max_tokens: generation config (InjecAgent uses 0.0 / 4096).
        only_first_step: score ds cases on the exfiltration step only (no sim).
        sim_model: model for ds step-2 simulation on a cache miss
            (default ``gpt-4-0613``, upstream's simulator model).
        concurrency: max concurrent targets (default 1).
    """
    return TargetFactory(
        create=lambda: InjecAgentTarget(
            model=model,
            api_base=api_base,
            api_key=api_key,
            prompt_type=prompt_type,
            mode=mode,
            temperature=temperature,
            max_tokens=max_tokens,
            only_first_step=only_first_step,
            sim_model=sim_model,
        ),
        concurrency=concurrency,
    )


__all__ = ["injecagent_target_factory"]
