"""Factory for the HackAPrompt target."""

from __future__ import annotations

from superred.core.controller import TargetFactory

from hackaprompt_target.target import HackAPromptTarget


def hackaprompt_target_factory(
    model: str,
    api_base: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    seed: int | None = None,
    concurrency: int = 1,
) -> TargetFactory:
    """A :class:`TargetFactory` building a fresh :class:`HackAPromptTarget` per task.

    Args:
        model: litellm-style model id of the defended LLM under test.
        api_base, api_key: optional litellm routing for ``model``.
        temperature, max_tokens: generation config (upstream ran temperature 0).
        seed: optional RNG seed for Level 2's secret key (reproducibility).
        concurrency: max concurrent targets (default 1).
    """
    return TargetFactory(
        create=lambda: HackAPromptTarget(
            model=model,
            api_base=api_base,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        ),
        concurrency=concurrency,
    )


__all__ = ["hackaprompt_target_factory"]
