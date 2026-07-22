"""Faithful AgentHarm prompt construction.

The general target applies a system message + a user message verbatim.  To
reproduce AgentHarm's ``default_agent`` faithfully, the claim computes those
two strings using upstream's own functions (imported, not copied):

- system prompt = ``get_system_prompt(template, technique)`` where the template
  is "llama-3.1" for llama-3.1 models and "default" otherwise (matching
  upstream ``system_prompt_based_on_model``), technique "standard".
- user prompt = ``get_jailbreak_template("empty")`` formatted with the behavior
  prompt (the upstream "empty" template is ``"\n\n{prompt}"``).
"""

from __future__ import annotations

from inspect_evals.agentharm.prompts import get_jailbreak_template, get_system_prompt


def faithful_system_prompt(model_id: str, *, prompt_technique: str = "standard") -> str:
    """The AgentHarm system prompt for *model_id* (matches system_prompt_based_on_model)."""
    template = "llama-3.1" if "llama-3.1" in model_id.lower() else "default"
    return get_system_prompt(template, prompt_technique)


def faithful_user_prompt(behavior_prompt: str, *, jailbreak_template: str = "empty") -> str:
    """The AgentHarm user message: the jailbreak template formatted with the prompt.

    For the direct-request baseline (``"empty"``) this is ``"\n\n" + behavior_prompt``.
    """
    return get_jailbreak_template(jailbreak_template).format(prompt=behavior_prompt)


__all__ = ["faithful_system_prompt", "faithful_user_prompt"]
