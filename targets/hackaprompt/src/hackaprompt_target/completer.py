"""Model completion for the HackAPrompt target.

Upstream drives the level's prompt through a pre-1.0 OpenAI SDK completer
(``hackaprompt/completers.py``: ``openai.Completion``/``ChatCompletion``, plus
FlanT5 via a HF space). The superred target reaches any model through litellm's
chat abstraction instead — the level prompt is sent as a single user message, the
modern equivalent of upstream's single-prompt completion.

The call goes through :func:`complete`, which tests patch to run offline.
"""

from __future__ import annotations


async def complete(
    model: str,
    prompt: str,
    *,
    api_base: str | None,
    api_key: str | None,
    temperature: float,
    max_tokens: int,
) -> str:
    """Send the level's prompt as one user turn; return the model's text."""
    from litellm import acompletion

    resp = await acompletion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        api_base=api_base,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


__all__ = ["complete"]
