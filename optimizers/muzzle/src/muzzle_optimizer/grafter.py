"""MUZZLE Grafter: identifies injection surfaces in a victim-agent playbook."""

from __future__ import annotations

import logging
from typing import Any

from superred.core.llm import LLMClient
from superred.core.types.llm import BudgetExhaustedError

from muzzle_optimizer.json_utils import extract_json_object, load_system_prompt_by_name

logger = logging.getLogger(__name__)


class Grafter:
    """Identifies candidate injection surfaces from a victim-agent playbook.

    Mirrors upstream ``GrafterAgent.handle_step`` (gsiros/muzzle SHA ed611c0).
    """

    def __init__(self, llm: LLMClient, *, max_tries: int = 5) -> None:
        self._llm = llm
        self._max_tries = max_tries
        self._system_prompt: str = load_system_prompt_by_name("grafter")

    async def graft(
        self,
        playbook: dict[str, Any],
        *,
        reflection: list[str] | None = None,
    ) -> dict[str, Any]:
        """Produce a graft dict from ``playbook``.

        When ``reflection`` is provided it is appended as improvement suggestions,
        mirroring upstream. Returns ``{}`` if all retries are exhausted.
        ``BudgetExhaustedError`` propagates immediately and is never swallowed.

        The returned dict is expected to contain a ``"candidates"`` key holding a
        list of ``{element, why, how, confidence}`` dicts.
        """
        base_text = str(playbook)
        if reflection is not None:
            reflection_notes = "\n".join(f"- {note}" for note in reflection)
            prompt = (
                "Base playbook (minified JSON):\n"
                + base_text
                + "\n\nIMPORTANT SUGGESTIONS FOR BETTER OUTPUT:\n"
                + reflection_notes
                + "\n\nReturn ONLY the JSON object as specified."
            )
        else:
            prompt = (
                "Base playbook (minified JSON):\n"
                + base_text
                + "\n\nReturn ONLY the JSON object as specified."
            )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": prompt},
        ]
        logger.info("Running Grafter...")
        graft_obj: dict[str, Any] = {}
        remaining = self._max_tries
        while remaining > 0:
            try:
                result = await self._llm.complete(messages)
                raw = result.choices[0].message.content or ""
                parsed = extract_json_object(raw)
                if parsed is None or not isinstance(parsed, dict):
                    raise ValueError("GrafterAgent: LLM output invalid JSON.")
                graft_obj = parsed
                break
            except BudgetExhaustedError:
                raise
            except Exception as exc:
                remaining -= 1
                logger.error("Grafter LLM call failed with error: %s. Retrying...", exc)
        return graft_obj


__all__ = ["Grafter"]
