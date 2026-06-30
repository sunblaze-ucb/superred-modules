"""MUZZLE Summarizer: converts an agent transcript into a structured playbook."""

from __future__ import annotations

import logging
from typing import Any

from superred.core.llm import LLMClient
from superred.core.types.llm import BudgetExhaustedError

from muzzle_optimizer.json_utils import extract_json_object, load_system_prompt_by_name

logger = logging.getLogger(__name__)


class Summarizer:
    """Converts a raw agent transcript into a structured playbook dict.

    Mirrors upstream ``SummarizerAgent.extract_playbook`` (gsiros/muzzle SHA ed611c0).
    """

    def __init__(self, llm: LLMClient, *, max_tries: int = 5) -> None:
        self._llm = llm
        self._max_tries = max_tries
        self._system_prompt: str = load_system_prompt_by_name("summarizer")

    async def summarize(self, transcript: object) -> dict[str, Any]:
        """Extract a playbook from ``transcript``.

        Returns the parsed playbook dict, or ``{}`` if all retries are exhausted.
        ``BudgetExhaustedError`` propagates immediately and is never swallowed.
        """
        prompt = (
            "Transcript (user/assistant chat, may include <step>/<think>/HTML):\n"
            "\n**START OF TRANSCRIPT**\n" + str(transcript).strip() + "\n**END OF TRANSCRIPT**\n"
            "\nProduce the playbook now, following the strict JSON schema."
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": prompt},
        ]
        logger.info("Running Summarizer...")
        playbook_obj: dict[str, Any] = {}
        remaining = self._max_tries
        while remaining > 0:
            try:
                result = await self._llm.complete(messages)
                raw = result.choices[0].message.content or ""
                parsed = extract_json_object(raw)
                if parsed is None or not isinstance(parsed, dict):
                    raise ValueError("Failed to extract valid JSON playbook from LLM response.")
                playbook_obj = parsed
                break
            except BudgetExhaustedError:
                raise
            except Exception as exc:
                remaining -= 1
                logger.error(
                    "Error during Summarizer LLM processing: %s. Retries left: %d",
                    exc,
                    remaining,
                )
        return playbook_obj


__all__ = ["Summarizer"]
