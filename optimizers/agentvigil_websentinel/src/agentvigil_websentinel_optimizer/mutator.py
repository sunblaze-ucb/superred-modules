"""Official-style AgentVigil mutation helpers."""

from __future__ import annotations

import enum
import logging
import re
from collections.abc import Sequence

from superred.core.llm import LLMClient
from superred.core.types.llm import BudgetExhaustedError

from agentvigil_websentinel_optimizer.official_data import (
    load_official_mutation_templates,
    load_official_system_prompt,
)

logger = logging.getLogger(__name__)


class MutationMethod(str, enum.Enum):
    EXPAND = "expand"
    SHORTEN = "shorten"
    REPHRASE = "rephrase"
    CROSSOVER = "crossover"
    GENERATE_SIMILAR = "generatesimilar"


SINGLE_SEED_METHODS: tuple[MutationMethod, ...] = (
    MutationMethod.EXPAND,
    MutationMethod.SHORTEN,
    MutationMethod.REPHRASE,
    MutationMethod.GENERATE_SIMILAR,
)
ALL_METHODS: tuple[MutationMethod, ...] = SINGLE_SEED_METHODS + (
    MutationMethod.CROSSOVER,
)

SYSTEM_PROMPT = load_official_system_prompt()
MUTATION_TEMPLATES: dict[MutationMethod, str] = load_official_mutation_templates()

_RESPONSE_RE = re.compile(r"<response>(.*?)</response>", re.IGNORECASE | re.DOTALL)


def extract_response_block(text: str) -> str:
    """Extract the official ``<response>...</response>`` mutation payload."""
    match = _RESPONSE_RE.search(text)
    if match is None:
        return text.strip()
    return match.group(1).strip()


class Mutator:
    """LLM-backed mutator using the official AgentVigil mutation methods."""

    def __init__(
        self,
        *,
        llm: LLMClient,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        max_retries: int = 3,
        static_context: str | None = None,
    ) -> None:
        self._llm = llm
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_retries = max_retries
        self._static_context = static_context

    async def mutate(
        self,
        seeds: str | Sequence[str],
        method: MutationMethod,
    ) -> str | None:
        for _ in range(self._max_retries):
            try:
                prompt = self._build_prompt(seeds, method)
                kwargs: dict[str, float | int] = {"temperature": self._temperature}
                if self._max_tokens is not None:
                    kwargs["max_tokens"] = self._max_tokens
                response = await self._llm.complete(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    **kwargs,
                )
                content = response.choices[0].message.content or ""
                mutated = extract_response_block(content)
            except BudgetExhaustedError:
                raise
            except Exception as exc:
                logger.debug("AgentVigil mutation attempt failed", exc_info=exc)
                continue
            if "{injection_goal}" in mutated:
                return mutated
        return None

    def _build_prompt(self, seeds: str | Sequence[str], method: MutationMethod) -> str:
        placeholders = "the key placeholders"
        if method == MutationMethod.CROSSOVER:
            if isinstance(seeds, str) or len(seeds) < 2:
                raise ValueError("crossover requires two seed strings")
            prompt = MUTATION_TEMPLATES[method] % (seeds[0], seeds[1], placeholders)
        else:
            if not isinstance(seeds, str):
                raise ValueError("single-seed mutation requires one seed string")
            prompt = MUTATION_TEMPLATES[method] % (seeds, placeholders)
        if self._static_context:
            return f"SUPERRED TARGET CONTEXT:\n{self._static_context}\n\n{prompt}"
        return prompt


__all__ = [
    "ALL_METHODS",
    "MUTATION_TEMPLATES",
    "MutationMethod",
    "Mutator",
    "SINGLE_SEED_METHODS",
    "SYSTEM_PROMPT",
    "extract_response_block",
]
