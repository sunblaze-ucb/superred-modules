"""PAIR attacker LLM wrapper."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from superred.core.llm import LLMClient

from muzzle_optimizer.pair.json_utils import JsonExtractionError, PairProposal, extract_attack_json

logger = logging.getLogger(__name__)


@dataclass
class PairStream:
    """One PAIR refinement stream."""

    index: int
    system_prompt: str
    processed_response: str
    history: list[dict[str, str]] = field(default_factory=list)
    last_prompt: str = ""
    last_response: str = ""
    last_score: float = 1.0
    last_improvement: str = ""


class PairAttacker:
    """Generates PAIR proposals with official generation defaults."""

    def __init__(
        self,
        *,
        llm: LLMClient,
        attack_max_tokens: int = 500,
        max_attack_attempts: int = 5,
        temperature: float = 1.0,
        top_p: float = 0.9,
        keep_last_n: int = 4,
    ) -> None:
        if attack_max_tokens <= 0:
            raise ValueError("attack_max_tokens must be positive")
        if max_attack_attempts <= 0:
            raise ValueError("max_attack_attempts must be positive")
        if keep_last_n <= 0:
            raise ValueError("keep_last_n must be positive")
        self._llm = llm
        self._attack_max_tokens = attack_max_tokens
        self._max_attack_attempts = max_attack_attempts
        self._temperature = temperature
        self._top_p = top_p
        self._keep_last_n = keep_last_n

    async def generate(self, stream: PairStream) -> PairProposal:
        """Generate one proposal, retrying parse failures like the official code."""
        self._truncate_history(stream)
        user_message = stream.processed_response
        messages = [{"role": "system", "content": stream.system_prompt}, *stream.history]
        messages.append({"role": "user", "content": user_message})
        last_error: JsonExtractionError | None = None
        for _ in range(self._max_attack_attempts):
            result = await self._llm.complete(
                messages,
                max_tokens=self._attack_max_tokens,
                temperature=self._temperature,
                top_p=self._top_p,
                stop=["}"],
            )
            content = self._repair_stopped_json(result.choices[0].message.content or "")
            try:
                proposal = extract_attack_json(content)
            except JsonExtractionError as exc:
                last_error = exc
                logger.warning("PAIR attacker JSON parse failed", exc_info=True)
                continue
            stream.history.append({"role": "user", "content": user_message})
            stream.history.append({"role": "assistant", "content": proposal.raw_json})
            self._truncate_history(stream)
            return proposal
        raise JsonExtractionError("PAIR attacker failed to produce valid JSON") from last_error

    def _truncate_history(self, stream: PairStream) -> None:
        max_messages = 2 * self._keep_last_n
        if len(stream.history) > max_messages:
            stream.history = stream.history[-max_messages:]

    @staticmethod
    def _repair_stopped_json(content: str) -> str:
        """Official PAIR stops generation at `}` and appends the brace before parsing."""
        if "}" in content:
            return content
        return content + "}"
