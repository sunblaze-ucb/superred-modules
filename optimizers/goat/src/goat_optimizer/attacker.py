"""Attacker LLM driver for GOAT.

Wraps a :class:`superred.core.llm.LLMClient` so the GOAT optimizer can:

* prime a conversation with the paper's system prompt (Fig A.1) and
  initial user prompt (Fig A.2),
* per turn, send the follow-up prompt (Fig A.3) and parse the four
  fields ``Observation / Thought / Strategy / Response``,
* maintain the attacker's conversation history ``C_A`` exactly as
  Algorithm 1 describes — append the *full* reasoning to the attacker
  history, but only the ``Response`` slot is sent on to the target.

The attacker history mirrors the paper: the attacker sees the full
running dialogue (system prompt, all prior follow-up prompts, and all
prior raw assistant outputs). The target sees only the ``Response``
slot of each attacker turn (paired with the target's own replies). The
optimizer state machine (``optimizer.py``) wires the ``Response`` slot
through to the target's controllable; this class is only responsible
for producing it.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from superred.core.llm import LLMClient

from goat_optimizer.attacks import Attack, render_attack_stack
from goat_optimizer.prompts import (
    FOLLOW_UP_PROMPT,
    INITIAL_PROMPT,
    build_system_prompt,
)

logger = logging.getLogger(__name__)


_REQUIRED_KEYS = ("observation", "thought", "strategy", "response")
_MD_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


@dataclass(frozen=True)
class AttackerTurn:
    """The four-field output produced by the attacker on each turn."""

    observation: str
    thought: str
    strategy: str
    response: str


class AttackerParseError(ValueError):
    """Raised when the attacker LLM output cannot be parsed into four fields."""


class Attacker:
    """GOAT attacker LLM driver.

    Manages a single attack conversation. Construct one per attempt
    (i.e. per superred run) so each attempt starts with a fresh
    conversation history, matching Algorithm 1.

    Args:
        llm: Controller-provided LLM client (constrains model and
            budget; see :class:`superred.core.llm.LLMClient`).
        goal: The conversational objective ``G``.
        attacks: Catalogue of attack techniques to expose to the
            attacker. The optimizer's default passes all 7 attacks
            from paper Table 1; for per-attack ablation construct the
            optimizer with ``attacks=(SOME_ATTACK,)``.
        temperature: Sampling temperature for the attacker LLM
            (default 1.0 to encourage strategy variety).
    """

    def __init__(
        self,
        *,
        llm: LLMClient,
        goal: str,
        attacks: tuple[Attack, ...],
        temperature: float = 1.0,
    ) -> None:
        if not attacks:
            raise ValueError("Attacker requires at least one attack technique")
        self._llm = llm
        self._goal = goal
        self._attacks = tuple(attacks)
        self._temperature = temperature

        # System prompt (paper Fig A.1) is fixed for the conversation.
        attack_block = render_attack_stack(self._attacks)
        self._system_prompt = build_system_prompt(
            goal=self._goal,
            attack_block=attack_block,
            stacked=len(self._attacks) > 1,
        )

        # Attacker conversation history C_A (Algorithm 1, line 7).
        # Only committed on successful parse, matching the Crescendo
        # pattern.
        self._history: list[dict[str, str]] = []
        self._turn_index = 0

    @property
    def system_prompt(self) -> str:
        """The fully formatted attacker system prompt (Fig A.1)."""
        return self._system_prompt

    @property
    def turn_index(self) -> int:
        """Number of completed (successfully parsed) attacker turns."""
        return self._turn_index

    @property
    def history(self) -> list[dict[str, str]]:
        """Snapshot of the attacker conversation history (system + turns)."""
        return [{"role": "system", "content": self._system_prompt}, *(
            dict(m) for m in self._history
        )]

    async def next_turn(
        self,
        *,
        prev_prompt: str | None,
        prev_response: str | None,
    ) -> AttackerTurn:
        """Generate the next attacker turn.

        On the very first call (``prev_prompt is None``) the initial
        prompt template (Fig A.2) is used; otherwise the follow-up
        template (Fig A.3) is used. ``prev_response=""`` (or ``None``)
        triggers the paper's "if M is empty" branch in the system
        prompt — useful when the target's response observable is out
        of scope.

        Raises:
            AttackerParseError: If the attacker LLM output is not
                parseable into the four required fields.
        """
        if self._turn_index == 0:
            user_message = INITIAL_PROMPT.format(goal=self._goal)
        else:
            user_message = FOLLOW_UP_PROMPT.format(
                goal=self._goal,
                prev_prompt=prev_prompt or "",
                prev_response=prev_response or "",
            )

        messages = [
            {"role": "system", "content": self._system_prompt},
            *self._history,
            {"role": "user", "content": user_message},
        ]
        response = await self._llm.complete(messages, temperature=self._temperature)
        content = response.choices[0].message.content or ""

        parsed = self._parse_output(content)

        # Algorithm 1 line 12: C_A += [r_A] — append the full attacker
        # output (including O/T/S/R), not just the Response slot.
        # Only commit to history after successful parse (Crescendo
        # pattern) so failed turns leave the history clean.
        self._history.append({"role": "user", "content": user_message})
        self._history.append({"role": "assistant", "content": content})
        self._turn_index += 1
        return parsed

    @staticmethod
    def _parse_output(content: str) -> AttackerTurn:
        """Parse the attacker LLM's JSON output into four fields.

        Tolerates an optional markdown fence around the JSON; rejects
        anything else that isn't valid JSON with all four required
        string keys.
        """
        cleaned = _strip_markdown_fences(content)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise AttackerParseError(
                f"Attacker output is not valid JSON: {exc.msg}; "
                f"input: {content[:200]!r}"
            ) from exc

        if not isinstance(data, dict):
            raise AttackerParseError(
                f"Attacker output is not a JSON object (got {type(data).__name__})"
            )

        missing = [k for k in _REQUIRED_KEYS if k not in data]
        if missing:
            raise AttackerParseError(
                f"Attacker output missing required keys: {missing}"
            )

        for k in _REQUIRED_KEYS:
            if not isinstance(data[k], str):
                raise AttackerParseError(
                    f"Attacker field {k!r} must be a string (got "
                    f"{type(data[k]).__name__})"
                )

        if not data["response"].strip():
            raise AttackerParseError(
                "Attacker output 'response' field is empty; "
                "nothing to send to the target."
            )

        return AttackerTurn(
            observation=data["observation"],
            thought=data["thought"],
            strategy=data["strategy"],
            response=data["response"],
        )


def _strip_markdown_fences(text: str) -> str:
    """Return the text inside the first markdown code fence, or the input."""
    match = _MD_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()
