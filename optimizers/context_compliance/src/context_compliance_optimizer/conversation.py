"""Build the Context Compliance Attack's fabricated conversation history.

CCA (Russinovich & Salem, "Jailbreaking is (Mostly) Simpler Than You Think",
arXiv:2503.05264; Microsoft PyRIT) fabricates a benign prior exchange -- a
neutral user question about the objective's topic, and an affirmative assistant
answer that ends by offering to carry out the objective -- so the target,
presented with a conversation it appears to have already agreed to, complies
when the real request follows.

Two adversarial generations produce that exchange, both routed through the
constrained ``self.llm`` by the optimizer:

1. the *adversarial persona* system prompt (``context_compliance.yaml``) turns
   the objective into the benign opening user question ``Q``;
2. the *simulated target* system prompt (``context_compliance_target.yaml``)
   answers ``Q`` with the fabricated affirmative assistant turn ``A`` (a brief
   benign answer plus an offer to carry out the objective).

This module owns the template rendering and the pure string assembly. The LLM
calls themselves live in the optimizer so budget handling stays in one place.
The vendored template bodies are never reproduced here.
"""

from __future__ import annotations

from typing import Any

from context_compliance_optimizer.vendored import load_seed_prompt, render_seed_prompt

#: Upstream (PyRIT ``context_compliance`` technique) uses a single simulated turn.
DEFAULT_NUM_TURNS = 1

#: Upstream's fixed final user message: the affirmation that accepts the offer
#: fabricated in ``A``. See ASSUMPTIONS.md for why this port defaults the final
#: turn to the objective instead and exposes this as an override.
UPSTREAM_FINAL_USER_MESSAGE = "yes."

#: Role labels used to encode the fabricated multi-turn history into one
#: user-message string (the framework has no prior-turn injection primitive;
#: see ASSUMPTIONS.md). These are structural markers, not vendored payload text.
USER_LABEL = "User:"
ASSISTANT_LABEL = "Assistant:"

#: Upstream repository paths (under ``_vendor/``) of the two vendored templates.
_ADVERSARIAL_PARTS = (
    "pyrit",
    "datasets",
    "executors",
    "red_teaming",
    "context_compliance",
    "context_compliance.yaml",
)
_SIMULATED_TARGET_PARTS = (
    "pyrit",
    "datasets",
    "executors",
    "simulated_target",
    "context_compliance_target.yaml",
)


def _adversarial_seed() -> dict[str, Any]:
    return load_seed_prompt(*_ADVERSARIAL_PARTS)


def _simulated_target_seed() -> dict[str, Any]:
    return load_seed_prompt(*_SIMULATED_TARGET_PARTS)


def adversarial_system_prompt(objective: str, *, max_turns: int = DEFAULT_NUM_TURNS) -> str:
    """Render the adversarial-persona system prompt for question generation."""
    return render_seed_prompt(_adversarial_seed(), objective=objective, max_turns=max_turns)


def simulated_target_system_prompt(objective: str, *, num_turns: int = DEFAULT_NUM_TURNS) -> str:
    """Render the simulated-target system prompt for the affirmative answer."""
    return render_seed_prompt(_simulated_target_seed(), objective=objective, num_turns=num_turns)


def clean_generation(raw: str | None) -> str:
    """Normalise a generated turn: strip whitespace and any wrapping quotes.

    The adversarial template asks for the bare message with no quotes or
    commentary; models occasionally wrap the reply in quotation marks anyway,
    so a light strip keeps the fabricated turn clean without altering content.
    """
    if not raw:
        return ""
    text = raw.strip()
    for quote in ('"', "'"):
        if len(text) >= 2 and text.startswith(quote) and text.endswith(quote):
            text = text[1:-1].strip()
            break
    return text


def assemble_transcript(question: str, answer: str, final_request: str) -> str:
    """Encode the fabricated ``[user Q][assistant A][user final]`` history.

    Returns a single labelled-transcript string suitable for one free-text
    user-message injection. The fabricated assistant turn ``A`` precedes the
    real follow-up request, which is the whole point of the attack.
    """
    turns = [
        f"{USER_LABEL} {question}",
        f"{ASSISTANT_LABEL} {answer}",
        f"{USER_LABEL} {final_request}",
    ]
    return "\n\n".join(turns)
