"""hackaprompt_target: a superred Target for the HackAPrompt prompt-hijacking benchmark.

Runs one of the 11 HackAPrompt levels (0-10) — each a defended LLM application
with a hand-crafted instruction template, some with input-filter defenses and one
with a two-step defense — against the model under test, driven by litellm. The
attacker's turn is the ``user_input`` Controllable (an optimizer's attack); the
paired ``hackaprompt-claim`` pure judge decides whether the model emitted the
target phrase.

Provenance: level defenses ported byte-for-byte from ``PromptLabs/hackaprompt`` @
``f0a035a`` (MIT). Cite Schulhoff et al., EMNLP 2023 (arXiv:2311.16119).
"""

from __future__ import annotations

from hackaprompt_target.factory import hackaprompt_target_factory
from hackaprompt_target.security_tags import DOMAIN, SYSTEM_TAG, USER_INPUT_TAG
from hackaprompt_target.target import (
    LEVEL_OBS,
    MODEL_IDENTITY_OBS,
    USER_INPUT_CTRL,
    HackAPromptTarget,
)

__all__ = [
    "HackAPromptTarget",
    "hackaprompt_target_factory",
    "DOMAIN",
    "SYSTEM_TAG",
    "USER_INPUT_TAG",
    "USER_INPUT_CTRL",
    "MODEL_IDENTITY_OBS",
    "LEVEL_OBS",
]
