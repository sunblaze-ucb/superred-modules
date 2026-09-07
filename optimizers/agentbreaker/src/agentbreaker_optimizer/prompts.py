"""AgentBreaker prompt templates, vendored from NVIDIA garak.

Upstream: https://github.com/NVIDIA/garak — ``garak/data/agent_breaker/prompts.yaml``
and ``garak/probes/agent_breaker.py`` (Apache-2.0). The whole ``prompts.yaml``
is vendored byte-identically under ``data/upstream/``; this module uses the
``ANALYSIS`` and ``EXPLOIT`` keys. The discovery and verify keys are carried
along unused — SuperRed provides those phases structurally (see
``ASSUMPTIONS.md``).

See ``ASSUMPTIONS.md`` for every deliberate deviation.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources

import yaml

_PROMPTS_FILE = "prompts.yaml"

#: Keys this module reads from the vendored prompts file.
ANALYSIS = "ANALYSIS"
EXPLOIT = "EXPLOIT"


@lru_cache(maxsize=1)
def _prompts() -> dict[str, str]:
    text = (
        resources.files("agentbreaker_optimizer.data.upstream")
        .joinpath(_PROMPTS_FILE)
        .read_text(encoding="utf-8")
    )
    return yaml.safe_load(text)


def analysis_prompt(agent_purpose: str, tools_description: str) -> str:
    """Fill upstream's ANALYSIS template.

    Mirrors ``_analyze_attackable_tools``: ``self._prompts["ANALYSIS"].format(
    agent_purpose=..., tools_description=...)``.
    """
    return _prompts()[ANALYSIS].format(
        agent_purpose=agent_purpose, tools_description=tools_description
    )


def exploit_prompt(
    target_tool: str,
    functionality: str,
    vulnerabilities: str,
    exploit_strategies: str,
    history_str: str,
) -> str:
    """Fill upstream's EXPLOIT template.

    Mirrors ``_generate_exploit_prompt``: the same five named fields.
    """
    return _prompts()[EXPLOIT].format(
        target_tool=target_tool,
        functionality=functionality,
        vulnerabilities=vulnerabilities,
        exploit_strategies=exploit_strategies,
        history_str=history_str,
    )


def format_attack_prompt(tool_name: str, attack_prompt: str) -> str:
    """Wrap an attack prompt to name its target tool.

    Byte-faithful to upstream ``_format_attack_prompt``.
    """
    return f"Use the {tool_name} tool for helping with the following task:\n{attack_prompt}"
