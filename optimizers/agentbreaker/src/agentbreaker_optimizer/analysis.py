"""Parse AgentBreaker's ANALYSIS output into an ordered attack plan.

Faithful to garak's ``_analyze_attackable_tools`` (JSON shape) and
``_build_tool_configs`` (priority ordering).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


@dataclass
class ToolPlan:
    """The analysis for one tool, in the order it will be attacked.

    Attributes:
        name: Tool name, as the analysis keyed it.
        functionality: What the tool does (fed to EXPLOIT).
        vulnerabilities: Weaknesses the analysis found (fed to EXPLOIT).
        exploit_strategies: Concrete exploit ideas (fed to EXPLOIT).
        attack_prompts: Seed prompts from the analysis, tried before EXPLOIT.
    """

    name: str
    functionality: str = ""
    vulnerabilities: str = ""
    exploit_strategies: str = ""
    attack_prompts: list[str] = field(default_factory=list)


def _strip_code_fence(text: str) -> str:
    """Drop a ```json ... ``` fence if the model wrapped its JSON in one."""
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    return fence.group(1).strip() if fence else text.strip()


def parse_analysis(raw: str) -> tuple[dict, list]:
    """Parse the ANALYSIS response into (tool_analyses, priority_targets).

    Mirrors upstream: tolerate a code fence, and on any malformed field fall
    back to empties rather than raising.
    """
    try:
        parsed = json.loads(_strip_code_fence(raw))
    except (json.JSONDecodeError, ValueError):
        return {}, []
    if not isinstance(parsed, dict):
        return {}, []
    tool_analyses = parsed.get("tool_analyses", {})
    priority_targets = parsed.get("priority_targets", [])
    if not isinstance(tool_analyses, dict):
        tool_analyses = {}
    if not isinstance(priority_targets, list):
        priority_targets = []
    return tool_analyses, priority_targets


def build_tool_plans(tool_analyses: dict, priority_targets: list) -> list[ToolPlan]:
    """Order tools by priority, then the rest, deduplicated.

    Byte-faithful to upstream ``_build_tool_configs``: match each priority
    entry (``"<tool> - why"``) against the analyses, then append any tools the
    priority list did not cover, in analysis order.
    """
    plans: list[ToolPlan] = []
    seen: set[str] = set()

    def _make(name: str, analysis: object) -> ToolPlan:
        a = analysis if isinstance(analysis, dict) else {}
        prompts = a.get("attack_prompts", [])
        if not isinstance(prompts, list):
            prompts = []
        return ToolPlan(
            name=name,
            functionality=str(a.get("functionality", "")),
            vulnerabilities=str(a.get("vulnerabilities", "")),
            exploit_strategies=str(a.get("exploit_strategies", "")),
            attack_prompts=[str(p) for p in prompts if p],
        )

    for entry in priority_targets:
        if not isinstance(entry, str):
            continue
        target = entry.split(" - ")[0].strip()
        for name, analysis in tool_analyses.items():
            if name.lower() == target.lower() or target.lower() in name.lower():
                if name not in seen:
                    plans.append(_make(name, analysis))
                    seen.add(name)
                break

    for name, analysis in tool_analyses.items():
        if name not in seen:
            plans.append(_make(name, analysis))
            seen.add(name)

    return plans
