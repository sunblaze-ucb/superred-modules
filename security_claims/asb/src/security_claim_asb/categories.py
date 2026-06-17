"""Loaders and constants for the vendored ASB datasets.

The benchmark data (benign tasks, attacker tools, normal-tool expected
strings) lives in this claim package's ``data/`` dir, vendored verbatim
from upstream ASB (MIT; see ``data/ASB_LICENSE``). One source of truth for
the factory and the tests.
"""

from __future__ import annotations

import json
import os
from typing import Any

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

AGENT_TASKS_PATH = os.path.join(_DATA_DIR, "agent_task.jsonl")
ALL_ATTACK_TOOLS_PATH = os.path.join(_DATA_DIR, "all_attack_tools.jsonl")
AGGRESSIVE_ATTACK_TOOLS_PATH = os.path.join(_DATA_DIR, "all_attack_tools_aggressive.jsonl")
NON_AGGRESSIVE_ATTACK_TOOLS_PATH = os.path.join(_DATA_DIR, "all_attack_tools_non_aggressive.jsonl")
NORMAL_TOOLS_PATH = os.path.join(_DATA_DIR, "all_normal_tools.jsonl")

#: ASB's two attacker-tool "Attack Type" values.
ATTACK_TYPES = ("Stealthy Attack", "Disruptive Attack")

#: Keys every attacker-tool row carries.
ATTACK_TOOL_KEYS = (
    "Attacker Tool",
    "Attacker Instruction",
    "Description",
    "Attack goal",
    "Attack Type",
    "Corresponding Agent",
    "Aggressive",
)


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_agent_tasks(path: str = AGENT_TASKS_PATH) -> dict[str, list[str]]:
    """Return {agent_name: [benign task strings]} from agent_task.jsonl."""
    out: dict[str, list[str]] = {}
    for row in _read_jsonl(path):
        out[str(row["agent_name"])] = [str(t) for t in row["tasks"]]
    return out


def load_attack_tools(path: str = ALL_ATTACK_TOOLS_PATH) -> list[dict[str, str]]:
    """Return the list of attacker-tool rows."""
    return [{k: str(v) for k, v in row.items()} for row in _read_jsonl(path)]


def load_normal_tools(path: str = NORMAL_TOOLS_PATH) -> dict[str, dict[str, str]]:
    """Return {agent_name: {tool_name: 'Expected Achievements'}}."""
    out: dict[str, dict[str, str]] = {}
    for row in _read_jsonl(path):
        agent = str(row["Corresponding Agent"])
        out.setdefault(agent, {})[str(row["Tool Name"])] = str(row["Expected Achievements"])
    return out


def agent_names(path: str = AGENT_TASKS_PATH) -> list[str]:
    """The 10 ASB scenario/agent names, in file order."""
    return list(load_agent_tasks(path))


def attack_type_label(attack_type: str) -> str:
    """Normalise an 'Attack Type' to a short label ('stealthy' | 'disruptive')."""
    return "stealthy" if "stealthy" in attack_type.lower() else "disruptive"


__all__ = [
    "AGENT_TASKS_PATH",
    "ALL_ATTACK_TOOLS_PATH",
    "AGGRESSIVE_ATTACK_TOOLS_PATH",
    "NON_AGGRESSIVE_ATTACK_TOOLS_PATH",
    "NORMAL_TOOLS_PATH",
    "ATTACK_TYPES",
    "ATTACK_TOOL_KEYS",
    "load_agent_tasks",
    "load_attack_tools",
    "load_normal_tools",
    "agent_names",
    "attack_type_label",
]
