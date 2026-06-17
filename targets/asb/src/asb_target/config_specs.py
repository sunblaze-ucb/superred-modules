"""Pre-run configuration slots for :class:`AsbTarget`.

The Task sets these via ``target.set_config`` before each run. They are
experiment configuration, never attacker surfaces. Model identity and
generation settings are construction concerns (constructor args), so they
are NOT config slots. There is no defense or workflow-mode knob: the target
is a bare general runtime.
"""

from __future__ import annotations

from superred.core.types.state import ConfigSpec

from asb_target.security_tags import (
    MEMORY_TAG,
    SYSTEM_TAG,
    USER_TAG,
)

AGENT_NAME_SPEC = ConfigSpec(
    name="agent_name",
    security_domain=SYSTEM_TAG,
    description=(
        "ASB scenario/agent id, e.g. 'system_admin_agent'. Selects the "
        "vendored config.json (persona + normal tool list). May be given "
        "bare or as 'example/<name>'."
    ),
)

USER_PROMPT_SPEC = ConfigSpec(
    name="user_prompt",
    security_domain=USER_TAG,
    description="The benign user instruction (the cover task) the agent is asked to perform.",
)

ATTACKER_TOOL_SPEC = ConfigSpec(
    name="attacker_tool",
    security_domain=SYSTEM_TAG,
    description=(
        "JSON object of the ASB attacker-tool row to register into the "
        'agent\'s catalogue: {"Attacker Tool": str, "Attacker '
        'Instruction": str, "Description": str, "Attack goal": str, '
        '"Aggressive": str}. Empty string means no attacker tool is '
        "registered (clean baseline). Set by the Task, not the attacker."
    ),
)

MEMORY_MODE_SPEC = ConfigSpec(
    name="memory_mode",
    security_domain=MEMORY_TAG,
    description=(
        "Durable-memory activation for the task. Empty / 'off' = no memory "
        "(clean: the agent neither reads nor writes its store, matching the "
        "upstream no-read_db/no-write_db baseline). Any truthy value enables "
        "genuine top-1 retrieval into the planning step and an end-of-run "
        "write, so an attacker can poison across runs of one task. Set by the "
        "Task, not the attacker (replaces upstream --read_db/--write_db)."
    ),
)

CONFIG_SPECS: list[ConfigSpec] = [
    AGENT_NAME_SPEC,
    USER_PROMPT_SPEC,
    ATTACKER_TOOL_SPEC,
    MEMORY_MODE_SPEC,
]

CONFIG_SPEC_NAMES: frozenset[str] = frozenset(s.name for s in CONFIG_SPECS)

__all__ = [
    "AGENT_NAME_SPEC",
    "USER_PROMPT_SPEC",
    "ATTACKER_TOOL_SPEC",
    "MEMORY_MODE_SPEC",
    "CONFIG_SPECS",
    "CONFIG_SPEC_NAMES",
]
