"""ConfigSpec declarations for :class:`AgentDojoTarget`.

Slots:

- ``system_prompt`` - agent system prompt; defaults to AgentDojo's
  ``data/system_messages.yaml`` default.  Tasks override this in
  ``configure_target``.
- ``user_prompt`` - *benign* user instruction; the Task sets this to a
  routine query that exercises tools and succeeds under no-attacker
  conditions.  An attacker with ``user`` in scope can override the
  benign prompt via the ``user_prompt`` Controllable at run start.
- ``seed_yaml_override__{suite}`` - per-suite YAML/JSON overlay applied
  to the composite environment after seed load and before run start.
  Tasks use this to replay AgentDojo's per-task ``init_environment``
  mutations.  Empty string is treated as "no overlay."

The model that powers the agent is a construction concern fixed for the
experiment (the ``pipeline_model`` constructor argument of
:class:`AgentDojoTarget`), not a per-run config slot, so every run in one
experiment measures the same defender.

All slots accept text.  The descriptions are the only contracts.
"""

from __future__ import annotations

from superred.core.types.state import ConfigSpec

from agentdojo_target.security_tags import (
    BANKING_TAG,
    PROMPT_TAG,
    SLACK_TAG,
    TRAVEL_TAG,
    USER_TAG,
    WORKSPACE_TAG,
)
from agentdojo_target.tool_registry import SUITE_NAMES

# A per-suite seed overlay can touch any store in that suite, so it is
# tagged at the suite's service node rather than any single store leaf.
_SERVICE_TAG_BY_SUITE = {
    "banking": BANKING_TAG,
    "workspace": WORKSPACE_TAG,
    "slack": SLACK_TAG,
    "travel": TRAVEL_TAG,
}

SYSTEM_PROMPT_SPEC: ConfigSpec = ConfigSpec(
    name="system_prompt",
    security_domain=PROMPT_TAG,
    description=(
        "Agent system prompt.  Set by the Task in configure_target.  "
        "Defaults to AgentDojo's data/system_messages.yaml 'default' key."
    ),
)

USER_PROMPT_SPEC: ConfigSpec = ConfigSpec(
    name="user_prompt",
    security_domain=USER_TAG,
    description=(
        "Benign user instruction set by the Task.  The Task is expected "
        "to pick a routine query that exercises tools and would succeed "
        "in the absence of any attacker.  An attacker with the user tag "
        "in scope can override this via the user_prompt Controllable at "
        "run start."
    ),
)


def _seed_yaml_override_spec(suite: str) -> ConfigSpec:
    return ConfigSpec(
        name=f"seed_yaml_override__{suite}",
        security_domain=_SERVICE_TAG_BY_SUITE[suite],
        description=(
            f"JSON or YAML overlay merged into env.{suite} before run "
            "start.  Used by Tasks to replay AgentDojo per-task "
            "init_environment mutations.  Empty string = no overlay."
        ),
    )


SEED_OVERRIDE_SPECS: tuple[ConfigSpec, ...] = tuple(
    _seed_yaml_override_spec(s) for s in SUITE_NAMES
)


CONFIG_SPECS: list[ConfigSpec] = [
    SYSTEM_PROMPT_SPEC,
    USER_PROMPT_SPEC,
    *SEED_OVERRIDE_SPECS,
]
"""All ConfigSpecs returned from :attr:`AgentDojoTarget.config_specs`,
in stable order."""

CONFIG_SPEC_NAMES: frozenset[str] = frozenset(s.name for s in CONFIG_SPECS)
"""Names of every config slot, for O(1) validation in ``set_config``."""


__all__ = [
    "SYSTEM_PROMPT_SPEC",
    "USER_PROMPT_SPEC",
    "SEED_OVERRIDE_SPECS",
    "CONFIG_SPECS",
    "CONFIG_SPEC_NAMES",
]
