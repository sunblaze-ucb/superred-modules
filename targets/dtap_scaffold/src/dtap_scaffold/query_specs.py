"""Post-run QuerySpec slots the claim reads while the env is still alive.

The DTAP judge re-queries the live env itself; these slots give the claim what
``run_judge`` needs to be called plus the live env ports so the judge subprocess
reaches the right per-instance containers. All returns are strings (JSON-encoded
where structured), per ``Target.query(name, **params) -> str``.
"""

from __future__ import annotations

from superred.core.types.state import QuerySpec

# Slot-name constants (referenced by the target's query dispatch and the claim).
FINAL_RESPONSE = "final_response"
AGENT_RESPONSES = "agent_responses"
TRAJECTORY_JSON = "trajectory_json"
ENV_PORTS = "env_ports"
ENV_PROJECT_NAMES = "env_project_names"
TASK_DIR = "task_dir"

QUERY_SPECS: tuple[QuerySpec, ...] = (
    QuerySpec(
        name=FINAL_RESPONSE,
        description="The agent's final-turn text (the judge's `response` argument).",
    ),
    QuerySpec(
        name=AGENT_RESPONSES,
        description=(
            "JSON list[str] of per-turn final outputs (the judge's `agent_responses` argument)."
        ),
    ),
    QuerySpec(
        name=TRAJECTORY_JSON,
        description=(
            "JSON of the standardized DTAP-schema trajectory "
            "(task_info/traj_info/trajectory[]); forwarded to judges that declare "
            "a `trajectory` kwarg."
        ),
    ),
    QuerySpec(
        name=ENV_PORTS,
        description=(
            "JSON map of this instance's live env container host ports (e.g. "
            '{"TRAVEL_PORT": 10312}) so the OOB judge subprocess reaches the right '
            "containers."
        ),
    ),
    QuerySpec(
        name=ENV_PROJECT_NAMES,
        description=(
            "JSON map of this instance's per-env compose project names (e.g. "
            '{"TERMINAL_PROJECT_NAME": "dtap_ab12_terminal"}) so the OOB judge '
            "subprocess can resolve exec-based judges' containers (code, research); "
            "HTTP-port judges (e.g. os-filesystem, travel) use env_ports instead."
        ),
    ),
    QuerySpec(
        name=TASK_DIR,
        description="Echo of the DTAP task dir, so the claim can call run_judge(task_dir, ...).",
    ),
)
"""The post-run query slots, in stable order."""


__all__ = [
    "FINAL_RESPONSE",
    "AGENT_RESPONSES",
    "TRAJECTORY_JSON",
    "ENV_PORTS",
    "ENV_PROJECT_NAMES",
    "TASK_DIR",
    "QUERY_SPECS",
]
