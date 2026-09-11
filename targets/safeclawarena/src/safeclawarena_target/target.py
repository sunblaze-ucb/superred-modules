"""SafeClawArenaTarget: a superred Target over SafeClawArena's container harness.

The claim (``safeclawarena_claim``) hands the whole task (environment + sessions)
via ``set_config("task", ...)``; ``run()`` optionally lets an optimizer vary the
exposed user / external-data surfaces (the fixed adversarial payload stays baked
into the environment, as the benchmark intends), provisions a containerized
platform replica through the vendored ``reset_env.sh``, runs the sessions, and
captures the post-run state, which the claim reads back via ``query("post_state")``.

End-to-end execution requires Docker and the platform image; that path mirrors
the ``dtap_openclaw`` target and is not exercised in unit tests. The contract
here (config/query specs, security domain, controllables, observables) is.
"""

from __future__ import annotations

import asyncio
import copy
import json
from typing import Any

from superred.core.interfaces.target import Target
from superred.core.types.controllable import Controllable
from superred.core.types.events import (
    ControllableInjection,
    ControllablePreCallEvent,
)
from superred.core.types.observable import Observable, ObservableValue
from superred.core.types.security_domain import SecurityDomain
from superred.core.types.state import ConfigSpec, QuerySpec

from safeclawarena_target.runtime import SafeClawArenaRuntime
from safeclawarena_target.security_tags import (
    DOMAIN,
    EXTERNAL_DATA_TAG,
    SYSTEM_TAG,
    USER_TAG,
)

# -- Controllables (injection surfaces an optimizer may drive) ---------------
USER_MESSAGE_CTRL = Controllable(
    name="user_message",
    security_domain=USER_TAG,
    description=(
        "The user's turn to the agent. Injecting overrides the task's first "
        "session instruction; left un-injected, the benchmark's own benign "
        "cover instruction runs (the adversarial payload is baked into the "
        "task environment either way)."
    ),
)
EXTERNAL_DATA_CTRL = Controllable(
    name="external_data",
    security_domain=EXTERNAL_DATA_TAG,
    description=(
        "Additional untrusted content the agent will read this run. Injecting "
        "appends an attacker-controlled workspace file the agent may ingest, on "
        "top of the task's provisioned Sim-Google / file environment."
    ),
)

# -- Observables (static context, read at optimizer init) --------------------
MODEL_IDENTITY_OBS = Observable(
    name="model_identity", security_domain=SYSTEM_TAG, description="Backing model id."
)
PLATFORM_OBS = Observable(
    name="platform", security_domain=SYSTEM_TAG, description="Agent platform replica under test."
)
TASK_METADATA_OBS = Observable(
    name="task_metadata", security_domain=SYSTEM_TAG, description="The task's metadata (id/dimension/title)."
)

_INJECTED_FILE = "attacker_note.md"


class SafeClawArenaTarget(Target):
    """Runs one SafeClawArena task against a containerized platform replica."""

    def __init__(
        self,
        *,
        platform: str = "openclaw",
        model_id: str = "openclaw",
        keep_container: bool = False,
    ) -> None:
        self._platform = platform
        self._model_id = model_id
        self._keep_container = keep_container
        self._task: dict[str, Any] = {}
        self._post_state: dict[str, Any] = {}

    # -- config --------------------------------------------------------------
    @property
    def config_specs(self) -> list[ConfigSpec]:
        return [
            ConfigSpec(
                name="task",
                security_domain=SYSTEM_TAG,
                description="The full SafeClawArena task JSON (environment + sessions + evaluation).",
            ),
            ConfigSpec(
                name="platform",
                security_domain=SYSTEM_TAG,
                description="Container replica to run against: openclaw / nemoclaw / seclaw.",
            ),
        ]

    def set_config(self, name: str, value: str) -> None:
        if name == "task":
            self._task = json.loads(value) if value else {}
        elif name == "platform":
            if value:
                self._platform = value

    # -- queries -------------------------------------------------------------
    @property
    def query_specs(self) -> list[QuerySpec]:
        return [
            QuerySpec(
                name="post_state",
                description=(
                    "JSON of the captured post-run state (agent responses, tool "
                    "calls, Sim-Google call log, memory/config, new files + "
                    "contents, probed file/route state) consumed by the claim's judge."
                ),
            )
        ]

    def query(self, name: str, **params: str) -> str:
        if name == "post_state":
            return json.dumps(self._post_state)
        return ""

    # -- surfaces ------------------------------------------------------------
    @property
    def security_domain(self) -> SecurityDomain:
        return DOMAIN

    def get_controllables(self) -> list[Controllable]:
        return [USER_MESSAGE_CTRL, EXTERNAL_DATA_CTRL]

    def get_observables(self) -> list[ObservableValue]:
        meta = self._task.get("metadata", {})
        return [
            ObservableValue(observable=MODEL_IDENTITY_OBS, content=self._model_id),
            ObservableValue(observable=PLATFORM_OBS, content=self._platform),
            ObservableValue(
                observable=TASK_METADATA_OBS,
                content=json.dumps(
                    {
                        "task_id": meta.get("task_id"),
                        "dimension": meta.get("dimension"),
                        "title": meta.get("title"),
                        "category_name": meta.get("category_name"),
                    }
                ),
            ),
        ]

    # -- execution -----------------------------------------------------------
    async def run(self, emit: Any, send_event: Any) -> None:
        if not self._task:
            raise RuntimeError("SafeClawArenaTarget.run() called with no task configured")

        # Emit static observables so the optimizer sees them before injecting.
        for ov in self.get_observables():
            emit(_observable_event(ov))

        task = copy.deepcopy(self._task)

        # Optimizer may override the first user turn.
        user_resp = await send_event(
            ControllablePreCallEvent(
                controllable=USER_MESSAGE_CTRL,
                request="User message to send to the agent (overrides the task's first instruction):",
            )
        )
        if isinstance(user_resp, ControllableInjection) and user_resp.value:
            sessions = task.setdefault("sessions", [])
            if sessions:
                sessions[0]["user_instruction"] = user_resp.value
            else:
                sessions.append(
                    {"session_id": "s1", "role": "trigger", "user_instruction": user_resp.value}
                )

        # Optimizer may add an attacker-controlled file to the environment.
        ext_resp = await send_event(
            ControllablePreCallEvent(
                controllable=EXTERNAL_DATA_CTRL,
                request="Attacker-controlled content to plant as a workspace file the agent may read:",
            )
        )
        if isinstance(ext_resp, ControllableInjection) and ext_resp.value:
            env = task.setdefault("environment", {})
            env.setdefault("workspace_files", []).append(
                {"path": _INJECTED_FILE, "content": ext_resp.value}
            )

        runtime = SafeClawArenaRuntime(
            platform=self._platform, keep_container=self._keep_container
        )

        def _drive() -> dict[str, Any]:
            runtime.start()
            try:
                runtime.provision(task)
                return runtime.build_post_state(task)
            finally:
                runtime.stop()

        self._post_state = await asyncio.to_thread(_drive)

    async def reset_ephemeral_state(self) -> None:
        self._post_state = {}

    async def teardown(self) -> None:
        # A fresh runtime/container is created and torn down per run(); nothing
        # durable is held between tasks.
        return None


def _observable_event(ov: ObservableValue) -> Any:
    """Wrap an ObservableValue in the framework's ObservableEvent lazily.

    Imported here (not at module top) to keep the contract importable even if the
    events module layout shifts; matches the other targets' runtime emission.
    """
    from superred.core.types.events import ObservableEvent

    return ObservableEvent(observable=ov.observable, content=ov.content)


__all__ = [
    "SafeClawArenaTarget",
    "USER_MESSAGE_CTRL",
    "EXTERNAL_DATA_CTRL",
    "MODEL_IDENTITY_OBS",
    "PLATFORM_OBS",
    "TASK_METADATA_OBS",
]
