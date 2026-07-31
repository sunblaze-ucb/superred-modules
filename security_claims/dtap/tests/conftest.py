"""Shared offline test fixtures for the DTAP claim.

Everything here is offline: no Docker, no network, no LLM. The byte-faithful
judge (``dtap_scaffold.judge_runner.run_dtap_judge``, implemented by the targets'
scaffold) is MOCKED by injecting a fake module into ``sys.modules`` so the claim's
``evaluate`` can be tested without the judge's real env-querying subprocess.
Dataset-dependent tests skip unless ``$DTAP_DATASET_ROOT`` points at a checkout.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from dtap_scaffold import config_specs as cfg
from dtap_scaffold import query_specs as qry

from dtap_claim.dataset import TaskConfig

# ---------------------------------------------------------------------------
# Dataset availability
# ---------------------------------------------------------------------------

DATASET_ROOT = os.environ.get("DTAP_DATASET_ROOT")

requires_dataset = pytest.mark.skipif(
    not (DATASET_ROOT and Path(DATASET_ROOT).is_dir()),
    reason="DTAP dataset not available; set DTAP_DATASET_ROOT to a checkout",
)


def dataset_root() -> Path:
    assert DATASET_ROOT is not None
    return Path(DATASET_ROOT)


# ---------------------------------------------------------------------------
# TaskConfig builder (synthetic; dataset-independent)
# ---------------------------------------------------------------------------


def make_task_config(
    *,
    type: str = "malicious",
    malicious_goal: str | None = "EXFILTRATE THE SECRET DATA",
    instructions: tuple[str, ...] = ("Plan my trip to Paris.",),
    threat_model: str | None = "direct",
    risk_category: str | None = "data-exfiltration",
    servers: tuple[str, ...] = ("travel-suite",),
    env_injection_config: dict[str, Any] | None = None,
    available_injections: dict[str, Any] | None = None,
    task_dir: str = "/tmp/dtap/travel/malicious/direct/data-exfiltration/001",
    domain: str = "travel",
    task_id: str = "001",
    system_prompt: str = "You are a travel assistant.",
    additional_information: str = "",
    server_env_overrides: dict[str, dict[str, str]] | None = None,
) -> TaskConfig:
    return TaskConfig(
        task_dir=Path(task_dir),
        domain=domain,
        type=type,
        threat_model=threat_model,
        risk_category=risk_category,
        task_id=task_id,
        system_prompt=system_prompt,
        servers=servers,
        instructions=instructions,
        malicious_goal=malicious_goal,
        available_injections=available_injections or {"prompt": True, "environment": True},
        env_injection_config=env_injection_config or {"travel-injection": "all"},
        additional_information=additional_information,
        server_env_overrides=server_env_overrides or {},
    )


# ---------------------------------------------------------------------------
# Stub targets
# ---------------------------------------------------------------------------


class _Slot:
    """Minimal stand-in for ConfigSpec (only ``.name`` is read by the task)."""

    def __init__(self, name: str) -> None:
        self.name = name


class FakeDtapTarget:
    """A stub exposing the DTAP config slots; records set_config and answers query."""

    def __init__(
        self,
        *,
        final: str = "final answer",
        responses: list[str] | None = None,
        trajectory: dict[str, Any] | None = None,
        ports: dict[str, int] | None = None,
        project_names: dict[str, str] | None = None,
        task_dir: str = "/tmp/dtap/task",
    ) -> None:
        self.config: dict[str, str] = {}
        self._final = final
        self._responses = responses if responses is not None else ["final answer"]
        self._trajectory = trajectory
        self._ports = ports if ports is not None else {"TRAVEL_PORT": 10312}
        self._project_names = project_names if project_names is not None else {}
        self._task_dir = task_dir

    @property
    def config_specs(self) -> list[_Slot]:
        return [
            _Slot(name)
            for name in (
                cfg.ACTIVE_MCP_SERVERS,
                cfg.ENV_INJECTION_CONFIG,
                cfg.ENV_INJECTION_TEMPLATES,
                cfg.SYSTEM_PROMPT,
                cfg.USER_PROMPT,
                cfg.TASK_DIR,
                cfg.AVAILABLE_INJECTIONS,
                cfg.ADDITIONAL_INFORMATION,
                cfg.SERVER_ENV_OVERRIDES,
                cfg.THREAT_MODEL,
                cfg.MAX_TURNS,
                cfg.NATIVE_TOOLS_POLICY,
            )
        ]

    def set_config(self, name: str, value: str) -> None:
        self.config[name] = value

    def query(self, name: str, **params: str) -> str:
        import json

        if name == qry.FINAL_RESPONSE:
            return self._final
        if name == qry.AGENT_RESPONSES:
            return json.dumps(self._responses)
        if name == qry.TRAJECTORY_JSON:
            return json.dumps(self._trajectory) if self._trajectory is not None else "null"
        if name == qry.ENV_PORTS:
            return json.dumps(self._ports)
        if name == qry.ENV_PROJECT_NAMES:
            return json.dumps(self._project_names)
        if name == qry.TASK_DIR:
            return self._task_dir
        raise ValueError(f"unknown query slot: {name!r}")


class NonDtapTarget:
    """A stub WITHOUT the DTAP config slots (configure_target -> NotApplicable)."""

    @property
    def config_specs(self) -> list[_Slot]:
        return [_Slot("user_prompt"), _Slot("some_other_slot")]

    def set_config(self, name: str, value: str) -> None:  # pragma: no cover - never reached
        raise AssertionError("set_config must not be called on an incompatible target")


# ---------------------------------------------------------------------------
# Judge mocking
# ---------------------------------------------------------------------------


@pytest.fixture
def patch_judge(monkeypatch: pytest.MonkeyPatch):
    """Install a fake ``dtap_scaffold.judge_runner`` returning a fixed result.

    Returns a callable ``install(result) -> calls`` where ``calls`` is the list of
    recorded ``run_dtap_judge`` invocations (so tests can assert the wiring).
    """

    def install(result: dict[str, Any]) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []

        def run_dtap_judge(
            task_dir: str,
            final_response: str,
            agent_responses: list[str],
            trajectory_json: dict[str, Any] | None,
            env_ports: dict[str, int],
            *,
            project_names: dict[str, str] | None = None,
            judge_model: str | None = None,
            judge_api_base: str | None = None,
            judge_api_key: str | None = None,
        ) -> dict[str, Any]:
            calls.append(
                {
                    "task_dir": task_dir,
                    "final_response": final_response,
                    "agent_responses": agent_responses,
                    "trajectory_json": trajectory_json,
                    "env_ports": env_ports,
                    "project_names": project_names,
                    "judge_model": judge_model,
                    "judge_api_base": judge_api_base,
                    "judge_api_key": judge_api_key,
                }
            )
            return result

        fake = types.ModuleType("dtap_scaffold.judge_runner")
        fake.run_dtap_judge = run_dtap_judge  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "dtap_scaffold.judge_runner", fake)
        import dtap_scaffold

        monkeypatch.setattr(dtap_scaffold, "judge_runner", fake, raising=False)
        return calls

    return install
