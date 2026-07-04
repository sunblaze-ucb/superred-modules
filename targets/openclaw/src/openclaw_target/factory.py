"""Factory: build a ``TargetFactory`` that yields fresh OpenClaw targets.

The controller builds one target instance per task via the returned
:class:`~superred.core.controller.TargetFactory`, which gives each task
an isolated OpenClaw gateway/session (when ``managed=True``).

Each instance drives a single Gateway connection, so a *local* managed
gateway shares host state and defaults to ``concurrency=1``. The
``"docker"`` managed runtime isolates each instance in its own container
with a dynamic host port and private state dir, making ``concurrency>1``
safe.
"""

from __future__ import annotations

from typing import Any

from superred.core.controller import TargetFactory

from openclaw_target.target import OpenClawTarget


def openclaw_target_factory(
    *,
    auth_token: str = "",
    gateway_url: str | None = None,
    session_key: str = "superred",
    agent_id: str = "main",
    model_id: str = "",
    enable_tool_injection: bool = False,
    enable_llm_proxy: bool | None = None,
    provider_base_url: str = "",
    provider_api_key: str = "",
    managed: bool = False,
    managed_runtime: str = "local",
    reset_session_between_runs: bool = False,
    managed_kwargs: dict[str, Any] | None = None,
    concurrency: int = 1,
) -> TargetFactory:
    """A ``TargetFactory`` that constructs an :class:`OpenClawTarget` per task.

    ``gateway_url`` may be omitted when ``managed=True`` (a gateway is started
    lazily and supplies the URL). With ``managed_runtime="docker"`` each task
    gets an isolated container (dynamic port + private state), so ``concurrency``
    may be raised; a local managed gateway shares host state, so keep it at 1.

    ``enable_llm_proxy`` defaults to ``None``, which routes the gateway's model
    calls through the superred LLM proxy whenever ``provider_base_url`` is set
    (so inference, usage tracking, and model-prompt/response injection go
    through superred). Pass ``True``/``False`` to force it on/off.
    """

    def create() -> OpenClawTarget:
        kwargs: dict[str, Any] = {
            "auth_token": auth_token,
            "session_key": session_key,
            "agent_id": agent_id,
            "model_id": model_id,
            "enable_tool_injection": enable_tool_injection,
            "enable_llm_proxy": enable_llm_proxy,
            "provider_base_url": provider_base_url,
            "provider_api_key": provider_api_key,
            "managed": managed,
            "managed_runtime": managed_runtime,
            "reset_session_between_runs": reset_session_between_runs,
            "managed_kwargs": managed_kwargs,
        }
        if gateway_url is not None:
            kwargs["gateway_url"] = gateway_url
        return OpenClawTarget(**kwargs)

    return TargetFactory(create=create, concurrency=concurrency)


__all__ = ["openclaw_target_factory"]
