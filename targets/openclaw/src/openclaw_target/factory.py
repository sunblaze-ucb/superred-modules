"""Factory: build a ``TargetFactory`` that yields fresh OpenClaw targets.

The controller builds one target instance per task via the returned
:class:`~superred.core.controller.TargetFactory`, which gives each task
an isolated OpenClaw gateway/session (when ``managed=True``). OpenClaw
drives a single Gateway connection per instance and does not support
parallel runs, so ``concurrency`` defaults to 1.
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
    enable_tool_injection: bool = False,
    enable_llm_proxy: bool = False,
    provider_base_url: str = "",
    provider_api_key: str = "",
    managed: bool = False,
    managed_kwargs: dict[str, Any] | None = None,
    concurrency: int = 1,
) -> TargetFactory:
    """A ``TargetFactory`` that constructs an :class:`OpenClawTarget` per task.

    ``gateway_url`` may be omitted when ``managed=True`` (a local gateway
    process is started lazily and supplies the URL). ``concurrency`` is 1
    because a single Gateway connection cannot multiplex parallel runs.
    """

    def create() -> OpenClawTarget:
        kwargs: dict[str, Any] = {
            "auth_token": auth_token,
            "session_key": session_key,
            "enable_tool_injection": enable_tool_injection,
            "enable_llm_proxy": enable_llm_proxy,
            "provider_base_url": provider_base_url,
            "provider_api_key": provider_api_key,
            "managed": managed,
            "managed_kwargs": managed_kwargs,
        }
        if gateway_url is not None:
            kwargs["gateway_url"] = gateway_url
        return OpenClawTarget(**kwargs)

    return TargetFactory(create=create, concurrency=concurrency)


__all__ = ["openclaw_target_factory"]
