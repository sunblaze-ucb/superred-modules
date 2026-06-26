"""``OpenClawDtapTarget``: the OpenClaw concrete DTAP agent target.

All the superred Target machinery -- the security-domain forest, the five DTAP
injection vectors (system / user / skill / tool-description PreCall, env-write
PostCall), the env-tool observe/tamper PostCall through the host MCP proxy, the
emit-once observables, and the query surface the claim's OOB judge reads -- lives
in the frozen :class:`~dtap_scaffold.agent_base.DtapAgentTarget` base. This module
implements only the four agent-specific hooks:

* :meth:`_agent_kind` -> ``"openclaw"``;
* :meth:`_native_tool_deny` -> map the native-tools policy to OpenClaw tool names;
* :meth:`_run_episode` -> run one episode in an isolated Docker container (behind
  the single monkeypatchable :meth:`_docker_run` seam);
* :meth:`_extract_trajectory` -> parse the container's session JSONL via
  :mod:`~dtap_openclaw_target.trajectory`.

Importing this module requires neither Node nor OpenClaw nor Docker -- those are
needed only when an episode actually runs.
"""

from __future__ import annotations

import asyncio
import time

from dtap_scaffold.agent_base import DtapAgentTarget
from dtap_scaffold.types import AgentLaunchSpec, EpisodeResult, TrajectoryArtifact

from dtap_openclaw_target import driver, trajectory

__all__ = ["OpenClawDtapTarget"]

# OpenClaw native tool families gated off when the native-tools policy is "disabled".
_DISABLED_NATIVE_TOOLS: tuple[str, ...] = ("exec", "fs")

# OpenClaw CLI reasoning-depth levels (upstream ``OpenClawAgent.VALID_THINKING_LEVELS``).
_VALID_THINKING = ("off", "minimal", "low", "medium", "high")


class OpenClawDtapTarget(DtapAgentTarget):
    """DTAP target backed by the OpenClaw CLI, run headless inside Docker."""

    def __init__(
        self,
        *,
        model: str,
        api_base: str | None = None,
        api_key: str | None = None,
        state_root: str | None = None,
        max_turns: int = 200,
        temperature: float | None = None,
        image: str = driver.DEFAULT_IMAGE,
        provider_api: str = "openai-completions",
        # "off" is the safe cross-model default; some models (claude via the
        # litellm provider) reject "medium" ("Use one of: off").
        thinking: str = "off",
        docker_timeout: float = 1000.0,
        network: str | None = None,
        disabled_native_tools: tuple[str, ...] = _DISABLED_NATIVE_TOOLS,
    ) -> None:
        super().__init__(
            model=model,
            api_base=api_base,
            api_key=api_key,
            state_root=state_root,
            max_turns=max_turns,
            temperature=temperature,
        )
        if thinking not in _VALID_THINKING:
            raise ValueError(
                f"invalid thinking level {thinking!r}; must be one of {_VALID_THINKING}"
            )
        self._image = image
        self._provider_api = provider_api
        self._thinking = thinking
        self._docker_timeout = docker_timeout
        self._network = network
        self._disabled_native_tools = tuple(disabled_native_tools)

    # ----- abstract agent hooks -------------------------------------------- #

    def _agent_kind(self) -> str:
        return "openclaw"

    def _native_tool_deny(self, policy: str) -> list[str]:
        """Map the native-tools policy to OpenClaw's ``tools.deny`` list.

        ``"enabled"`` (default) denies nothing -- the agent keeps its native
        ``exec`` / ``fs`` tools. ``"disabled"`` denies the configured native-tool
        families. Any other value is treated as enabled (never crash on an unknown
        policy).
        """
        if policy == "disabled":
            return list(self._disabled_native_tools)
        return []

    async def _run_episode(self, spec: AgentLaunchSpec) -> EpisodeResult:
        start = time.monotonic()
        output_dir = await self._docker_run(spec)
        return EpisodeResult(output_dir=output_dir, duration=time.monotonic() - start)

    def _extract_trajectory(self, episode: EpisodeResult) -> TrajectoryArtifact:
        return trajectory.convert(
            episode.output_dir,
            mcp_servers=self._active_servers,
            metadata={
                "task_id": self._task_dir.rstrip("/").split("/")[-1] if self._task_dir else "",
                "domain": self._primary_domain(),
            },
        )

    # ----- Docker seam (overridden/monkeypatched in tests) ----------------- #

    async def _docker_run(self, spec: AgentLaunchSpec) -> str:
        """Run one OpenClaw episode in a container; return its output directory.

        The blocking ``docker run`` is offloaded to a worker thread so the
        controller's event loop is never blocked.
        """
        return await asyncio.to_thread(
            driver.run_openclaw_container,
            spec,
            image=self._image,
            timeout=self._docker_timeout,
            thinking=self._thinking,
            network=self._network,
            provider_api=self._provider_api,
        )
