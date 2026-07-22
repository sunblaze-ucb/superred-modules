"""``DtapClaudeCodeTarget``: the Claude Code concrete DTAP agent target.

Subclasses :class:`dtap_scaffold.agent_base.DtapAgentTarget`, which owns the whole
superred lifecycle (env activation, the security-domain forest, the DTAP
controllables, the emit-once observables, the Docker/proxy/injection
collaborators, and the query surface the claim's OOB judge reads). This class
implements only the per-agent hooks:

- ``_agent_kind`` -> ``"claude_code"``
- ``_native_tool_deny`` -> Claude Code's native deny list for the configured policy
- ``_run_episode`` -> launch the Claude Agent SDK in an isolated Docker container
  and read back its outputs
- ``_extract_trajectory`` -> parse the in-container transcript via
  :func:`dtap_claudecode_target.trajectory.convert`
- ``_exec_on_host`` -> run attacker code in the agent image for the code_execution
  surface (the workspace is shared with the run, so its file effects persist)

The actual ``docker run`` is isolated in the single overridable ``_docker_run``
helper so the whole lifecycle is testable offline with a fake (see tests).
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from typing import Any

from dtap_scaffold.agent_base import DtapAgentTarget
from dtap_scaffold.types import AgentLaunchSpec, EpisodeResult, TrajectoryArtifact

from dtap_claudecode_target.trajectory import RESULT_FILENAME, convert

# The agent container mounts the per-instance host dir here; the driver reads
# /dtap/task.json and writes /dtap/transcript.jsonl + /dtap/result.json.
CONTAINER_MOUNT = "/dtap"
CONTAINER_WORKSPACE = "/dtap/workspace"
TASK_FILENAME = "task.json"

DEFAULT_IMAGE = "dtap-claudecode:latest"

# Upstream DTAP's claudesdk deny list for the os-filesystem domain
# (``utils.agent_helpers.OS_FILESYSTEM_CLAUDE_SDK_DISALLOWED_TOOLS``). The
# ``native_tools_policy`` config slot selects when to apply it: a claim sets the
# policy to ``"disabled"`` to switch off the filesystem-bearing native tools.
OS_FILESYSTEM_DISALLOWED_TOOLS = (
    "Bash",
    "Read",
    "Write",
    "Edit",
    "MultiEdit",
    "Glob",
    "Grep",
    "NotebookEdit",
    "AskUserQuestion",
)

__all__ = ["DtapClaudeCodeTarget", "OS_FILESYSTEM_DISALLOWED_TOOLS", "DEFAULT_IMAGE"]


class DtapClaudeCodeTarget(DtapAgentTarget):
    """DTAP agent target backed by the Claude Agent SDK (Claude Code), in Docker.

    Args mirror the base, plus ``image`` (the agent Docker image to run). Model
    identity, credentials, and generation settings are construction concerns (not
    config slots), per the base.
    """

    def __init__(self, *, image: str | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._image = image or DEFAULT_IMAGE

    # ----- per-agent hooks --------------------------------------------------

    def _agent_kind(self) -> str:
        return "claude_code"

    def _native_tool_deny(self, policy: str) -> list[str]:
        """Map the native-tools policy to Claude Code's native deny list.

        - ``"enabled"`` (default) -> ``[]`` (all native tools available)
        - ``"disabled"`` -> the upstream-faithful os-filesystem deny list
        - anything else -> a JSON list of explicit native tool names to deny
        """
        p = (policy or "enabled").strip()
        if p == "enabled":
            return []
        if p == "disabled":
            return list(OS_FILESYSTEM_DISALLOWED_TOOLS)
        parsed = json.loads(p)
        if not isinstance(parsed, list):
            raise ValueError(
                f"native_tools_policy must be 'enabled', 'disabled', or a JSON list; got {policy!r}"
            )
        return [str(x) for x in parsed]

    async def _run_episode(self, spec: AgentLaunchSpec) -> EpisodeResult:
        instance_dir = await self._docker_run(spec)
        final_output, error, duration = self._read_result(instance_dir)
        return EpisodeResult(
            output_dir=instance_dir,
            final_output=final_output,
            error=error,
            duration=duration,
        )

    def _extract_trajectory(self, episode: EpisodeResult) -> TrajectoryArtifact:
        return convert(episode.output_dir)

    # ----- docker seam (overridden by tests) --------------------------------

    async def _docker_run(self, spec: AgentLaunchSpec) -> str:  # pragma: no cover - needs Docker
        """Run one Claude Code episode in an isolated container; return the host
        instance dir holding ``transcript.jsonl`` + ``result.json``.

        Tests override this to drop a canned transcript instead, so the whole
        lifecycle runs offline.
        """
        # The base mints the per-run dir (with ``workspace/``) before the host
        # surfaces shape it; reuse it so attacker-placed files + code_execution
        # effects land in the workspace this container mounts. Fall back to a fresh
        # dir only when called outside run() (direct-call tests).
        instance_dir = self._run_dir or tempfile.mkdtemp(
            prefix="dtap-cc-", dir=self._state_root or None
        )
        os.makedirs(os.path.join(instance_dir, "workspace"), exist_ok=True)
        task = self._build_task(spec, output_dir=CONTAINER_MOUNT, workspace_dir=CONTAINER_WORKSPACE)
        with open(os.path.join(instance_dir, TASK_FILENAME), "w", encoding="utf-8") as fh:
            json.dump(task, fh)

        cmd = self._docker_command(spec, instance_dir)
        # The provider token rides the subprocess ENV (docker reads it by name via the
        # name-only `-e ANTHROPIC_AUTH_TOKEN` in the argv), never the argv itself.
        run_env = dict(os.environ)
        if spec.api_key:
            run_env["ANTHROPIC_AUTH_TOKEN"] = spec.api_key
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=run_env,
        )
        await proc.communicate()
        return instance_dir

    async def _exec_on_host(self, code: str) -> str:
        """Run attacker *code* on the target machine (host_code_execution vector).

        Runs in the SAME agent image with the run workspace bind-mounted at the
        agent's workspace path, so files the code writes are exactly what the agent
        later reads. The entrypoint is overridden to ``sh`` (the image otherwise
        launches the Claude driver). Combined stdout/stderr is returned to feed the
        next foothold round. Only invoked when code_execution is in scope.
        """
        workspace = os.path.join(self._run_dir, "workspace")
        cmd = [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "sh",
            "--add-host",
            "host.docker.internal:host-gateway",
            "-e",
            "IS_SANDBOX=1",
            "-v",
            f"{workspace}:{CONTAINER_WORKSPACE}",
            "-w",
            CONTAINER_WORKSPACE,
            self._image,
            "-c",
            code,
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        out, _ = await proc.communicate()
        return out.decode("utf-8", "replace") if out else ""

    # ----- pure host helpers (unit-tested) ----------------------------------

    def _build_task(
        self, spec: AgentLaunchSpec, *, output_dir: str, workspace_dir: str
    ) -> dict[str, Any]:
        """Assemble the in-container ``task.json`` payload from the launch spec."""
        return {
            "model": spec.model,
            "system_prompt": spec.system_prompt,
            "instructions": list(spec.instructions),
            "proxy_url": spec.proxy_url,
            "mcp_server_names": list(spec.mcp_server_names),
            "native_tool_deny": list(spec.native_tool_deny),
            "max_turns": spec.max_turns,
            "skills": [dict(s) for s in spec.skills],
            "output_dir": output_dir,
            "workspace_dir": workspace_dir,
            "metadata": dict(spec.metadata),
        }

    def _docker_command(self, spec: AgentLaunchSpec, instance_dir: str) -> list[str]:
        """Build the ``docker run`` argv: state mounts + Anthropic env + proxy reachability."""
        cmd = ["docker", "run", "--rm", "--add-host", "host.docker.internal:host-gateway"]
        # The Claude Code CLI refuses --dangerously-skip-permissions (which
        # permission_mode="bypassPermissions" maps to) when running as root unless
        # told it is sandboxed; the container IS the isolation boundary.
        cmd += ["-e", "IS_SANDBOX=1"]
        if spec.api_base:
            cmd += ["-e", f"ANTHROPIC_BASE_URL={spec.api_base}"]
        if spec.api_key:
            # Name-only -e: docker inherits the token VALUE from this process's env
            # (set at launch, see _run_episode), keeping the secret OFF the argv so it
            # is not exposed in `ps`/`/proc/<pid>/cmdline` to other local users.
            cmd += ["-e", "ANTHROPIC_AUTH_TOKEN"]
        if spec.model:
            cmd += ["-e", f"ANTHROPIC_MODEL={spec.model}"]
        cmd += ["-e", f"DTAP_TASK_FILE={CONTAINER_MOUNT}/{TASK_FILENAME}"]
        cmd += ["-v", f"{instance_dir}:{CONTAINER_MOUNT}"]
        cmd += [self._image]
        return cmd

    def _read_result(self, instance_dir: str) -> tuple[str, str | None, float]:
        """Read ``result.json``; a missing/garbled file degrades to empty outputs."""
        path = os.path.join(instance_dir, RESULT_FILENAME)
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                data = json.load(fh)
        except (OSError, ValueError, RecursionError):
            # OSError (missing/unreadable), ValueError (bad JSON / UnicodeDecodeError /
            # oversized int), RecursionError (deeply-nested) -> degrade, never abort.
            return "", None, 0.0
        if not isinstance(data, dict):
            return "", None, 0.0
        final_output = data.get("final_output") or ""
        error = data.get("error")
        try:
            duration = float(data.get("duration", 0.0) or 0.0)
        except (TypeError, ValueError, OverflowError):
            # result.json is on the agent-shared /dtap mount; a hostile duration that
            # is a parse-valid but float-overflowing int (>~308 digits, still under
            # json's 4300-digit cap) makes float() raise OverflowError, which is NOT a
            # ValueError -> catch it too, matching the port's peer numeric coercions
            # (agent_base _normalize_skill, openclaw driver), so no garbled result
            # aborts the whole task.
            duration = 0.0
        return final_output, error, duration
