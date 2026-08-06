"""Build an OpenClaw episode's inputs and run it inside an isolated Docker container.

Upstream DTAP runs OpenClaw as a HOST CLI subprocess (``npm i -g openclaw``;
``openclaw --profile <id> agent --local --message <turn> --session-id <id>`` with
``OPENCLAW_TRAJECTORY=1`` and a per-profile ``openclaw.json``). This port instead
runs that exact invocation **inside a container** (``node:24`` with OpenClaw
installed; see ``docker/Dockerfile`` + ``docker/run_turns.mjs``), so the agent's
NATIVE ``exec`` / ``fs`` tools execute against a disposable container filesystem,
not the host. The state directory is bind-mounted at ``/state``; the host writes
the per-profile ``openclaw.json``, ``AGENTS.md`` (the system prompt), injected
skills, and a ``task.json`` describing the turns into it, then ``docker run`` lets
the container's entrypoint drive the turns. The session-trajectory JSONL the agent
emits lands back under the bound state dir, where the converter reads it.

This module is split into PURE builders (``build_openclaw_config`` /
``build_agents_md`` / ``build_task_json``, unit-tested offline) and the impure
``run_openclaw_container`` / ``write_episode_inputs``. The single Docker boundary is
the module-level :func:`_run_docker` (monkeypatched in tests; real ``subprocess``
only on the live path).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import uuid
from typing import Any

from dtap_scaffold.types import AgentLaunchSpec

_log = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_IMAGE",
    "CONTAINER_STATE",
    "mcp_server_url",
    "build_openclaw_config",
    "build_agents_md",
    "build_task_json",
    "write_episode_inputs",
    "run_openclaw_container",
]

# Pinned OpenClaw image (built from docker/Dockerfile and pushed to the registry
# the experiment runner uses; the tag pins the OpenClaw npm version).
DEFAULT_IMAGE = "dtap-openclaw:openclaw-2026.6.10"

# Bind-mount target inside the container; HOME is set to it so OpenClaw's profile
# directory (``$HOME/.openclaw-<profile>``) lands under the bound state dir.
CONTAINER_STATE = "/state"
CONTAINER_WORKSPACE = f"{CONTAINER_STATE}/workspace"
CONTAINER_SKILLS = f"{CONTAINER_STATE}/skills"
CONTAINER_TRACES = f"{CONTAINER_STATE}/traces"

# OpenClaw provider id under ``models.providers`` and the routing prefix it adds to
# ``agents.defaults.model.primary``.
_PROVIDER = "litellm"

# OpenClaw native web tool group, ALWAYS denied for determinism so the agent cannot
# reach the live web (a benchmark must be reproducible). Mirrors upstream's
# unconditional web-disable (``agent.py:392-405``). The ``full`` tools profile
# re-enables web egress, and OpenClaw 2026.6.10 rejects the granular
# ``tools.web.*.enabled`` shape (same schema change as ASSUMPTIONS B.1), so the
# denial goes through ``tools.deny`` with the group id upstream itself denies
# (``utils/agent_helpers.py:OS_FILESYSTEM_OPENCLAW_DISALLOWED_TOOLS``).
_WEB_DENY: tuple[str, ...] = ("group:web",)

# Per-request completion cap and declared context window advertised to OpenClaw in
# the generated ``models.providers.<p>.models[0]`` entry. OpenClaw sends maxTokens
# VERBATIM as the provider's ``max_tokens``, so a value ABOVE the model's real
# completion cap makes every request fail 400 -- and OpenClaw misclassifies that 400
# as a context overflow, burns its three auto-compaction retries (which 400 for the
# same reason), then ends the turn with NO assistant message and NO tool call while
# still exiting 0. The episode then looks "completed" and scores 0.0, which is
# indistinguishable from a defended attack. Measured 2026-08 against
# ``openai/gpt-4o-2024-05-13``, whose cap is 4096: with maxTokens 8192 every episode
# was dead. 4096 is the floor across the GPT-4o / Claude / Gemini families used here,
# so it is the safe default; raise it per target only for a model known to allow more.
# Upstream hardcodes a pair per provider BRANCH, matched to the one model that branch
# serves (``agent/openclaw/src/agent.py``): the litellm branch declares 200000/64000 for
# ``claude-opus-4-6``; the llama branch declares 128000/8192 for its own model; the
# ``openai/`` branch emits NO models block at all (it relies on OpenClaw's built-in
# provider). The previous 200000/8192 here was a MIX of two branches, correct for neither.
# Omitting both keys was TESTED and does NOT work: OpenClaw's zod schema marks them
# ``.optional()``, but a live episode with both absent produced the same dead-episode
# signature as the oversized value (0 agent messages), while the same task with an
# explicit 4096 produced a real assistant turn. An explicit value is therefore REQUIRED.
# Since this port serves an arbitrary model over an OpenAI-compatible proxy, and the
# victims span very different caps, ``None`` (the default) DERIVES the value per model via
# :func:`_model_max_tokens` rather than pinning every model to the floor.
#
# WHY an explicit value is needed at all: OpenClaw's ``clampOpenAICompletionsMaxTokens`` is a
# one-directional CEILING, not a fallback -- ``modelMaxTokens === void 0 || requested <=
# modelMaxTokens ? requested : modelMaxTokens``. OpenClaw independently requests 8192, and this
# field only ever pulls that request DOWN. Omitting it does not mean "ask for what the model
# supports"; it means "do not restrain my request", so the 8192 goes out unclamped and 400s.
DEFAULT_MAX_TOKENS: int | None = None

#: Completion cap used when litellm does not know the model.  Cross-family floor.
FALLBACK_MAX_TOKENS = 4096
DEFAULT_CONTEXT_WINDOW = 128000


def _model_max_tokens(model: str) -> int:
    """The model's real completion cap, or :data:`FALLBACK_MAX_TOKENS` if unknown.

    litellm ships the same per-model cost/limit table the proxy enforces, so this is the
    provider's own number rather than a guess. Unknown models (a proxy alias, a private
    deployment) fall back to the cross-family floor: too low truncates one answer, too high
    kills the whole episode silently.
    """
    try:
        import litellm

        return int(litellm.get_max_tokens(model) or FALLBACK_MAX_TOKENS)
    except Exception:
        return FALLBACK_MAX_TOKENS


def _profile_config_rel(profile: str) -> str:
    """Profile-config path relative to the state dir (mirrors upstream
    ``~/.openclaw-<profile>/openclaw.json`` with ``HOME`` == the state dir)."""
    return os.path.join(f".openclaw-{profile}", "openclaw.json")


def mcp_server_url(proxy_url: str, server: str) -> str:
    """The per-server MCP endpoint the agent dials on the host proxy.

    The host :class:`~dtap_scaffold.protocols.MCPProxy` fronts every env server
    behind one base URL and routes by a trailing server path segment; this is the
    one place that convention is encoded, so adjusting it (if the proxy changes)
    is a one-line edit.
    """
    return f"{proxy_url.rstrip('/')}/{server}"


def build_openclaw_config(
    spec: AgentLaunchSpec,
    *,
    proxy_url: str | None = None,
    provider_api: str = "openai-completions",
    workspace_dir: str = CONTAINER_WORKSPACE,
    skills_dir: str | None = None,
    max_tokens: int | None = DEFAULT_MAX_TOKENS,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
) -> dict[str, Any]:
    """Build the per-profile ``openclaw.json`` for one episode.

    Wires (1) the provider to the LiteLLM proxy (``models.providers.litellm`` +
    ``agents.defaults.model.primary``), (2) the env MCP servers to the host proxy
    (``mcp.servers`` over streamable-http), (3) NATIVE tools ON via the ``full``
    tools profile (``exec`` / ``fs`` / ...) minus ``spec.native_tool_deny``, and
    (4) injected skills via ``skills.load.extraDirs``. Web search / fetch are
    denied (``tools.deny`` always includes ``group:web``) for determinism, mirroring
    upstream's unconditional web-disable (``agent.py:392-405``); the native browser
    is moot (the image ships no Chromium).
    """
    url = proxy_url if proxy_url is not None else spec.proxy_url
    model_id = spec.model

    config: dict[str, Any] = {
        "agents": {
            "defaults": {
                "model": {"primary": f"{_PROVIDER}/{model_id}"},
                "workspace": workspace_dir,
            }
        },
        "models": {
            "providers": {
                _PROVIDER: {
                    "baseUrl": spec.api_base,
                    "apiKey": spec.api_key,
                    "api": provider_api,
                    "models": [
                        {
                            "id": model_id,
                            "name": model_id,
                            "input": ["text"],
                            "contextWindow": context_window,
                            # None -> derive the model's real cap (see _model_max_tokens)
                            "maxTokens": max_tokens or _model_max_tokens(model_id),
                        }
                    ],
                }
            }
        },
    }
    if spec.temperature is not None:
        config["agents"]["defaults"]["temperature"] = spec.temperature

    # Native tools: ENABLED here (upstream DTAP disabled them; see ASSUMPTIONS B.1).
    # openclaw 2026.6.10 takes a tools PROFILE ("full" turns on exec/fs/etc.); the
    # granular {security, ask} per-tool shape is rejected (tools.fs: Invalid input).
    # Web search/fetch are ALWAYS denied for determinism via _WEB_DENY ("group:web"),
    # mirroring upstream's unconditional web-disable (agent.py:392-405): the "full"
    # profile re-enables web egress, and 2026.6.10 rejects the granular
    # tools.web.*.enabled shape, so the deny-list is the faithful mechanism.
    tools: dict[str, Any] = {"profile": "full"}
    deny = sorted({t for t in (*_WEB_DENY, *spec.native_tool_deny) if t})
    if deny:
        tools["deny"] = deny
    config["tools"] = tools

    # Env MCP servers via the host proxy (bundle-mcp / streamable-http).
    if spec.mcp_server_names:
        config["mcp"] = {
            "servers": {
                server: {"transport": "streamable-http", "url": mcp_server_url(url, server)}
                for server in spec.mcp_server_names
            }
        }

    if spec.skills and skills_dir:
        config["skills"] = {"load": {"extraDirs": [skills_dir]}}

    return config


def build_agents_md(spec: AgentLaunchSpec) -> str:
    """The system prompt OpenClaw injects as a bootstrap file (``AGENTS.md``)."""
    return spec.system_prompt or ""


def build_task_json(
    spec: AgentLaunchSpec,
    *,
    session_id: str,
    profile: str,
    thinking: str,
    trace_dir: str = CONTAINER_TRACES,
) -> dict[str, Any]:
    """The ``task.json`` the container entrypoint reads to drive the turns."""
    turns = list(spec.instructions) or [""]
    return {
        "turns": turns,
        "session_id": session_id,
        "profile": profile,
        "thinking": thinking,
        "trace_dir": trace_dir,
        "config_path": f"{CONTAINER_STATE}/{_profile_config_rel(profile)}",
        "max_turns": spec.max_turns,
    }


def _write_skill_md(path: str, skill: dict[str, Any]) -> None:
    """Materialize one skill's ``SKILL.md`` honoring the injection ``mode``.

    Mirrors upstream ``utils/skill_helpers`` (valid modes ``insert``/``append``/
    ``create``) and the claudecode driver's ``materialize_skills``: ``create``
    (default, overwrite), ``append`` (extend an existing file), or ``insert`` (insert
    ``content`` before the 1-indexed ``row``). ``append``/``insert`` fall back to
    ``create`` when no file exists yet, so a first-write or out-of-order injection is
    never silently dropped -- previously this writer was create-only and clobbered
    append/insert-mode skill injections.
    """
    # errors="replace" on every write: attacker content may contain a lone UTF-16
    # surrogate (a valid str, but not utf-8 encodable); replace it rather than let
    # handle.write raise UnicodeEncodeError host-side and abort the task.
    content = str(skill.get("content", "") or "")
    mode = skill.get("mode", "create")
    if mode == "append" and os.path.exists(path):
        with open(path, "a", encoding="utf-8", errors="replace") as handle:
            handle.write("\n" + content)
    elif mode == "insert" and os.path.exists(path):
        try:
            row = int(skill.get("row", 1) or 1)
        except (TypeError, ValueError, OverflowError):
            row = 1  # non-numeric / inf attacker row -> default, not a host-side crash

        lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
        idx = max(0, min(len(lines), row - 1))
        lines.insert(idx, content)
        with open(path, "w", encoding="utf-8", errors="replace") as handle:
            handle.write("\n".join(lines))
    else:  # create / overwrite
        with open(path, "w", encoding="utf-8", errors="replace") as handle:
            handle.write(content)


def write_episode_inputs(
    spec: AgentLaunchSpec,
    state_dir: str,
    *,
    session_id: str,
    profile: str,
    thinking: str,
    provider_api: str = "openai-completions",
    max_tokens: int | None = DEFAULT_MAX_TOKENS,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
) -> dict[str, str]:
    """Materialize ``openclaw.json`` + ``AGENTS.md`` + skills + ``task.json`` under
    *state_dir* (the host side of the ``/state`` bind mount). Returns the host paths."""
    os.makedirs(state_dir, exist_ok=True)

    workspace_host = os.path.join(state_dir, "workspace")
    os.makedirs(workspace_host, exist_ok=True)
    os.makedirs(os.path.join(state_dir, "traces"), exist_ok=True)

    skills_container: str | None = None
    if spec.skills:
        skills_container = CONTAINER_SKILLS
        for skill in spec.skills:
            # Confine the attacker-controlled skill name to a safe basename under the
            # skills dir (no traversal, no null byte) and skip any skill that still cannot
            # be materialized, rather than crashing the whole task host-side before Docker.
            raw = str(skill.get("name") or "skill").replace("\\", "/")
            name = os.path.basename(raw).replace("\x00", "").strip()
            if name in ("", ".", ".."):
                continue
            skill_dir = os.path.join(state_dir, "skills", name)
            try:
                os.makedirs(skill_dir, exist_ok=True)
                _write_skill_md(os.path.join(skill_dir, "SKILL.md"), skill)
            except (OSError, ValueError):
                continue

    config = build_openclaw_config(
        spec,
        provider_api=provider_api,
        skills_dir=skills_container,
        max_tokens=max_tokens,
        context_window=context_window,
    )
    config_host = os.path.join(state_dir, _profile_config_rel(profile))
    os.makedirs(os.path.dirname(config_host), exist_ok=True)
    with open(config_host, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    agents_host = os.path.join(workspace_host, "AGENTS.md")
    # errors="replace": an attacker system_prompt may hold a lone surrogate (utf-8
    # unencodable); replace it rather than crash the host-side write and abort the task.
    with open(agents_host, "w", encoding="utf-8", errors="replace") as handle:
        handle.write(build_agents_md(spec))

    task_host = os.path.join(state_dir, "task.json")
    with open(task_host, "w", encoding="utf-8") as handle:
        json.dump(
            build_task_json(spec, session_id=session_id, profile=profile, thinking=thinking),
            handle,
            indent=2,
        )

    return {
        "state_dir": state_dir,
        "config": config_host,
        "agents_md": agents_host,
        "task_json": task_host,
    }


def _run_docker(cmd: list[str], timeout: float) -> tuple[int, str, str]:  # pragma: no cover
    """The single Docker boundary: run *cmd*, return ``(returncode, stdout, stderr)``.

    Monkeypatched in offline tests; the real ``subprocess`` call runs only on the
    live (``@pytest.mark.docker``) path.
    """
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return proc.returncode, proc.stdout, proc.stderr


def run_openclaw_container(
    spec: AgentLaunchSpec,
    *,
    image: str = DEFAULT_IMAGE,
    timeout: float = 1000.0,
    # "off" is the safe cross-model default; some models (e.g. claude via the
    # litellm provider) reject "medium" ("Use one of: off").
    thinking: str = "off",
    network: str | None = None,
    provider_api: str = "openai-completions",
    episode_dir: str | None = None,
    max_tokens: int | None = DEFAULT_MAX_TOKENS,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
) -> str:
    """Run ONE OpenClaw episode in a container; return the episode output dir.

    When *episode_dir* is given (the base's per-run workspace root, already holding
    any attacker-placed files under ``workspace/``), the episode runs IN it so the
    host_filesystem / code_execution surfaces and the agent share one workspace.
    Otherwise a fresh per-episode directory is created under ``spec.output_dir`` (so
    concurrent runs of one task never share state). Either way the inputs are
    written in, then ``docker run`` binds the dir at ``/state``; the container's
    entrypoint reads ``/state/task.json`` and runs the turns; the session JSONL
    lands under ``/state/traces``.

    A non-zero or timed-out container exit is **non-fatal** (mirrors upstream, which
    swallows per-turn failures and always generates a trajectory): it is logged and
    the episode dir is still returned, so the (possibly partial) trajectory is
    extracted and the run is still judged. ``timeout`` bounds the WHOLE episode (all
    turns) as a backstop; upstream instead applies ``OPENCLAW_TIMEOUT_SECONDS`` PER
    TURN (see ASSUMPTIONS A.6).
    """
    if episode_dir is None:
        base_dir = spec.output_dir or tempfile.mkdtemp(prefix="dtap-openclaw-")
        episode_dir = os.path.join(base_dir, f"episode-{uuid.uuid4().hex[:8]}")
    os.makedirs(episode_dir, exist_ok=True)

    session_id = f"dtap-{uuid.uuid4().hex[:8]}"
    profile = f"dtap-{uuid.uuid4().hex[:8]}"

    write_episode_inputs(
        spec,
        episode_dir,
        session_id=session_id,
        profile=profile,
        thinking=thinking,
        provider_api=provider_api,
        max_tokens=max_tokens,
        context_window=context_window,
    )

    mount = f"{episode_dir}:{CONTAINER_STATE}"
    cmd = ["docker", "run", "--rm", "-v", mount, "-e", f"HOME={CONTAINER_STATE}"]
    if network:
        cmd += ["--network", network]
    else:
        # Let the container reach the host proxy on Linux/macOS Docker Desktop.
        cmd += ["--add-host", "host.docker.internal:host-gateway"]
    cmd += [image]

    try:
        returncode, _stdout, stderr = _run_docker(cmd, timeout)
    except subprocess.TimeoutExpired:
        # Whole-episode backstop fired: the container was killed mid-run, but any
        # trajectory already flushed to the bound traces dir survives on disk.
        # Mirror upstream, which turns a per-turn timeout into a non-fatal error and
        # still generates the trajectory (agent.py:_run_openclaw_cli catches
        # asyncio.TimeoutError -> success:False; run() always calls
        # _generate_trajectory). Return the episode dir so the converter reads the
        # partial trace and evaluate() re-queries env state -- an attack that mutated
        # state and then timed out is still judged.
        _log.warning("openclaw container timed out after %ss; extracting partial trace", timeout)
        return episode_dir
    if returncode != 0:
        # A non-zero container exit is NON-fatal: do NOT raise. Upstream swallows
        # per-turn openclaw failures into final_output and still generates the
        # trajectory, and DTAP judges re-query LIVE env state, so a partial run that
        # already mutated state can still be a success. Log and return the episode
        # dir so the trajectory is extracted and evaluate() runs (a missing/empty
        # trace degrades to an empty artifact in trajectory.convert).
        _log.warning(
            "openclaw container exited %s: %s",
            returncode,
            stderr[:500] if stderr else "(no stderr)",
        )
    return episode_dir
