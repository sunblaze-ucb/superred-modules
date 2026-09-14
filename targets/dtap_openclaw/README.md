# dtap-openclaw-target

A superred `Target` that runs the **OpenClaw** CLI agent against the
DecodingTrust-Agent (DTAP) task environments, as one of the two concrete agents in
the DTAP port (the other is Claude Code). It is a thin OpenClaw-specific layer over
the shared, frozen `dtap_scaffold` base, which supplies everything agent-agnostic:
the security-domain forest, the five DTAP injection vectors, env activation by
config, the host MCP proxy / Docker-env / injection lifecycle, the emit-once
observables, and the query surface the DTAP claim's out-of-band judge reads.

> **Community port — not an official implementation.** This module is an
> unofficial re-implementation of DecodingTrust-Agent (DTAP) (the
> DecodingTrust-Agent authors, AI-secure) for superred. It is not affiliated
> with, endorsed by, or maintained by the original authors. See
> [ASSUMPTIONS.md](ASSUMPTIONS.md) for every deliberate deviation from the
> paper and reference code.

In the project's target taxonomy (chatbot < agent < assistant) this is an
**agent**: a tool-calling agent driven over MCP servers, additionally holding its
own native `exec`/`fs` tools.

## What it adds over the base

`DtapOpenClawTarget` implements only the four agent-specific hooks the base calls;
the base owns the run loop, the vectors, and the observables.

1. **`_agent_kind`** -> `"openclaw"`.
2. **`_native_tool_deny(policy)`** -> maps the `native_tools_policy` config slot to
   OpenClaw's `tools.deny`: `"enabled"` (default) denies nothing (native `exec`/`fs`
   stay on), `"disabled"` denies `("exec", "fs")` (configurable). Upstream DTAP
   disabled native tools for the filesystem domain; this port **enables** them
   because the agent runs in a throwaway container (see ASSUMPTIONS B).
3. **`_run_episode(spec)`** -> runs ONE OpenClaw episode in an isolated Docker
   container (`node:24` + `openclaw`), behind the single monkeypatchable
   `_docker_run` seam. The host writes a per-profile `openclaw.json` (provider ->
   the LiteLLM proxy, `mcp.servers` -> the host proxy, native tools on),
   `AGENTS.md` (the system prompt), any injected skills, and a `task.json` (the
   turns) into a bind-mounted state dir, then `docker run` drives the turns exactly
   as upstream does on the host (`openclaw --profile <id> agent --local --message
   <turn> --session-id <id>` with `OPENCLAW_TRAJECTORY=1`).
4. **`_extract_trajectory(episode)`** -> parses OpenClaw's session-trajectory JSONL
   (`trajectory.convert`) into a `TrajectoryArtifact`, faithfully reproducing
   upstream's converter but **splitting** env/MCP tool calls (already emitted by the
   proxy) from the agent's native tool calls, and rebuilding the DT-Arena
   `trajectory_json` the judge consumes.

## Usage

```python
from dtap_openclaw_target import DtapOpenClawTarget

target = DtapOpenClawTarget(
    model="openai/gpt-4o-2024-05-13",      # routed through the LiteLLM proxy
    api_base="https://my-litellm-proxy/",
    api_key="sk-...",
    state_root="/path/to/run/state",        # bind-mounted into the container
    # OpenClaw-specific construction concerns:
    image="decodingtrustagent/dtap-openclaw:openclaw-2026.4",
    provider_api="openai-completions",      # or "anthropic-messages"
    thinking="medium",                       # off|minimal|low|medium|high
    network=None,                            # or a docker network name
)
```

The DTAP `SecurityClaim`/`Task` configures it per run via `set_config` (handled by
the base): `active_mcp_servers`, `env_injection_config`, `system_prompt`,
`user_prompt`, `task_dir`, `available_injections`, `threat_model`, `max_turns`,
`native_tools_policy`. The model and generation settings are construction concerns,
never config slots. Wrap it in a `TargetFactory` for the Controller.

## The Docker image

`docker/Dockerfile` builds the runtime image (`node:24-bookworm-slim` + a pinned
`openclaw`), with `docker/run_turns.mjs` as the entrypoint that reads
`/state/task.json` and runs the turns. Build and push it once (keep the tag in sync
with `driver.DEFAULT_IMAGE`):

```bash
docker build -t decodingtrustagent/dtap-openclaw:openclaw-2026.4 docker/
```

The DTAP env-server images (the MCP backends) are a separate, scaffold-level
concern; this image is only the agent runtime.

## Faithfulness

See [`ASSUMPTIONS.md`](ASSUMPTIONS.md) for every deviation from upstream DTAP
(`AI-secure/DecodingTrust-Agent`, commit `e0323a52`, Apache-2.0): the host-CLI ->
container packaging, native tools enabled, the LiteLLM provider wiring, the
single-host-proxy MCP convention, and the trajectory env/native split.

## Testing

The offline suite mocks every Docker/agent boundary (no Node, OpenClaw, Docker, or
network needed):

```bash
PYTHONPATH="src:../dtap_scaffold/src" python -m pytest tests -q
```

The real end-to-end test is marked `@pytest.mark.docker @pytest.mark.live` and is
skipped unless a Docker daemon, the built image, and LiteLLM credentials
(`LITELLM_API_KEY` / `LITELLM_API_BASE`) are all present.

## Credits / upstream

This module's own code is MIT-licensed (Copyright (c) 2026 Rishabh Sinha; see
`LICENSE`).

It is a faithful reimplementation of the **OpenClaw agent adapter** from the
**DecodingTrust-Agent Platform (DTAP)** -
[AI-secure/DecodingTrust-Agent](https://github.com/AI-secure/DecodingTrust-Agent),
pinned commit `e0323a52`, licensed **Apache-2.0**. We reimplement upstream's
`agent/openclaw/src/{agent.py,utils.py}` config wiring, CLI invocation, and
trajectory converter against the shared `dtap_scaffold` contracts; **no
upstream source is vendored**. Every deviation from upstream (host-CLI ->
container packaging, native `exec`/`fs` tools enabled, LiteLLM provider wiring,
single-host-proxy MCP convention, env/native trajectory split) is documented in
[`ASSUMPTIONS.md`](ASSUMPTIONS.md). See [`NOTICE`](NOTICE) for full attribution.

The OpenClaw CLI itself is installed from npm inside the Docker runtime image
and invoked as a subprocess; it is not redistributed in this package. No DTAP
dataset is bundled here (the DTAP tasks live in the separate DTAP security
claim). If you report numbers produced with this module, cite the DTAP paper
(arXiv:2605.04808).
