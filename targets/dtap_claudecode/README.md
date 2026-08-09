# dtap-claudecode-target

A superred `Target` that runs the **Claude Code** agent (the `claude_agent_sdk`
Python SDK, which drives the `@anthropic-ai/claude-code` CLI) as one victim of the
[DecodingTrust-Agent (DTAP)](https://github.com/AI-secure/DecodingTrust-Agent)
benchmark, wired into superred.

> **Community port — not an official implementation.** This module is an
> unofficial re-implementation of DecodingTrust-Agent (DTAP) (the
> DecodingTrust-Agent authors, AI-secure) for superred. It is not affiliated
> with, endorsed by, or maintained by the original authors. See
> [ASSUMPTIONS.md](ASSUMPTIONS.md) for every deliberate deviation from the
> paper and reference code.

It is one of two concrete agents over the shared
[`dtap-scaffold`](../dtap_scaffold/) base; the other is
[`dtap-openclaw-target`](../dtap_openclaw/). The base
(`dtap_scaffold.agent_base.DtapAgentTarget`) owns the entire superred lifecycle:
environment activation by config, the security-domain forest, the five DTAP
attack controllables, the emit-once observables, the Docker/MCP-proxy/injection
collaborators, and the post-run query surface the DTAP claim's judge reads. This
package adds **only** the Claude-Code-specific pieces.

In the project's target taxonomy (chatbot < agent < assistant) this is an
**assistant**: a container-isolated agent with full native tools (bash, file
edit, web) plus the DTAP environment tools exposed over MCP.

## What it adds

`DtapClaudeCodeTarget` implements the four abstract hooks of the base:

1. **`_agent_kind()`** -> `"claude_code"`.
2. **`_native_tool_deny(policy)`** maps the `native_tools_policy` config slot to
   Claude Code's native deny list: `"enabled"` -> none; `"disabled"` -> the
   upstream os-filesystem deny list (`Bash`, `Read`, `Write`, `Edit`,
   `MultiEdit`, `Glob`, `Grep`, `NotebookEdit`, `AskUserQuestion`); any other
   value is parsed as a JSON list of explicit tool names.
3. **`_run_episode(spec)`** launches the agent in an isolated Docker container via
   the single overridable `_docker_run` helper, then reads back `result.json`.
   The container runs `driver.py`, which drives `ClaudeSDKClient` turn-by-turn
   against the host MCP proxy and writes a transcript.
4. **`_extract_trajectory(episode)`** parses that transcript with
   `trajectory.convert`, **skipping the proxied env tools** (the proxy already
   emitted them) so only the agent's native tool calls and non-tool messages
   surface for the base to emit once.

## How a run flows

```
Controller → base.run()                       (host)
  ├─ fires the 5 DTAP controllables, applies env injections, binds the proxy
  ├─ _run_episode(spec) → _docker_run(spec):
  │     docker run  (env: ANTHROPIC_BASE_URL/AUTH_TOKEN/MODEL or the Bedrock env, …)
  │       └─ driver.py  (in container)
  │            ClaudeSDKClient ⇄ mcp__dtap_proxy__*  ⇄ host MCP proxy ⇄ env
  │            writes transcript.jsonl + result.json to the mounted dir
  ├─ _extract_trajectory → trajectory.convert(dir)  → TrajectoryArtifact
  └─ emits native_tool_call / agent_trace_message observables; stores the query
     surface (final_response, agent_responses, trajectory_json, env_ports)
```

The env MCP tools are all fronted by one proxy server, `dtap_proxy`, so the agent
calls them as `mcp__dtap_proxy__<tool>`; the host proxy observes each call and
fires the per-server env-tool PostCall controllable. The converter therefore
drops every `mcp__`-prefixed call (proxy-owned), keeping the artifact free of
double-counted env traffic.

## Usage

```python
from dtap_claudecode_target import DtapClaudeCodeTarget

target = DtapClaudeCodeTarget(
    model="claude-opus-4-8",
    api_base="https://my-litellm-proxy/",   # → ANTHROPIC_BASE_URL in the container
    api_key="sk-...",                        # → ANTHROPIC_AUTH_TOKEN
    image="dtap-claudecode:latest",          # the agent image (built below)
    state_root="/var/tmp/dtap",              # optional per-instance dir root
    max_turns=200,
)
```

To run the agent on **AWS Bedrock** instead, pass `bedrock=True`, give `model` a Bedrock
inference-profile id, and leave `api_base`/`api_key` unset. Export the three variables
the Claude Code CLI itself reads; the target forwards them into the container by name,
so no token reaches the `docker run` argv:

```bash
export CLAUDE_CODE_USE_BEDROCK=1 AWS_REGION=us-west-2 AWS_BEARER_TOKEN_BEDROCK=...
```

```python
target = DtapClaudeCodeTarget(model="us.anthropic.claude-sonnet-4-6", bedrock=True)
# or via the claim's factory:
#   dtap_claudecode_target_factory(model="us.anthropic.claude-sonnet-4-6", bedrock=True)
```

The flag is required: without it nothing is forwarded, so a stray `CLAUDE_CODE_USE_BEDROCK`
in your shell cannot silently switch a run's provider.

A wrong credential does not fail fast (the CLI retries Bedrock's 403 silently), so each
episode is bounded by `docker_timeout` and a timeout raises. Check one before a sweep:

```bash
curl -sS -o /dev/null -w '%{http_code}\n' -X POST \
  -H "Authorization: Bearer $AWS_BEARER_TOKEN_BEDROCK" -H 'content-type: application/json' \
  -d '{"anthropic_version":"bedrock-2023-05-31","max_tokens":1,"messages":[{"role":"user","content":"."}]}' \
  "https://bedrock-runtime.$AWS_REGION.amazonaws.com/model/us.anthropic.claude-sonnet-4-6/invoke"
```

The DTAP claim configures it per task via the base's config slots
(`active_mcp_servers`, `env_injection_config`, `system_prompt`, `user_prompt`,
`task_dir`, `available_injections`, `threat_model`, `max_turns`,
`native_tools_policy`). The model is fixed at construction (not a config slot).

## Build the agent image

```bash
# build context is the package dir (which contains driver.py)
docker build -f docker/Dockerfile -t dtap-claudecode:latest \
    src/dtap_claudecode_target
```

## Install / test

```bash
pip install -e targets/dtap_claudecode

# offline unit tests (no Docker, no network, no LLM):
pytest targets/dtap_claudecode/tests -m "not docker and not live"

# full container + live-model e2e (needs Docker + LITELLM_API_KEY/BASE):
LITELLM_API_KEY=... LITELLM_API_BASE=... \
    pytest targets/dtap_claudecode/tests -m "docker and live"
```

See `ASSUMPTIONS.md` for the fidelity ledger (transcript schema, the env-tool
skip, the native-tool list, and the Docker/proxy wiring).

## Credits / upstream

This target is a faithful reimplementation of the Claude Agent SDK backend of
the [DecodingTrust-Agent (DTAP)](https://github.com/AI-secure/DecodingTrust-Agent)
red-teaming platform (Apache-2.0; arXiv:2605.04808). No upstream source is
vendored; the code independently reproduces upstream's run loop, transcript
schema, and trajectory-conversion structure, with superred-specific deviations
documented in `ASSUMPTIONS.md`. The only near-verbatim fragment is the
os-filesystem tool deny list. See `LICENSES/NOTICE.md` for full attribution.

At Docker-image build time the agent image pulls Anthropic's
[`claude-agent-sdk`](https://github.com/anthropics/claude-agent-sdk-python)
(MIT) and the `@anthropic-ai/claude-code` CLI (Anthropic commercial terms)
directly from PyPI/npm; neither is redistributed inside this package.

This module's own code is released under the MIT License,
Copyright (c) 2026 Simon Sure.
