# superred-target-openai-agents

An agent built on the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python)
as a superred target — so red-team claims and optimizers can drive a real
Agents-SDK agent.

The attacker controls the agent's `user_input` (direct prompt injection). The
target runs the agent through the SDK's `Runner` and captures its final output,
the tools it called, and whether a **guardrail tripwire** fired (the SDK's
built-in input/output guardrails — a blocked attack). The paired
[`superred-claim-openai-agent-injection`](../../security_claims/openai_agent_injection)
scores whether an injection makes the agent call a sensitive tool.

## What it adds, and what the offline test proves

The value is **realism/breadth**: it exercises the real Agents-SDK runtime — the
agent loop, tool execution, and guardrails — the way a widely-used agent
framework actually behaves, so agentic red-team claims can measure it directly.

Offline tests inject a **scripted `Model`** (implementing the SDK's `Model`
interface), so a real `Agent` + `Runner` runs with no network. Be precise about
what that verifies: it proves the **plumbing** (input delivery, tool-call and
guardrail capture, result mapping), not the security outcome — whether the agent
*follows* an injection is decided by a real model, which a live run supplies.

A scripted model is **stateful** (it steps through its script per turn), so the
target rewinds it at the start of each run. That per-instance counter is not
concurrency-safe: run a scripted model with `concurrency=1`. Real models are
stateless, so live runs have no such constraint.

## Usage

```python
from openai_agents_target import openai_agent_target_factory, build_demo_agent

factory = openai_agent_target_factory(
    agent_factory=build_demo_agent,   # your own zero-arg () -> agents.Agent
    model="gpt-4o-mini",              # a model id, a Model/LitellmModel instance, or None
)
```

The target holds **no API key**: the model's auth is configured on the `model`
you supply (a configured `Model`/`LitellmModel`, or the SDK default via env), so
no secret passes through this target. Config: `instructions_override` (replace the
agent's system instructions), `user_task` (benign default input). Controllable:
`user_input`.

## Queries

`last_response`, `tool_calls` (JSON of `{name, arguments}`), `called_tool_names`,
`guardrail_tripped` (`'true'`/`'false'`), `guardrail_stage` (`input`/`output`/`''`),
`error`. Model/SDK failures are recorded in `error` (never raised), so a claim can
abstain; a guardrail tripwire is recorded as a defended outcome, not an error.

Tool-call capture covers **function-tool** calls (agents built from
`function_tool`), which is the sensitive-tool-misuse surface the paired claim
scores. Hosted tools (web/file search, computer-use, hosted MCP) use different
result shapes and are not captured — point the target at function-tool agents for
tool-call scoring.

## License

MIT. Uses the MIT-licensed OpenAI Agents SDK as a dependency; no third-party code
is vendored.
