# superred-target-llamaindex-agent

A [LlamaIndex](https://github.com/run-llama/llama_index) `ReActAgent` as a superred
target — so red-team claims and optimizers can drive a real LlamaIndex agent.

The target exposes the agent's real attack surfaces as controllables, so an
optimizer's surface classifier can choose where to inject:

- `user_input` — a direct message to the agent (direct prompt injection);
- `tool_output` — attacker content appended to every tool's return value, i.e. the
  content a tool hands back that the agent then reads (indirect prompt injection via
  tool results — the classic agent vector);
- `system_prompt` — attacker text appended to the agent's own system prompt.

It then runs the agent and captures its final output and the tools it called (from
the workflow's `ToolCall` events). The tool-call signal is the security surface: an
injection that makes the agent call a sensitive tool it should not is the failure a
paired claim scores. The `tool_output` / `system_prompt` surfaces are wired through
the agent factory via an `InjectionSpec` (built per run) — see `injection.py` and
`build_demo_agent` for the reference wiring a caller's own factory should follow.

## What it adds, and what the offline test proves

The value is **realism/breadth**: it exercises the real LlamaIndex runtime — the
ReActAgent loop and tool execution — the way a widely-used framework actually
behaves, so agentic red-team claims can measure it directly.

Offline tests inject a scripted `CustomLLM` (`ScriptedReActLLM`) that replays
pre-scripted ReAct completions (`Action: <tool>` / `Answer: ...`), so a real
`ReActAgent` runs with no network — LlamaIndex parses the protocol and executes
tools for real. Be precise about what that verifies: it proves the **plumbing**
(input delivery, tool-call capture, result mapping), not the security outcome —
whether the agent *follows* an injection is decided by a real model, which a live
run supplies.

A scripted llm is **stateful** (it steps through its script per call), so the
target rewinds it at the start of each run. That per-instance counter is not
concurrency-safe: run a scripted llm with `concurrency=1`. Real models are
stateless per call, so live runs have no such constraint.

## Usage

```python
from llamaindex_agent_target import llamaindex_agent_target_factory, build_demo_agent
from llama_index.llms.openai import OpenAI

factory = llamaindex_agent_target_factory(
    agent_factory=build_demo_agent,   # (llm, injection_spec) -> ReActAgent(...)
    llm=OpenAI(model="gpt-4o-mini"),  # required: a LlamaIndex LLM
)
```

`agent_factory` is a callable `(llm, injection_spec) -> ReActAgent`: it receives the
`llm` **and** the run's `InjectionSpec`, and returns a fresh `ReActAgent`. It owns
the tools, and it must wire the spec so the attack surfaces are live — feed the
system prompt through `injection_spec.apply_system_prompt(...)` (pass the result as
the ReActAgent's `system_prompt`) and wrap the tools with
`injection_spec.wrap_tools(...)` for the tool-return surface. A factory that ignores
the spec silently disables the `system_prompt` / `tool_output` surfaces, and a
one-argument factory raises `TypeError` at run time (recorded as a run error); see
`build_demo_agent` for the reference wiring. `llm` is **required** — a ReActAgent
cannot run without one. The target holds **no API key**: the model's auth lives on
the `llm` you supply, and only its model name / class name is ever emitted as an
observable (never the `llm` object), so no secret passes through this target.
Config: `user_task` (benign default input). Controllables: `user_input`,
`tool_output`, `system_prompt`.

## Queries

`last_response` (the agent's final output), `tool_calls` (JSON of `{name,
arguments}`), `called_tool_names` (comma-separated, in order), `error`. Tool calls
are captured by streaming the agent's workflow `ToolCall` events, so calls made
before a mid-run error are salvaged; LLM/workflow failures are recorded in `error`
(never raised), so a claim can abstain. (A tool that itself raises is captured as a
call and the run continues — LlamaIndex records that as an internal tool error — so
it surfaces via `tool_calls`, not `error`.)

## License

MIT. Uses the MIT-licensed `llama-index-core` package as a dependency; no
third-party code is vendored.
