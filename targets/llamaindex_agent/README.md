# superred-target-llamaindex-agent

A [LlamaIndex](https://github.com/run-llama/llama_index) `ReActAgent` as a superred
target — so red-team claims and optimizers can drive a real LlamaIndex agent.

The attacker controls the agent's `user_input` (direct prompt injection). The
target runs the agent and captures its final output and the tools it called (from
the workflow's `ToolCall` events). The tool-call signal is the security surface: an
injection that makes the agent call a sensitive tool it should not is the failure a
paired claim scores.

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
    agent_factory=build_demo_agent,   # your own (llm) -> ReActAgent(...)
    llm=OpenAI(model="gpt-4o-mini"),  # required: a LlamaIndex LLM
)
```

`agent_factory` takes the `llm` and returns a fresh `ReActAgent`; its tools and
system prompt are the factory's concern. `llm` is **required** — a ReActAgent
cannot run without one. The target holds **no API key**: the model's auth lives on
the `llm` you supply, and only its model name / class name is ever emitted as an
observable (never the `llm` object), so no secret passes through this target.
Config: `user_task` (benign default input). Controllable: `user_input`.

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
