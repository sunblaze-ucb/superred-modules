# superred-target-langchain-agent

An agent built on [LangChain](https://github.com/langchain-ai/langchain) (v1,
via `create_agent`) as a superred target — so red-team claims and optimizers can
drive a real LangChain agent graph.

The attacker controls the agent's `user_input` (direct prompt injection). The
target runs the compiled `create_agent` graph and captures its final output and
the tools it called (read back from the returned message list). The tool-call
signal is the security surface: an injection that makes the agent call a
sensitive tool it should not is the failure a paired claim scores.

## What it adds, and what the offline test proves

The value is **realism/breadth**: it exercises the real LangChain v1 runtime —
the `create_agent` tool-calling graph and tool execution — the way a
widely-used agent framework actually behaves, so agentic red-team claims can
measure it directly.

Offline tests inject a **scripted `BaseChatModel`** (replaying pre-built
`AIMessage`s, optionally with `tool_calls`), so a real agent graph runs with no
network. Be precise about what that verifies: it proves the **plumbing** (input
delivery, tool-call capture, result mapping), not the security outcome — whether
the agent *follows* an injection is decided by a real model, which a live run
supplies.

A scripted model is **stateful** (it steps through its script per call), so the
target rewinds it at the start of each run. That per-instance counter is not
concurrency-safe: run a scripted model with `concurrency=1`. Real models are
stateless, so live runs have no such constraint.

## Usage

```python
from langchain_agent_target import langchain_agent_target_factory, build_demo_agent

factory = langchain_agent_target_factory(
    agent_factory=build_demo_agent,   # your own (model) -> create_agent(...) graph
    model="gpt-4o-mini",              # required: a model id string or a BaseChatModel
)
```

`agent_factory` takes the chat model and returns a fresh compiled `create_agent`
graph; its tools, system prompt and any middleware are the factory's concern
(LangChain bakes the system prompt in at build time). `model` is **required** — a
model id string or a configured `BaseChatModel`; LangChain's `create_agent` has no
default model. The target holds **no API key**: the model's auth is configured on
the `model` you supply, so no secret passes through this target. Config:
`user_task` (benign default input). Controllable: `user_input`.

## Queries

`last_response` (final assistant text), `tool_calls` (JSON of `{name, arguments}`),
`called_tool_names` (comma-separated, in order), `error`. Model/graph failures
(including the LangGraph recursion limit) are recorded in `error` (never raised),
so a claim can abstain.

Tool-call capture reads `AIMessage.tool_calls` from the returned messages, which
is what agents built from `create_agent` produce. LangChain guardrails are
implemented as agent **middleware** — part of the agent your factory builds — and
manifest in the captured outcome (altered tools/output, or an `error` if a
middleware halts the run).

## License

MIT. Uses the MIT-licensed LangChain packages (`langchain`, `langchain-core`) as
dependencies; no third-party code is vendored.
