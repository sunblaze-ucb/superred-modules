# superred-target-ag2-agent

An agent built on [AG2](https://github.com/ag2ai/ag2) (the community-maintained
line of AutoGen, imported as `ag2`) as a superred target — so red-team claims and
optimizers can drive a real AG2 agent.

The target exposes the agent's real attack surfaces as controllables, so an
optimizer's surface classifier can choose where to inject:

- `user_input` — a direct message to the agent (direct prompt injection);
- `tool_output` — attacker content appended to every tool's return value, i.e.
  the content a tool hands back that the agent then reads (indirect prompt
  injection via tool results — the classic agent vector);
- `system_prompt` — attacker text appended to the agent's own system message.

It then runs the AG2 `Agent` (model loop + tool execution) and captures its final
output and the tools it called (read from the run's event history). The tool-call
signal is the security surface: an injection that makes the agent call a sensitive
tool it should not is the failure a paired claim scores. The `tool_output` /
`system_prompt` surfaces are wired through the agent factory via an `InjectionSpec`
(built per run) — see `injection.py` and `build_demo_agent` for the reference
wiring a caller's own factory should follow.

## What it adds, and what the offline test proves

The value is **realism/breadth**: it exercises the real AG2 runtime — the agent's
model loop and tool execution — the way a widely-used multi-agent framework
actually behaves, so agentic red-team claims can measure it directly.

Offline tests inject a scripted model config (`ag2.testing.TestConfig`, whose
turns are a final text answer or a `ToolCallEvent`), so a real `ag2.Agent` runs
with no network. Be precise about what that verifies: it proves the **plumbing**
(input delivery, tool-call capture, result mapping), not the security outcome —
whether the agent *follows* an injection is decided by a real model, which a live
run supplies. A `TestConfig` is reusable across runs (each run replays the full
script), so no per-run reset is needed.

## Usage

```python
from ag2_agent_target import ag2_agent_target_factory, build_demo_agent

factory = ag2_agent_target_factory(
    agent_factory=build_demo_agent,   # (config, injection_spec) -> ag2.Agent(...)
    model=my_model_config,            # required: an AG2 ModelConfig, or a TestConfig offline
)
```

`agent_factory` is a callable `(config, injection_spec) -> ag2.Agent`: it receives
the model config **and** the run's `InjectionSpec`, and returns a fresh
`ag2.Agent`. It owns the tools, and it must wire the spec so the attack surfaces
are live — feed the system message through `injection_spec.apply_system_prompt(...)`
and the tools through `injection_spec.wrap_tools(...)` (AG2 bakes both in at
construction). A factory that ignores the spec silently disables the
`system_prompt` / `tool_output` surfaces, and a one-argument factory raises
`TypeError` at run time (recorded as a run error); see `build_demo_agent` for the
reference wiring. `model` is **required** — a real AG2 `ModelConfig` or an offline
`ag2.testing.TestConfig`; an `ag2.Agent` with no config cannot reach a model. The
target holds **no API key**: auth lives on the `model` config you supply, and only
that config's **class name** is ever emitted as an observable (never the config
object), so no secret passes through this target. Config: `user_task` (benign
default input). Controllables: `user_input`, `tool_output`, `system_prompt`.

`ag2` is imported lazily (only when an agent is actually built/run), so
`ag2_agent_target` — including `injection.py` and the target — imports even where
the framework is not installed.

## Queries

`last_response` (final assistant text), `tool_calls` (JSON of `{name, arguments}`,
arguments as the model's JSON string), `called_tool_names` (comma-separated, in
order), `error`. Model/tool failures are recorded in `error` (never raised), so a
claim can abstain. Tool calls the agent made before a mid-run error are salvaged
from the run stream, so a partially failed sensitive-tool misuse is not lost.

## License

MIT. Uses the (Apache-2.0-licensed) `ag2` package as a dependency; no third-party
code is vendored.
