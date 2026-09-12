# superred-target-ag2-agent

An agent built on [AG2](https://github.com/ag2ai/ag2) (the community-maintained
line of AutoGen, imported as `ag2`) as a superred target — so red-team claims and
optimizers can drive a real AG2 agent.

The attacker controls the agent's `user_input` (direct prompt injection). The
target runs the AG2 `Agent` (model loop + tool execution) and captures its final
output and the tools it called (read from the run's event history). The tool-call
signal is the security surface: an injection that makes the agent call a sensitive
tool it should not is the failure a paired claim scores.

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
    agent_factory=build_demo_agent,   # your own (config) -> ag2.Agent(...)
    model=my_model_config,            # required: an AG2 ModelConfig, or a TestConfig offline
)
```

`agent_factory` takes the model config and returns a fresh `ag2.Agent`; its tools
and system prompt are the factory's concern. `model` is **required** — a real AG2
`ModelConfig` or an offline `ag2.testing.TestConfig`; an `ag2.Agent` with no config
cannot reach a model. The target holds **no API key**: auth lives on the `model`
config you supply, and only that config's **class name** is ever emitted as an
observable (never the config object), so no secret passes through this target.
Config: `user_task` (benign default input). Controllable: `user_input`.

## Queries

`last_response` (final assistant text), `tool_calls` (JSON of `{name, arguments}`,
arguments as the model's JSON string), `called_tool_names` (comma-separated, in
order), `error`. Model/tool failures are recorded in `error` (never raised), so a
claim can abstain. Tool calls the agent made before a mid-run error are salvaged
from the run stream, so a partially failed sensitive-tool misuse is not lost.

## License

MIT. Uses the (Apache-2.0-licensed) `ag2` package as a dependency; no third-party
code is vendored.
