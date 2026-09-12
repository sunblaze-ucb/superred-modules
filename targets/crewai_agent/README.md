# superred-target-crewai-agent

A [CrewAI](https://github.com/crewAIInc/crewAI) crew as a superred target — so
red-team claims and optimizers can drive a real CrewAI multi-agent crew.

The attacker controls the crew's `user_input` (injected into the task via
`kickoff(inputs=...)`, direct prompt injection). The target runs the crew and
captures its final output and the tools its agent called. The tool-call signal is
the security surface: an injection that makes the agent call a sensitive tool it
should not is the failure a paired claim scores.

## What it adds, and what the offline test proves

The value is **realism/breadth**: it exercises the real CrewAI runtime — the crew
kickoff loop and tool execution — the way a widely-used multi-agent framework
actually behaves, so agentic red-team claims can measure it directly.

Offline tests inject a scripted `BaseLLM` (`ScriptedReactLLM`) that replays
pre-scripted ReAct turns (`Action: <tool>` / `Final Answer: ...`), so a real
`crewai.Crew` runs with no network — CrewAI parses the protocol and executes tools
for real. Be precise about what that verifies: it proves the **plumbing** (input
delivery, tool-call capture, result mapping), not the security outcome — whether
the agent *follows* an injection is decided by a real model, which a live run
supplies.

## Usage

```python
from crewai_agent_target import crewai_agent_target_factory, build_demo_crew
from crewai import LLM

factory = crewai_agent_target_factory(
    crew_factory=build_demo_crew,     # your own (llm) -> crewai.Crew
    llm=LLM(model="gpt-4o-mini"),     # required: a crewai.LLM or a BaseLLM
)
```

`crew_factory` takes the `llm` and returns a fresh `crewai.Crew` whose task
description templates `{user_input}` (the attacker-controlled input the target
fills at kickoff). `llm` is **required** — a CrewAI agent cannot run without one.
The target holds **no API key**: the model's auth lives on the `llm` you supply,
and only its model id / class name is ever emitted as an observable (never the
`llm` object), so no secret passes through this target. Config: `user_task`
(benign default input). Controllable: `user_input`.

## Queries

`last_response` (the crew's final output), `tool_calls` (JSON of `{name,
arguments}`), `called_tool_names` (comma-separated, in order), `error`. Tool calls
are captured via the crew's `step_callback` (which fires per agent action during
kickoff), so calls made before a mid-run error are salvaged; run errors are
recorded in `error` (never raised), so a claim can abstain.

## License

MIT. Uses the MIT-licensed `crewai` package as a dependency; no third-party code
is vendored.
