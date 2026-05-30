# inspect-agent-target

A general, benchmark-agnostic superred `Target` that runs an
[inspect-ai](https://inspect.ai-safety-institute.org.uk/) tool-calling agent over
whatever tools, prompts, and model it is handed. It is the reusable substrate for
any inspect-tool agentic SecurityClaim (the AgentHarm claim is the first consumer).

In the project's target taxonomy (chatbot < agent < assistant) this is an
**agent**: a scoped tool-caller.

## What it does

`InspectAgentTarget.run()`:
1. fires the `system_prompt` and `user_prompt` Controllables (an attacker /
   optimizer may override either via `ControllableInjection`);
2. resolves the configured tool names to inspect `Tool`s via the **tool resolver**
   passed at construction;
3. runs the tool-calling loop (`model.generate` + `execute_tools`, capped at
   `message_limit`) -- the body of inspect's `generate(tool_calls="loop")`;
4. emits one observable per chat message and per tool call;
5. stores the full `list[ChatMessage]` (typed `messages` property) for a bound
   Task to grade, plus string `query()` readers.

The benchmark-specific seam is the `tool_resolver` (name -> `Tool`) and the
per-run config (prompts, tool names). The target itself knows nothing about any
benchmark and depends only on `superred` + `inspect-ai`.

## Usage

```python
from inspect_agent_target import InspectAgentTarget

def my_resolver(name: str):
    # return an inspect Tool for `name` (e.g. getattr(my_tools_module, name)())
    ...

target = InspectAgentTarget(
    model="openai/gpt-4o-2024-08-06",
    tool_resolver=my_resolver,
    api_base="https://my-litellm-proxy/",   # optional
    api_key="sk-...",                        # optional
)
```

A Task configures it per run via `set_config`:

- `system_prompt`, `user_prompt` (text)
- `tool_names` (JSON `list[str]`)
- `tool_choice` (`"auto"`/`"any"`/`"none"`, default `"auto"`)
- `message_limit` (int as string)
- `model` (optional per-run override)

## Security domain

A small forest: `user` (user/jailbreak channel), `system_prompt`
(+ readable child), `model_identity`, `tools` (+ `tools_readable` and
`tools_addable` children), and `agent_trace` (+ messages / tool_calls children).
A SecurityClaim pins its Scores to whichever tag matches its threat model.

## Tool-catalogue attack surface

Besides the two prompt Controllables, the target fires four tool-catalogue
Controllables **once at run start** (after seeding the catalogue from the static
Task config), so an attacker-scoped optimizer from a tool-poisoning claim can
edit the tool registry before the agent runs:

- `tool_catalog_register` (scope `tools_addable`): add an attacker-defined tool
  with a canned return.
- `tool_catalog_replace` / `tool_catalog_unregister` / `tool_catalog_rewrite_doc`
  (scope `tools`, broad): shadow / remove / re-describe an existing tool.

The target **always** fires these; the Controller's scope filter decides whether
a given optimizer may inject (out-of-scope -> auto no-injection). The resulting
catalogue is exposed via the `tool_catalog_listing` observable (`tools_readable`).
Initial tools are still set statically by the Task; a passthrough optimizer leaves
the catalogue untouched (so faithful benchmark baselines are unaffected).

Note: AgentDojo fires its equivalent catalogue hook before *every* LLM turn; this
target deliberately fires once at the start to avoid per-turn event noise (the
optimizer gets a single edit, then the tool set is fixed for the run).

## Install / test

```bash
pip install -e targets/inspect_agent
pytest -m "not smoke" targets/inspect_agent/tests    # offline unit tests
# real-LLM smoke (agent loop + tool-poisoning + scope gating against a live model):
LITELLM_API_KEY=... LITELLM_API_BASE=... pytest -m smoke targets/inspect_agent/tests
```

See `ASSUMPTIONS.md` for the rollout-faithfulness notes.
