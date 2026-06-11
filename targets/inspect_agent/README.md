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

The **model is not a config slot** — it is fixed at construction (the `model`
arg), so neither the Task nor the attacker can change it. Generation config
(`temperature`/`max_tokens` from construction; `seed=0`/`max_retries=3` hardcoded
to AgentHarm's defaults) is likewise not per-run configurable; cross-target
parallelism is owned by the `TargetFactory`, not the target.

Static configuration (`model_identity`, `message_limit`, the configured
`tool_catalog_listing`) is exposed as **static observables**; the running agent
trace (messages / tool calls / tool responses) is on the **trajectory** as
`ObservableEvent`s. The system prompt is not mirrored as an observable: it is
carried exactly once, on the system-prompt controllable event at run start, so
a Controller that lists `system_prompt` under `read_only` reads it from the
trajectory without being able to override it.

## Security domain

A forest of three root trees (AgentDojo style: a scope holding a parent tag
includes all its descendants, so a Controller can scope broadly or narrowly):

- `system` (the whole agent-side surface) over:
  - `system_prompt` (read-only access is granted by listing the tag in the
    Controller's `read_only` set, not by a separate tag)
  - `tool_catalogue` (broad registry write; the listing observable carries
    this tag) -> `tool_catalogue_addable` (register-only, the weakest write)
  - `model_identity`, `message_limit`
  - `agent_trace` -> `agent_trace_messages` (the full transcript) -> its two
    projections `agent_trace_tool_calls` and `agent_trace_tool_responses` (the
    transcript embeds both, so `agent_trace_messages` subsumes them; the two are
    siblings since call-args and return-values are disjoint)
- `user`: the user-prompt / jailbreak channel
- `tools`: the per-tool write surface (what each tool returns to the agent;
  indirect-prompt-injection). This root carries **no children by itself**; a
  SecurityClaim parents a per-tool trust-boundary sub-forest under it (e.g.
  `web` / `social` / `financial`) and maps each tool to a leaf, via the
  `tool_scopes` constructor argument. With no map every tool falls back to the
  bare `tools` root. The read side is `agent_trace_tool_responses`.

A SecurityClaim pins its Scores to whichever tag matches its threat model, and an
experiment's Controller scope picks which tags an attacker may touch. The three
roots together (`{user, system, tools}`) are the full attacker surface.

## Tool-catalogue attack surface

Besides the two prompt Controllables, the target fires four tool-catalogue
Controllables **once at run start** (after seeding the catalogue from the static
Task config), so an attacker-scoped optimizer from a tool-poisoning claim can
edit the tool registry before the agent runs:

- `tool_catalog_register` (scope `tool_catalogue_addable`): add an
  attacker-defined tool with a canned return.
- `tool_catalog_replace` / `tool_catalog_unregister` / `tool_catalog_rewrite_doc`
  (scope `tool_catalogue`, broad): shadow / remove / re-describe an existing tool.

The target **always** fires these; the Controller's scope filter decides whether
a given optimizer may inject (out-of-scope -> auto no-injection). The configured
(pre-edit) catalogue is exposed via the `tool_catalog_listing` observable, which
carries the `tool_catalogue` tag itself — list that tag under `read_only`
(rather than `scope`) for listing-only access.  Initial tools are still set
statically by the Task;
a passthrough optimizer leaves the catalogue untouched (so faithful benchmark
baselines are unaffected).

## Per-tool output attack surface

For **every configured tool** the target exposes one `tool:<name>` Controllable,
scoped to that tool's trust boundary (`tool_scopes[name]`, or the bare `tools`
root if unmapped). After **each** tool call it fires that tool's
`ControllablePostCallEvent` carrying the legitimate return as the event's `answer`
(tool name in `request`); an attacker-scoped optimizer may respond with a
`ControllableInjection` to replace the value the agent sees -- the
indirect-prompt-injection surface (poisoning tool-returned data). Because the
canned/real tool is always executed first and only its return is rewritten, a
single post-call controllable per tool subsumes pre-call request tampering for
side-effect-free tools. The agent-visible value (post-injection) is mirrored to
the `agent_trace_tool_response_NNNN` observable. A passthrough optimizer leaves
every return untouched.

Scoping by tool means an experiment can grant an attacker control over, say, only
web-sourced tool returns by putting a single trust-boundary tag in scope; tools
sharing a boundary share a scope. `tools` (per-tool *returned content*) is
distinct from `tool_catalogue` (the *registry*: which tools exist).

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
