# Assumptions and divergences: inspect-agent-target

This target re-implements the orchestration of an inspect tool-calling agent so
that it runs under superred's Controller instead of `inspect_ai.eval()`. The
parts that determine a benchmark score (model, tools, message types) are inspect's
own; only the loop is ours.

## A. The rollout loop
`run_rollout` reproduces the body of inspect's `generate(tool_calls="loop")` using
inspect's own primitives: `Model.generate(messages, tools, tool_choice)` then
`execute_tools(messages, tools)` (the current API; supersedes the deprecated
`call_tools`). The resulting `list[ChatMessage]` is structurally identical to what
an `inspect_ai.eval` run produces. This loop is the single hand-replicated
control-flow point; a credentialed trace-parity test guards it.

## B. Message limit
`message_limit` caps the total number of messages (system + user + assistant +
tool). The loop breaks after appending tool results once the count reaches the
limit. inspect enforces its `message_limit` as a run limit; for behaviors that
finish before the cap (the common case) the two are equivalent. A consumer that
needs exact long-chain parity should pin the same value the benchmark uses
(AgentHarm: 20) and rely on the trace-parity test.

## C. Tool order
Tools are exposed in the order the caller supplies them (the resolved
`tool_names`). inspect's AgentHarm `setup_tools_from_metadata` shuffles tools with
`random.shuffle`; that shuffle is reproducible upstream (inspect seeds the RNG
from `GenerateConfig(seed=0)`), but order does not matter either way: benchmark
grading inspects which tools were called and in what call order, not their menu
order, so presentation order does not affect scores.

## D. Generation config
The model's `GenerateConfig` is built in `run()` from: `temperature` (default 0.0)
and `max_tokens` (default 4096), both construction parameters (AgentHarm uses
0.0 / 4096); plus `seed=0` and `max_retries=3`, hardcoded to AgentHarm's upstream
values as benchmark-agnostic generation defaults (not configurable, not
constructor params). `max_connections` is deliberately NOT set: cross-target
parallelism is owned by the `TargetFactory` (its `concurrency` + the controller's
semaphore), not by per-target generation config, so baking a connection cap into
the target would conflict with the factory's concurrency and risk the proxy's
rate limit. AgentHarm sets `max_connections=100` for its own scheduler; a claim
that needs strict parity can opt in at its target factory.

## E. Model client
The target uses its own inspect `Model` (via `get_model`, optionally pointed at a
LiteLLM proxy with `base_url`/`api_key`), not superred's `LLMClient`. This mirrors
the AgentDojo target, which also drives its own provider client. Consequently the
agent's own token spend is not counted in the Controller's `llm_usage` (which
tracks the optimizer's constrained client); it is out-of-band, like an OOB judge.
The model id is fixed at construction (the `model` arg): it is **not** a config
slot, so neither the Task nor the attacker can change the agent's model; a per-run
model change is not part of this surface. The read-only identity is still exposed
via the `model_identity` observable.

## E.1 Static observables vs the trajectory
Static configuration is exposed as **static observables** (`get_observables`,
handed to the optimizer at init): `model_identity`, `system_prompt`,
`message_limit`, and `tool_catalog_listing` (the configured, pre-edit catalogue).
Everything that **happens during the run** is on the **trajectory** as
`ObservableEvent`s (the only one-way target-to-trajectory mechanism, each wrapping
an `Observable`): the per-message / per-tool-call / per-tool-response agent trace.
Attacker actions (prompt and catalogue injections) are on the trajectory too, as
the controllable events. So the catalogue is intentionally NOT emitted on the
trajectory: its static snapshot is an observable, and its dynamic edits are the
catalogue controllable events.

## F. Tool choice modes
Only inspect's standard `tool_choice` values are supported (`auto`/`any`/`none`).
The AgentHarm "forced_first" agent variant is out of scope; AgentHarm's default
(and this port's baseline) is `auto`.

## G. Tool-catalogue Controllables (AgentDojo-style, fired once)
The target exposes four tool-catalogue Controllables (`tool_catalog_register`,
`_replace`, `_unregister`, `_rewrite_doc`) fired **once at run start** (after the
catalogue is seeded from the static `tool_names` config), plus a
`tool_catalog_listing` observable. Design points:

- **Once, not per turn.** AgentDojo fires its catalogue hook before *every* LLM
  turn (outside the loop for the first turn, inside `ToolsExecutionLoop` for the
  rest). We deliberately fire once at the start to avoid per-turn event noise: the
  optimizer gets a single chance to edit the registry, then the tool set is fixed
  for the run. The per-turn `tools_provider` seam in `run_rollout` still exists
  (a static provider is used here), so per-turn firing could be reinstated.
- They are fired **unconditionally** by the target; the Controller's
  `security_domain_filter` middleware gates injection by scope, so an out-of-scope
  optimizer simply receives no-injection. The target's job is to expose the
  surface; the experiment designer's scope decides exposure to the optimizer.
- Scopes mirror AgentDojo: register is `tool_catalogue_addable` (weakest write),
  the other three and the static tool set are `tool_catalogue` (broad); the
  listing is `tool_catalogue_readable`. Broad implies the children. The
  `tool_catalogue` tag lives under the `system` umbrella root, alongside
  `system_prompt`, `model_identity`, and `agent_trace`.
- Initial tools remain a static Task config (`tool_names`); the catalogue is
  seeded from them each run and edited only by accepted injections.
- **Faithfulness-safe**: a passthrough optimizer (the AgentHarm baseline) injects
  nothing, so the catalogue stays exactly the Task-configured tools and the run is
  identical to having no catalogue surface. The surface exists for *other* claims
  (tool poisoning / malicious-MCP), not for AgentHarm itself.
- Registered/replaced tools are built with `ToolDef` from an attacker-supplied
  name/description/JSON-Schema params and a canned-return body; malformed payloads
  are logged and ignored (never abort the run).

## H. Per-tool output Controllables (indirect prompt injection, scoped by trust boundary)
For each configured tool the target exposes one `tool:<name>` Controllable, fired
as a `ControllablePostCallEvent` after **that** tool returns.  Design points:

- **One controllable per tool, post-call only.** The event carries the tool's
  legitimate return as `answer` and the tool name as `request`. A
  `ControllableInjection` replaces the content the agent sees; a no-injection (or
  out-of-scope filter) leaves it verbatim. Pre-call request tampering is
  deliberately *not* a separate controllable: the tools here are side-effect-free
  (canned returns / read-only benchmark tools), so executing the call and then
  rewriting its whole return is indistinguishable from first rewriting the request
  then rewriting the return. One post-call surface per tool subsumes both.
- **Scoped by trust boundary, not one tag per tool.** Each `tool:<name>` is scoped
  to `tool_scopes[name]` -- a claim-supplied leaf of a trust-boundary sub-forest
  parented under the `tools` root (e.g. `web`, `social`, `financial`). The
  *principle* is "scopes where content could realistically be compromised"; tools
  hitting the same external system share a scope (so it is often, but not always,
  one scope per tool). The general target knows none of these boundaries; it just
  accepts the map and assembles the domain via `build_domain` (the ancestor
  closure of the map's values, plus the fixed trees). Unmapped tools fall back to
  the bare `tools` root.
- `tools` (the per-tool *returned content*) is distinct from `tool_catalogue`
  (the *registry*: which tools exist and their docs). The `tool_catalog_replace`
  catalogue edit overlaps with a `tool:<name>` injection for shadowing a return,
  but register/unregister/rewrite-doc do not; the two surfaces are kept separate.
- Fired **unconditionally**; the Controller's `security_domain_filter` gates
  injection by scope, exactly like the catalogue controllables.
- The agent-visible value (after any injection) is mirrored to the
  `agent_trace_tool_response_NNNN` observable (scope `agent_trace_tool_responses`),
  so a read-scoped optimizer sees exactly what the agent saw.
- **Faithfulness-safe**: a passthrough optimizer injects nothing, so every tool
  return reaches the agent unchanged and AgentHarm baselines are unaffected.
