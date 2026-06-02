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
an un-seeded `random.shuffle`, so presentation order is not reproducible upstream
either; benchmark grading inspects which tools were called and in what call order,
not their menu order, so this does not affect scores.

## D. Generation config
`temperature` (default 0.0) and `max_tokens` (default 4096) are construction
parameters bound onto the model via `GenerateConfig`. A faithful benchmark port
sets these to the benchmark's values (AgentHarm: 0.0 / 4096).

## E. Model client
The target uses its own inspect `Model` (via `get_model`, optionally pointed at a
LiteLLM proxy with `base_url`/`api_key`), not superred's `LLMClient`. This mirrors
the AgentDojo target, which also drives its own provider client. Consequently the
agent's own token spend is not counted in the Controller's `llm_usage` (which
tracks the optimizer's constrained client); it is out-of-band, like an OOB judge.

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

## H. Tool-output Controllable (indirect prompt injection, fired per result)
The target exposes one `tool_output` Controllable (scope `tool_output`, its own
root tree) fired as a `ControllablePostCallEvent` after **each** tool result.
Design points:

- **Post-call only, one per result.** The event carries the tool's legitimate
  return as `answer` and the tool name as `request`. A `ControllableInjection`
  replaces the content the agent sees; a no-injection (or out-of-scope filter)
  leaves it verbatim. Pre-call request tampering is deliberately *not* a separate
  controllable: the tools here are side-effect-free (canned returns / read-only
  benchmark tools), so executing the call and then rewriting its whole return is
  indistinguishable from first rewriting the request then rewriting the return.
  One post-call surface subsumes both, with less event noise.
- Fired **unconditionally**; the Controller's `security_domain_filter` gates
  injection by scope, exactly like the catalogue controllables.
- The agent-visible value (after any injection) is mirrored to the
  `agent_trace_tool_response_NNNN` observable (scope `agent_trace_tool_responses`),
  so a read-scoped optimizer sees exactly what the agent saw.
- **Faithfulness-safe**: a passthrough optimizer injects nothing, so every tool
  return reaches the agent unchanged and AgentHarm baselines are unaffected.
