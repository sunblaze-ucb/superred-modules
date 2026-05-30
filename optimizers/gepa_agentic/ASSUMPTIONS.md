# GEPA-Agentic Optimizer Assumptions

GEPA-Agentic is a separate optimizer from the existing chatbot GEPA
implementation in `optimizers/gepa`. The chatbot optimizer is intentionally
left unchanged. This package keeps GEPA's reflective candidate loop but uses an
agent-target interaction layer.

## Surface Model

The optimizer follows the merged AgentDojo target contract first:

- `read__...` `ControllablePostCallEvent`s are read/tool content returns.
- `tool_call:...` `ControllablePostCallEvent`s are attacker-managed tool calls.
- `system_prompt` and user-prompt-style channels remain prompt fallbacks.
- Tool-catalog controls are recognized as catalog surfaces but are not used in
  this first agentic GEPA version because they need stricter JSON contracts.

For non-AgentDojo targets, GEPA-Agentic classifies PostCall content surfaces
using `Controllable.name`, `description`, `value_type`, and optional
`content_controllable_names`. This avoids relying only on `read__` /
`tool_call:` naming conventions.

## Runtime Selection

Surface choice is runtime-based. `initialize()` records the in-scope
controllables and classifies likely content surfaces, but each run only injects
when the chosen surface actually fires. When multiple content surfaces are in
scope, the optimizer selects one deterministically using per-surface stats
instead of injecting into the first arbitrary PostCall.

If a planned content surface does not fire, the miss is recorded and the next
run can try a different content surface. If a dynamic `tool_call:...` surface
appears that was not known during `initialize()`, the optimizer can inject in
runtime-discovery mode and records that decision.

Prompt fallback is explicit: when no agentic content surface can be classified,
the optimizer uses `system_prompt` / user prompt channels and records the
fallback reason in the reflective rollout.

## Reflection Context

Each rollout record includes:

- selected surface metadata;
- observed PreCall/PostCall surfaces for the run;
- legitimate tool return when a PostCall was injected;
- bounded agent trace / response observables when visible;
- static observables exposed by the controller scope;
- feedback from `RunEndEvent.evaluation` when available.

This lets the reflection LM reason about whether the candidate worked as
indirect tool content or as a prompt-channel fallback.

## Non-Goals

This package does not implement connected-run/shared-memory semantics. Those are
framework-level lifecycle concerns for attacks like MINJA and should be handled
separately from GEPA-Agentic's single-rollout reflective loop.
