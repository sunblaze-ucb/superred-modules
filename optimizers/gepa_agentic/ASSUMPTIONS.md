# GEPA-Agentic Optimizer Assumptions

GEPA-Agentic is a separate optimizer from the existing chatbot GEPA
implementation in `optimizers/gepa`. The chatbot optimizer is intentionally left
unchanged. This package keeps GEPA's reflective candidate loop but uses an
agent-target delivery layer.

## Surface Model

GEPA's optimized object remains one textual candidate instruction. The agentic
extension is the delivery policy: given the controller-filtered scope, the same
candidate can be delivered through compatible target surfaces that represent
external content, tool returns, retrieved context, memory records, or web
environment data.

Surface naming is not treated as a framework contract. Known names are weak
signals:

- AgentDojo-style `read__...` and `tool_call:...` PostCall events;
- inspect-agent-style `tool:<name>` PostCall events;
- skill-layer aliases such as `skill:` and `skill_call:`;
- RAG/memory/web/document/content hints in `name`, `description`, `value_type`,
  and runtime PostCall content.

`system_prompt` and user-prompt-style channels remain prompt fallbacks when no
agentic content surface is available. `tool_catalog_rewrite_doc` is supported as
a minimal text-rewrite capability extension when a visible tool catalog
identifies a content/read/search-like tool. Full catalog register/replace are
still deferred because those operations need stricter JSON contracts and
attack-specific tool bodies.

## Runtime Selection

Surface choice is runtime-based. `initialize()` records the in-scope
controllables and classifies likely content surfaces. When multiple content
surfaces are in scope, the optimizer builds a deterministic planned order from
per-surface stats, then injects into compatible PostCall surfaces that actually
fire up to a small per-run cap.

This keeps GEPA faithful to one candidate mutation per rollout while using the
available agentic attack surface more fully than a single arbitrary PostCall. If
planned surfaces do not fire, the miss is recorded. If a dynamic compatible
surface appears that was not known during `initialize()`, including a
content-like opaque return discovered from HTML/DOM-like content, the optimizer
can inject in runtime-discovery mode and records that decision.

Prompt fallback is explicit: when no agentic content surface can be classified,
the optimizer uses `system_prompt` / user prompt channels and records the
fallback reason in the reflective rollout.

## Reflection Context

Each rollout record includes:

- selected surface metadata, including all injected surfaces when more than one
  surface was used;
- observed PreCall/PostCall surfaces for the run;
- legitimate tool return when a PostCall was injected;
- bounded agent trace, tool-call, tool-response, action, and response
  observables when visible;
- static observables exposed by the controller scope;
- feedback from `RunEndEvent.evaluation` when available.

This lets the reflection LM reason about whether the candidate worked as
indirect tool content, memory/RAG/environment content, catalog setup, or a
prompt-channel fallback.

If no payload reached the target and there is no feedback or trajectory signal,
the run does not spend a GEPA attempt. This avoids penalizing a candidate for a
surface-delivery miss.

## Deliberate Limits

GEPA-Agentic is not a full replacement for specialized attacks like MINJA,
PoisonedRAG, EIA, AgentVigil, or CHORD. It does not hard-code their full
multi-stage algorithms. It does, however, use memory/RAG/environment/tool-return
surfaces when the target exposes them in scope. Durable state across runs is a
target/controller lifecycle property; if memory persists within a task and is in
scope, GEPA-Agentic may use it like any other visible/writable surface.

Full tool-catalog register/replace remains a follow-up. `tool_catalog_rewrite_doc`
is included because it is a bounded text rewrite and fits GEPA's instruction
evolution model; registering or replacing tools requires target-specific JSON
payload bodies and separate tests.
