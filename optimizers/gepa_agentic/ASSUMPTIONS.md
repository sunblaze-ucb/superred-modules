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

Reflected candidates are not added to the population just because they were
generated. A pending reflected candidate receives one rollout and is retained
only if the run succeeds or its scored rollout strictly improves over the
parent's effective score. This mirrors upstream GEPA's strict-improvement
acceptance at the granularity SuperRed exposes here. Runs without objective
feedback can still be observed and reflected on, but they do not grow the
population.

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

`response_observable_names` is an override for exact response channels. When it
is set, heuristic response-name detection is disabled; when it is unset, the
optimizer uses default response names plus broad agent-trace heuristics.

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

Full upstream GEPA Pareto-frontier maintenance is also deferred. The current
SuperRed optimizer operates on one task rollout at a time rather than GEPA's
multi-example minibatch/full-validation loop, so this PR implements the
faithfulness-critical acceptance gate without inventing a frontier abstraction
that the controller does not yet expose.

## DTAP fitness: free-text gating, exact user-prompt match, no command surface

Three plumbing corrections so the reflective loop reaches a real surface on the DTAP
targets without changing the algorithm:
- `_accepts_free_text` gate (value_type in text/str/string/html/markdown) on both the
  PreCall and PostCall emission sites. A schema-typed content surface (DTAP
  `env_inject:<server>`, json) discards a raw string, so injecting it is a no-op
  recorded as a scored attack; it is now declined WITHOUT spending the per-run content
  budget, so a free-text surface (e.g. `env_tool:<server>`) later in the run is reached.
- `_is_user_prompt` is an exact reserved-name match. The prior `"user" in name`
  substring test misclassified PostCall content surfaces whose authorization node key
  is `user` (e.g. DTAP `env_tool:atlassian.user`) as a prompt channel.
- the `"answer carries"` content hint was removed. It classified `code_execution`
  (a shell surface) as content, so the reflective prose was run as a shell command
  (a side-effecting nondeterminism hazard) and consumed the injection budget that
  `env_tool` needs. `code_execution` has no faithful natural-language payload, so it is
  no longer treated as a content surface.
## Surface classifier: empty categories and out-of-money budget

The shared LLM surface classifier (`surface_llm.classify_controllables`,
byte-identical across the agentic optimizers) sorts each granted surface into a
role category by reading its description. Two behaviours deviate from a naive
reading and are load-bearing:

- Categories are roles to match, not a partition to fill. When a scope grants no
  surface of a given role -- e.g. the experiment drops the user-prompt surface
  from a threat model -- the prompt tells the model a category may match zero
  surfaces and forbids relabelling content surfaces to populate it. Without this,
  gpt-4o-2024-05-13 put every DTAP `env_tool:<server>` surface into `user-prompt`
  under category-completion pressure. Measured on the DTAP indirect claim at scope
  s3 (11 text domains, one task each), the false label made the primary consumer
  of this signal (the AgentVigil chain) vacuous -- its reachable surface set
  collapsed to one and it finished after a single non-delivering run -- in 5 of 11
  domains; the improved prompt gives 0 of 11 at s3, s4 and s6, while a control arm
  that keeps the user-prompt surface stays at 0 throughout. The prompt also
  classifies by role, not goal-relevance, so a live indirect-injection surface is
  not dropped to `irrelevant` merely because it looks off-topic for the task.

- Out-of-money is distinguished from "no LLM". A genuinely exhausted attacker (a
  positive per-task cap consumed, so the raised `BudgetExhaustedError` carries
  `usage.cost > 0`) is re-raised, so the controller records the task as
  budget-exhausted instead of the bare handler swallowing it into an empty
  classification that a dead proxy or a target with nothing to attack would also
  produce. The deliberately budget-less noop client the controller hands a
  non-LLM optimizer raises the same error with nothing spent (`usage.cost == 0`);
  that is "no LLM configured", not "out of money", and still degrades to the
  caller's name-based backstop. `fill_value` gates on the same distinction.
## PostCall timing is read from the target, not presumed

GEPA arms a surface as a PostCall content surface -- one whose returned value the
agent reads back as data -- only if the target can actually fire a PostCall event
on it. Content is a ROLE (does the agent read the value back?); PostCall firing is
a TIMING fact about the target. The shared LLM classifier answers only the role
question, so it labels a PreCall-only surface such as DTAP's `filesystem`
(attacker files the agent later reads) as content: true as a role, wrong as
timing, because that surface is consumed before the run and never returns a value
for the agent to read back. Arming it would send the search to inject content into
a surface that never fires the event it waits for; measured on 20 real DTAP
surfaces, the prior code (which hard-coded `event_kind="post"` during discovery)
armed most PreCall-only surfaces wrongly. The `Controllable` type carries no
timing field, so timing is read from the target's own declaration in the
description: a surface the target marks `PreCall` is excluded from PostCall content
(`_can_fire_postcall`). A description that declares neither token keeps the prior
assumption (eligible), so targets that do not annotate timing -- AgentDojo, ASB,
inspect_agent -- are unaffected, and the runtime path is unchanged because a
surface that actually fires a PostCall event genuinely can. Sourcing timing from a
description string rather than a typed field is a deliberate minimal choice: a
structured `Controllable` timing field would be the robust fix but is a
framework-wide change for the target authors to weigh, not something to slip into
an attacker.
