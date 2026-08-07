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

## DTAP fitness: content-surface deferral is no longer unconditional

A planned content surface is a deferred delivery channel: `env_tool:<server>` (and
AgentDojo/inspect-agent equivalents) is a PostCall that only fires if the agent calls
that tool. Until now, planning such a surface made `_handle_pre_call` decline
`user_prompt` and `system_prompt` for the whole task, on the assumption that the
deferred surface would eventually fire.

On a DTAP `threat_model="direct"` task that assumption is false and unrecoverable. The
task carries no user instructions, so `dtap_scaffold`'s agent loop launches the agent
with the empty-string prompt (`for turn in (self._user_instructions or ("",))`). The
agent has nothing to act on, calls no tool, and the PostCall never fires. Every run
therefore delivered nothing at all and scored 0.0, for the entire attempt budget.

The optimizer cannot detect this at decision time: DTAP emits both prompt PreCalls
before any PostCall, so when `user_prompt` is decided it is structurally unknown
whether the content surface will fire this run. The only available evidence is the
previous run. `_update_surface_stats` already computes which planned content surfaces
actually delivered; that same computation now also sets `_content_delivered_recently`,
and the decline branch requires it. The flag starts `True`, so run 1 is unchanged and
still prefers the content surface. The first run that plans a content surface and
delivers nothing flips it, and from run 2 on the prompt channels are eligible again,
in parallel with the content surface still being attempted on the same run.

Fidelity cost: none against the GEPA paper (Agrawal et al., arXiv:2507.19457), which
defines reflective mutation over scored rollouts and says nothing about multi-surface
agent delivery. The surface-selection policy is this port's own invention (see
"Surface Model" above), so this is a delivery-plumbing change, not a method change. The
one property given up is a port-local one: "at most one clean attributable surface per
run". After a missed run, a stochastic content surface that does fire can now coincide
with a prompt injection in the same run, so the rollout record lists both surfaces and
attribution between them is ambiguous. Cost on a genuinely dead task is one wasted run
instead of `max_attempts` (default 20).

## DTAP fitness: an empty user turn is always filled

The previous section made the prompt channels eligible again after a content surface
missed. It did not make them eligible on the runs where the content surface does
deliver, and on a DTAP `threat_model="direct"` task that is not enough.

The reason is that the DTAP user prompt is not one payload channel among several. It
is the only thing that gives the agent a turn. `dtap_scaffold` reaches the
`user_prompt` controllable with `request=""` on a direct task (there are no benign
instructions to put there), and whatever comes back becomes the agent container's
message. `dtap_openclaw` then launches with `--message ""`, which its CLI rejects, so
the container exits 1 before the agent runs. The task is recorded as defended when
nothing was ever tried.

So the rule is not "prefer this surface", it is "this surface cannot be left empty".
`_handle_pre_call` computes `empty_user_turn` (a reserved user-prompt name whose
`request` is blank) and exempts it from two declines: the content-surface deferral,
and the first-wins `_primary_pre_controllable` lock. Nothing else changes. A user
prompt that carries a benign request is untouched, so an indirect task keeps the
legitimate instruction that its injected content is supposed to subvert, and every
other target (AgentDojo, ASB, chatbot, inspect-agent) passes a non-empty user request
and is unaffected.

Fidelity cost: none against the GEPA paper (Agrawal et al., arXiv:2507.19457). The
payload is unchanged, the reflective loop is unchanged, and only the delivery of an
otherwise empty field changes. The port-local cost is the same one the previous
section already accepted: on a run where another surface already holds the lock, the
rollout record lists two injected surfaces (both carrying the same evolved text), so
attribution between them is ambiguous. `_selected_surface` still reports the first.

What this does not do: it does not decide anything when the target hands the optimizer
a non-empty user request, so a scope that grants both prompts still settles on
whichever surface the target offers first (the system prompt on DTAP).
