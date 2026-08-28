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

A run that delivered no payload is never allowed to penalize a candidate, and
that holds whether or not the threat model returned feedback. If there is also no
feedback and no trajectory signal, the run does not spend a GEPA attempt at all.
If there is feedback, the run does spend an attempt and is still recorded as a
rollout, so the reflection LM can see the miss, but its score is `None` rather
than the reported value: an undelivered candidate did not earn that number, the
target did. `None` propagates correctly, because `effective_score` averages only
non-`None` scores and `_should_accept_fresh_candidate` already requires a
non-`None` score on every path except a reported success, which ends the task
anyway.

`response_observable_names` is an override for exact response channels. When it
is set, heuristic response-name detection is disabled; when it is unset, the
optimizer uses default response names plus broad agent-trace heuristics.

## When Reflection Does Not Produce a Mutation

Reflection is the only thing that makes one run differ from the last. When it
produces nothing, the next run re-sends the *identical* payload to the victim:
full victim cost, zero search progress, and a record that looks like a
legitimate multi-run search that failed. Measured on the DTAP indirect sweep,
that was **2,974 of 18,013 reflections (16.5%)**, and the rate is a property of
the attacker model rather than the task (no-fence misses per cell ran from 2 on
deepseek-v3-2 to 1,719 on opus-4-8), so it silently biased the attacker-model
axis the experiment exists to compare.

The three ways reflection can produce nothing are now kept apart, matching the
sibling `gepa` package, and none of them is silent:

1. **Cost cap spent** (`BudgetExhaustedError`). Re-raised untouched and never
   retried, since retrying a spent cap is a cap escape. The controller records
   `stop_reason="budget_exhausted"`. Previously a bare `except Exception`
   swallowed this, so an attacker that was out of money silently carried on
   spending victim episodes.
2. **The call failed** (provider or transport error). Retried
   `reflection_retries` times (default 2, so 3 attempts) with full-jitter
   exponential backoff, jittered because a whole matrix cell retries against one
   provider at the same instant. Retrying stops early once
   `reflection_retry_deadline` seconds (default 120) have passed, because a
   single provider timeout can itself be minutes and retrying into the
   controller's `task_time_cap_s` would discard the task outright. If every
   attempt fails, `ReflectionUnavailable` is raised and the controller records
   `stop_reason="error"`. A dead reflection LM means GEPA never searched, and
   that must not be recorded as a target that held.
3. **The LM answered but proposed nothing parseable** (no fenced block, in
   practice an attacker model declining to improve an attack). Legitimate
   attacker output, not an infrastructure failure, so the parent is re-rolled,
   which is worth something: it refreshes the parent's rollout buffer so the
   next reflection sees different side-info. After
   `max_consecutive_no_mutation` misses in a row (default 3) the optimizer
   stops with `done=True` and logs why, rather than spending its remaining
   attempts re-sending one payload. Set the knob to 0 to restore unbounded
   re-rolling.

Classes 1 and 2 are raised at the *next* `RunStartEvent` rather than from the
`RunEndEvent` that detected them. The controller sends `RunStartEvent` before it
calls the target, so the task ends without paying for another victim episode
while the just-completed run keeps its real evaluation instead of being
overwritten by a synthetic zero-score error result.

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
  not dropped to `irrelevant` merely because it looks off-topic for the task. One
  wording constraint is load-bearing: the prompt describes each role in prose and
  must never spell one out as a label-shaped phrase. An earlier revision said a
  qualifying value "is a content/environment surface"; the model answered with
  that literal string, every entry failed the `cat in allowed` filter, and
  `classify_controllables` returned `{}`. That total discard is invisible to a
  vacuity check, because an empty result is never vacuous.

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

## The surface ladder: deferral to a content surface is bounded, and monotone

A planned content surface is a *deferred* delivery channel. `env_tool:<server>`
(and the AgentDojo/inspect-agent equivalents) is a PostCall that fires only if the
agent happens to call that tool. Until now, planning any such surface made
`_handle_pre_call` decline `user_prompt` and `system_prompt` for the whole task,
on the standing assumption that the deferred surface would eventually fire. Since
`_choose_content_surfaces` returns every classified content surface, the plan is
non-empty whenever a single content surface exists at all, so the assumption was
never re-examined: a surface planned and missed twenty times running still made
the prompt channels ineligible on run twenty.

That is a silent zero, not a slow attack. Measured over 40 persisted DTAP runs,
`user_prompt` was offered 40 times and written 0 times, and 22 of the 40 runs
injected nothing anywhere. At scope s5 (`{SYSTEM, HOST}`) the failure is total: all
seven surfaces the scope grants besides the system prompt are PreCall-only, so on
the pre-`_can_fire_postcall` classifier every one of them was armed as a content
surface, the plan was never empty, and the optimizer declined the one surface it
could actually write for all 20 runs across all 11 domains.

The optimizer cannot detect this at decision time: the DTAP scaffold emits both
prompt PreCalls before any PostCall, so when `user_prompt` is decided it is
structurally unknown whether the content surface will fire this run. The only
available evidence is the previous runs, and it already exists. `_SurfaceStats`
already counts a `misses` entry for every planned surface that was not injected.
`_ladder_depth` now reads that same computation: every run whose planned content
surfaces all missed increments it by one. Depth 0 is the old behaviour (defer to
the content surface, decline the prompts) and is what run 1 still does unchanged;
any greater depth also writes the prompt, alongside the content surface, which is
still attempted on the same run.

The depth is **monotone**: it grows on a miss and is never reset by a subsequent
delivery. This is a deliberate choice over the resetting alternative, which
reopened the prompt channel after a miss and closed it again on the next delivery.
A stochastic content surface that fires every other run would, under the resetting
rule, re-blind the optimizer to the only channel that has provably delivered,
every other run, for the whole budget. With a 20-run budget, repeated misses
against a channel that may be dead are the expensive failure, and a redundant
prompt injection is the cheap one.

Attribution cost, accepted: from the first miss onward a prompt injection and a
content injection can coexist in the same run, both carrying the same evolved
text. The rollout record then lists both surfaces under `all_injected_surfaces`
and which one earned the score is ambiguous. `_selected_surface` still reports the
first. This is the port-local property being given up ("at most one clean
attributable surface per run"); the alternative is a guaranteed zero.

Fidelity cost against the GEPA paper (Agrawal et al., arXiv:2507.19457): none.
GEPA defines reflective mutation over scored rollouts and says nothing about
delivery surfaces at all. The surface-selection layer is this port's own addition
(see "Surface Model" above), so the ladder is a change to our own plumbing, not a
deviation from the published method.

## DTAP fitness: an empty user turn is always filled

The ladder makes the prompt channels eligible again *after* a content surface has
missed once. On a DTAP `threat_model="direct"` task that one wasted run is one too
many, for a reason specific to the user prompt.

The DTAP user prompt is not one payload channel among several. It is the only
thing that gives the agent a turn. `dtap_scaffold` reaches the `user_prompt`
controllable with `request=""` on a direct task (there are no benign instructions
to put there), and whatever comes back becomes the agent container's message.
`dtap_openclaw` then launches with `--message ""`, which its CLI rejects, so the
container exits 1 before the agent runs. The task is recorded as defended when
nothing was ever tried.

So the rule for this one case is not "prefer this surface", it is "this surface
cannot be left empty". `_handle_pre_call` computes `empty_user_turn` (a reserved
user-prompt name whose `request` is blank) and exempts it from two declines: the
ladder's rung-0 deferral, and the first-wins `_primary_pre_controllable` lock.
Nothing else changes. A user prompt that carries a benign request is untouched, so
an indirect task keeps the legitimate instruction that its injected content is
supposed to subvert, and every other target (AgentDojo, ASB, chatbot,
inspect-agent) passes a non-empty user request and is unaffected.

Fidelity cost: none, for the same reason as the ladder. The payload is unchanged
and the reflective loop is unchanged; only the delivery of an otherwise empty
field changes. The port-local cost is the same attribution ambiguity the ladder
already accepts.

What this does not do: it decides nothing when the target hands the optimizer a
non-empty user request, so a scope that grants both prompts still settles on
whichever surface the target offers first (the system prompt on DTAP).
