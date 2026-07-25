# GEPA Optimizer Assumptions

References:

* Agrawal et al., "GEPA: Reflective Prompt Evolution Can Outperform
  Reinforcement Learning," arXiv:2507.19457, ICLR 2026.
* Official reference implementation: ``gepa-ai/gepa``
  (specifically ``src/gepa/strategies/instruction_proposal.py``,
  ``src/gepa/proposer/reflective_mutation/reflective_mutation.py``,
  ``src/gepa/strategies/candidate_selector.py``,
  ``src/gepa/api.py``).

## Algorithmic Faithfulness

This module implements the **reflective mutation** core of GEPA's
Algorithm 1 — the part of the paper that the authors highlight in
Section 1: "in most cases, even a single reflective prompt update can
give large improvements." Mapped to superred:

| GEPA term | superred equivalent |
| --- | --- |
| System Φ being optimized | the user-message prompt sent to the target |
| Component / module | one controllable's injection text |
| Rollout | one superred run |
| Metric μ score | ``RunEndEvent.evaluation.primary_score.value`` |
| Feedback μ_f text | ``RunEndEvent.evaluation.rationale`` plus the in-scope target response observable |
| Trace | trajectory items: response observable + (optional) post-call answer |
| Budget B | ``max_attempts`` (number of superred runs) |
| Reflection LM | the controller-provided ``LLMClient`` (separate budget from target rollouts) |

One superred run is one rollout of one candidate. Across runs, the
optimizer grows a candidate pool by reflective mutation: at the end of
each run we record ``(response, score, rationale)`` on the candidate
that was just rolled out, then call the reflection LM on the
best-scoring candidate to propose a new instruction for the next run
to evaluate.

## Verbatim Meta-Prompt

``prompts.py`` reproduces the meta-prompt verbatim from the upstream
``InstructionProposalSignature.default_prompt_template`` (also
Appendix B of the paper), including the two required placeholders
``<curr_param>`` and ``<side_info>``.

``_extract_fenced_block`` in ``reflector.py`` ports upstream's
``output_extractor`` first-open-to-last-close span and incomplete-block
handling exactly (including stripping the optional language tag on
the first line and trimming a leading or trailing fence when only
one is present), with two deliberate divergences:

1. When the input contains no fences at all, we return the empty
   string so reflection can no-op the mutation rather than the
   upstream behaviour of returning the raw stripped text. The
   divergence is the right call for adversarial use — a reflection LM
   that ignored the fence contract should not silently ship its
   rambling as the next candidate.
2. In the "incomplete block" branch (only one fence present), upstream
   re-matches the opening-fence-and-language-tag regex against the
   *original, unstripped* text, so a response with leading whitespace
   before the opening fence (e.g. ``"  ```python\nhello"``) fails that
   match and falls through to returning the whole stripped block
   *including* the fence and language tag. We instead re-match against
   the *left-stripped* text, so leading whitespace before a lone
   opening fence is tolerated and the language tag is still stripped
   correctly. This only changes behaviour for a corner case upstream
   itself likely didn't intend (LM output essentially never leads with
   whitespace before a fence); we keep the more robust extraction
   rather than reproducing the upstream quirk.

This preserves any internal triple-backticks the reflection LM may
emit when its proposed instruction itself contains nested fenced
examples.

## Adversarial Information-Access Settings

The optimizer naturally operates in all four settings — there is no
setting knob. The framework's scope filter and ``include_feedback``
flag select which information surfaces are visible:

| Setting | Reflective dataset has | Selection (mean across buffer) | Behaviour |
| --- | --- | --- | --- |
| 1 user query only | ``goal`` only | latest-wins (all scores 0) | LLM-driven prompt mutation chain; no early-stop |
| 2 + feedback | ``goal``, score, rationale | best-scoring | score-driven evolution + early-stop on success |
| 3 + responses | ``goal``, response | latest-wins (all scores 0) | response-aware mutation; no early-stop |
| 4 full | ``goal``, response, score, rationale | best-scoring | full GEPA loop + early-stop |

The reflective dataset only includes fields that are non-``None`` /
non-empty (see ``RolloutRecord.to_sample`` and
``format_reflective_dataset``). Settings that strip a surface produce
a smaller side-info block rather than a noisier one.

Independently, **all** in-scope static observables are captured at
``initialize`` and surfaced on every rollout sample as a single
``target_observables`` dict (e.g. ``{"system_prompt": "...",
"model": "gpt-4"}``). The reflection LM sees whatever capability
the threat model granted — system prompt content, model identity,
anything else the controller routed in — rather than only one
hardcoded surface. When no static observables are in scope, the
field is omitted entirely. Non-string observable contents and empty
/ whitespace strings are dropped (matches the
``format_reflective_dataset`` field-skip rule).

Capability symmetry on the **write** side: when the controller's
scope grants ``system_prompt`` as a writable controllable and the
caller hasn't pinned ``target_controllable_name`` explicitly, the
optimizer auto-claims it as the attack channel. The system prompt
is the higher-leverage attack surface (the assistant is conditioned
on it from the first token, before any user message arrives) and
auto-claiming it whenever it's available keeps the optimizer
*threat-model-faithful* — the same scope grant that previously gave
the optimizer "I can read the system prompt" capability now also
gives it "I can write the system prompt" if the controller wants to
expose that. The explicit ``target_controllable_name`` constructor
knob still wins over auto-claim. When ``system_prompt`` is not
writable, behaviour is unchanged: attack ``user_message`` and skip
read-only ``system_prompt`` PreCalls.

``max_no_signal_runs`` (default ``0``, disabled) bounds the
user-query-only setting's cost: if positive, terminate after that many
consecutive runs in which neither response nor evaluation was visible.
This matches GOAT's ``max_no_response_runs`` and FlipAttack's
``max_no_feedback_runs``.

## Deliberate Departures from the Paper

These departures are forced by superred's single-Goal session model.
None of them affect alignment on the four-setting interface contract.

### Single-instance Pareto collapses to current-best

The paper's headline candidate selection (Algorithm 2) ranks
candidates over a multi-instance ``D_pareto``. Superred has one
``Goal`` per session, so the per-instance Pareto frontier degenerates
to "candidates with the maximum score on the single instance." We
therefore default to ``current_best`` selection with latest-wins
tie-breaking — the paper's reference implementation lists
``current_best`` as a first-class candidate selector.

### No acceptance test

The paper's acceptance check ("did Φ' beat Φ on the same minibatch?")
requires running both candidates on the *same* minibatch. With a
stochastic target and a single-instance trainset, this isn't possible
— each rollout is fresh stochasticity. We therefore append every
rolled-out candidate to the pool unconditionally, and let
best-scoring selection decide which to mutate from next.

### Minibatch size 1, with a per-candidate rollout history

The paper default minibatch size is 3. With a single-instance
trainset, running the same candidate against the same goal multiple
times only produces N noisy samples of the same task, so each
*rollout* is its own data point (effective minibatch size 1).

To still feed the reflection LM the multi-rollout signal the paper
relies on, each candidate keeps a bounded ring buffer of its most
recent rollouts (size 3 — same number as the paper's minibatch).
When the same parent is re-rolled (e.g. because the previous
reflection failed to produce a parseable mutation, or because no
fresh proposal beat it in the pool), the new rollout appends to that
history; when the buffer is full the oldest entry is dropped.
``Reflector.propose`` then receives every entry in the buffer as the
side-info dataset, so the meta-prompt sees as much signal as we've
already paid for.

Parent selection (``effective_score`` on ``_Candidate``) is the
**mean score across the buffer** so a one-off lucky/unlucky rollout
doesn't dominate over a steadier candidate — the same buffer that
feeds reflection also feeds the selection rule. Latest-wins
tie-breaking still applies (relevant in settings 1 and 3, where
every candidate sits at 0.0).

Caveat: a buffer-mean over a partially-filled deque has a subtle
freshness bias. A candidate with one rollout at 0.9 (mean 0.9)
beats a candidate with three rollouts averaging 0.85 (mean 0.85),
even though the latter is the more reliable estimate. We
intentionally keep this — the bias prefers fresh exploration over
old stable results, which is the right behaviour during a
short-budget GEPA loop where every rollout costs a target call.
Future work could swap in a confidence-weighted score (e.g. a
shrinkage estimator with a fixed prior count) if the noise budget
becomes the dominant cost.

### One component, auto-claimed from the writable scope

The paper supports multi-component systems (e.g. multi-hop QA with
several modules). We optimize a single component per session.

Default channel selection is *threat-model-driven*: the optimizer
auto-claims ``system_prompt`` when the controller's scope grants it
as a writable controllable (higher-leverage attack surface, and the
single component the paper would naturally optimise on a chatbot
target), otherwise falls back to ``user_message`` and skips the
read-only ``system_prompt`` PreCall. The
``target_controllable_name`` constructor knob (default ``None``)
always wins over auto-claim and lets callers pin any in-scope
controllable as the attack channel. Multi-component (simultaneous
multi-channel) support is a future extension, not a current
requirement.

### No merge / crossover proposer

The paper's merge / crossover proposer is gated by ``use_merge=False``
in the reference implementation's ``optimize`` API. We omit it to
match the paper default and Crescendo's minimalism.

## Framework Integration Pattern

GEPA mirrors the integration shape established by Crescendo, FlipAttack,
and GOAT:

* Same ``Optimizer`` ABC, same ``on_event`` dispatch, same minimal
  ctor surface (``max_attempts``, ``response_observable_names``,
  ``max_no_signal_runs``, plus ``target_controllable_name`` for
  explicit-channel attacks).
* Same primary-controllable locking on the first non-``system_prompt``
  ``ControllablePreCallEvent`` *in the default mode*; explicit-target
  mode (``target_controllable_name`` set) locks onto exactly that name.
* Same ``system_prompt`` skip rule **only when ``system_prompt`` is
  not writable** (matches GOAT and FlipAttack on read-only system
  prompts); the skip is bypassed by auto-claim when ``system_prompt``
  is a writable controllable, or when
  ``target_controllable_name="system_prompt"`` is set explicitly.
* Same trajectory-first response recovery over
  ``{response, model_response, assistant_response}``.
* Same 3-way PostCall pairing (same controllable, request matches the
  PreCall request, or request matches the injected value).
* Same ``RunEnd → done`` semantics: ``done=True`` on success
  (``RunEndEvent.evaluation.success``), on
  ``max_no_signal_runs`` overflow, or on ``max_attempts`` exhaustion;
  ``done=False`` otherwise.
* Same delegation of success judgement to ``RunEndEvent.evaluation`` —
  no in-loop scorer (matches GOAT's pattern; the paper's evaluation
  metric is exogenous).

## Reflection on Parse Failure

If the reflection LM's output does not contain a fenced code block,
``Reflector.propose`` returns ``None`` and the optimizer logs a warning
and skips the mutation for that iteration. The next run re-rolls the
existing best candidate (no new pending proposal stashed), so the
attempt budget is not burned producing a malformed candidate. This
matches Crescendo's "scoring failed → fall back" pattern.

## One Run = One Rollout, ASR-Style Across Runs

Each superred run is one independent rollout of one candidate. Multiple
runs are how the candidate pool grows. The default
``max_attempts=20`` aligns with the paper's "few rollouts" sample-
efficient regime.

## Staged-Running Compatibility

GEPA is compatible with staged running: per-run state is reset on
``RunStartEvent``; the candidate pool, the pending mutation, and the
seed are preserved across runs; no LLM work happens during
``RunStartEvent`` itself (reflection runs in ``RunEndEvent``).
