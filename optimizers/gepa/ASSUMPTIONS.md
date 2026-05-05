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
``<curr_param>`` and ``<side_info>``. The ``output_extractor``
behaviour (extract the new instruction from a fenced code block,
tolerate optional language tag, tolerate missing closing fence) is
mirrored in ``reflector.py``.

## Adversarial Information-Access Settings

The optimizer naturally operates in all four settings — there is no
setting knob. The framework's scope filter and ``include_feedback``
flag select which information surfaces are visible:

| Setting | Reflective dataset has | Selection | Behaviour |
| --- | --- | --- | --- |
| 1 user query only | ``goal`` only | latest-wins (all scores 0) | LLM-driven prompt mutation chain; no early-stop |
| 2 + feedback | ``goal``, score, rationale | best-scoring | score-driven evolution + early-stop on success |
| 3 + responses | ``goal``, response | latest-wins (all scores 0) | response-aware mutation; no early-stop |
| 4 full | ``goal``, response, score, rationale | best-scoring | full GEPA loop + early-stop |

The reflective dataset only includes fields that are non-``None`` /
non-empty (see ``RolloutRecord.to_sample`` and
``format_reflective_dataset``). Settings that strip a surface produce
a smaller side-info block rather than a noisier one.

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

### Minibatch size 1

The paper default is 3. With a single-instance trainset, running the
same candidate against the same goal multiple times only produces
N noisy samples of the same task. We use minibatch size 1 (each
rollout is its own data point); the running pool effectively serves
as the cross-candidate score history.

### One component

The paper supports multi-component systems (e.g. multi-hop QA with
several modules). We optimize a single component — the user-message
channel — which is the right shape for one-prompt jailbreak
optimization. Multi-component support is a future extension, not a
current requirement.

### No merge / crossover proposer

The paper's merge / crossover proposer is gated by ``use_merge=False``
in the reference implementation's ``optimize`` API. We omit it to
match the paper default and Crescendo's minimalism.

## Framework Integration Pattern

GEPA mirrors the integration shape established by Crescendo, FlipAttack,
and GOAT:

* Same ``Optimizer`` ABC, same ``on_event`` dispatch, same five-knob
  ctor surface (``max_attempts``, ``reflection_temperature``,
  ``response_observable_names``, ``max_no_signal_runs``, plus a single
  ``__init__`` for setup).
* Same primary-controllable locking on the first non-``system_prompt``
  ``ControllablePreCallEvent``.
* Same hardcoded ``system_prompt`` skip rule (matches GOAT and
  FlipAttack — the paper's optimization target is the user-message
  channel).
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
