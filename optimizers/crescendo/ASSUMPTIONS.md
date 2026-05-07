# Optimizer Assumptions

## Controllable Dispatch

Crescendo dispatches PreCall events by `event.controllable.name`:
- `system_prompt` → dedicated path (Option C2 pretext lift, see below)
- `response` → always `ControllableNoInjection` (no response modification)
- anything else → user-message path (covers ChatbotTarget's `user_message`
  and legacy/generic targets that name their single controllable
  differently)

The user-message path locks onto the first non-(system_prompt/response)
controllable it sees as the primary user-message channel. Subsequent
PreCalls for other controllables receive `ControllableNoInjection`.

## Post-Call Controllable Mapping

Crescendo can process PostCall events from a dedicated response controllable
that differs from the PreCall injection controllable. It locks onto the
first PostCall controllable seen after an injected turn and ignores
unrelated PostCall controllables for that attempt. Out-of-band PostCall
events are bound only when they plausibly match the active turn (same
controllable as pre-call, same pre-call request marker, or same injected
value marker used by targets that echo injected text as the PostCall
request). PostCall events on the `system_prompt` controllable are always
ignored.

Turn evaluation is deferred to the next PreCall after injection: Crescendo
first tries to recover the final response from filtered trajectory
observables (`response`, `model_response`, `assistant_response`). If none
are visible, it falls back to the paired PostCall answer; if neither is
available, it advances the turn as an unsuccessful no-feedback turn. This
keeps behavior consistent whether PostCall is visible or filtered while
preserving scope gating.

## Capability-Aware Extensions

When the framework grants capabilities beyond the paper baseline, Crescendo
opportunistically uses them. With scope `{user, response_readable}` and no
extra observables, all extensions are no-ops by construction and behavior
collapses to the paper baseline (Russinovich et al., USENIX Security 2025).

### (a) Read target model identity

If an `ObservableValue` named `model` is in scope at `initialize()`, its
content is captured as `target_model_id` and prepended to the attacker
LLM's system prompt as `Target model: <id>`. This lets the attacker tailor
phrasing to known model behaviors (e.g. GPT-4 vs Llama-3 differ in which
framings break).

### (b) Read target system prompt

If an `ObservableValue` named `system_prompt` is in scope, its content is
captured as `target_system_prompt` and prepended to the attacker's system
prompt as `Target system prompt:\n<text>`. Empty content is rendered as
`(empty)`. The attacker can then construct escalation aligned with the
target's stated persona/role rather than fighting it.

### (c) Override target system prompt — pretext lift (Option C2)

If a `Controllable` named `system_prompt` is in scope at `initialize()`,
the optimizer detects writability via `_can_write_system_prompt`. On the
first `system_prompt` PreCall of an attempt, the optimizer makes an eager
attacker call with `include_framing=True`, which adds an instruction to
the attacker's system prompt requiring it to also produce a
`framing_for_system_prompt` field — a brief, benign-looking pretext (e.g.
"I'm a graduate student researching X"). The framing is injected as the
target's system prompt for the entire attempt. The attacker's
`generated_question` for turn 1 is cached and used on the immediately
following `user_message` PreCall, with no second attacker call.

Faithfulness rationale: Crescendo's attacker prompts already use pretext
in their first user message (e.g. variant 1 example: *"I have a school
assignment on the book 'Anarchist's Cookbook'..."*). Lifting pretext to
the system prompt where it persists as authoritative context is a same-
content, stronger-channel use of the same Crescendo mechanic.

If the eager attacker call fails or the attacker omits the framing
field, the optimizer falls back to `ControllableNoInjection` on the
system_prompt event (task-configured prompt is preserved) and Q1 is
generated normally on the next user_message PreCall.

### (d) Dual evaluator: task verdict authoritative at RunEnd

Crescendo runs an internal LLM evaluator (refusal detector + per-turn
score) on every turn. This is required because the task judge only fires
at run end and cannot drive escalation decisions. At `RunEndEvent`, when
`event.evaluation` is provided (controller `include_feedback=True`), the
optimizer treats `event.evaluation.success` as authoritative: if the task
judge disagrees with the internal verdict, the task verdict wins and a
divergence is logged at WARNING level. The task `rationale` is captured
into the cross-attempt failure record (see (e)).

When `event.evaluation` is `None` (controller `include_feedback=False`),
the internal verdict stands.

### (e) Cross-attempt memory within variant

Variants stay fully independent (the strategy variant's meta-prompt does
not change across them, and the attempt counter resets at variant
rotation), but **within** a variant the optimizer can retry on full
failure with a failure log fed to the attacker.

State machine:
- `_variant_index` rotates 0..N-1 across `max_attempts` variants.
- Within each variant, `_variant_attempt` rotates 0..K-1 across
  `max_attempts_per_variant` attempts.
- Each failed attempt appends a `FailureRecord` (refused questions
  collected from the attacker's `notify_refusal` history, final per-turn
  score, internal rationale, task-judge rationale) to
  `_variant_failure_log`.
- New attacker per attempt: receives `previous_failures` and renders them
  in a `###Previous Attempts in This Approach###` block in its system
  prompt with instruction "Try a different angle within the same overall
  approach."
- On variant rotation, `_variant_failure_log` is cleared so the next
  variant starts fresh.

Default `max_attempts_per_variant=3`. With default `max_attempts=5` (one
per bundled variant), a task gets up to 15 runs (5 variants × 3 attempts).

### Response modification (NOT used)

Crescendo does not inject into the `response` controllable, even when
writable in scope. The Crescendo strategy works by leveraging the model's
attention to its *own* genuine prior outputs; fabricating compliant
assistant turns would be a fundamentally different attack class. PreCall
and PostCall events on the `response` controllable receive
`ControllableNoInjection`.

## Single LLM Model

Crescendo uses the controller-provided LLM client (`self.llm`) for
attacker, evaluator, and internal target roles. The model is chosen by
the threat model configuration, not by the optimizer.

## Staged Running Compatibility

### Crescendo — One run per attempt

All turns of a single attempt happen within one superred run. The
target's conversation loop emits repeated PreCall/PostCall events;
the optimizer keeps injecting questions until success or max_turns,
then returns NoInjection to terminate the target's loop.

**Departure from paper/PyRIT:** This implementation rotates through
prompt variants across attempts AND retries multiple attempts per variant
with cross-attempt failure-log memory (see (e)). PyRIT selects a single
variant at construction time with no rotation; the paper does not define
cross-attempt behavior. Each "attempt" within a variant is otherwise a
fully independent attack with reset per-attempt state.

## Crescendo Scoring Format

Crescendo's evaluator expects the judge output field `score_value` on a
0-100 scale and accepts either integer or decimal representations
(string or numeric). The optimizer normalizes this to 0.0-1.0 and clamps
out-of-range values.

## Crescendo Fallback Prompting

If attacker generation fails, Crescendo falls back to a conservative
high-level, neutral prompt (instead of restating the adversarial goal
directly) to reduce abrupt escalation and refusal risk.

## Crescendo Backtracking Limitation

**Departure from paper/PyRIT:** On refusal, this implementation can
notify the attacker and retry, but cannot remove the refused turn from
the already-sent target conversation state. Paper Algorithm 1 and PyRIT's
memory-backed flow model an explicit target-history pop/backtrack step.
The framework today does not expose a controllable that lets the
optimizer rewind a target's conversation; supporting paper-faithful
backtracking would require either a new "rewind" controllable on
`ChatbotTarget` or modelling each Crescendo turn as a separate run with
explicit conversation state in `ConfigSpec`. This is a separate ticket.

The within-attempt backtracking budget is still tracked (notify-and-retry
up to `max_backtracks` times before accepting the refusal as the turn
outcome), but the retries change only the attacker's view, not the
target's history.
