# GOAT Optimizer Assumptions

Reference: Pavlova et al., "Automated Red Teaming with GOAT: the
Generative Offensive Agent Tester," arXiv:2410.01606.

## Algorithmic Faithfulness

Algorithm 1 maps to the implementation as follows:

* A fresh attacker conversation history `C_A` is built per attempt
  (one superred run = one attempt). The attacker is constructed in
  `RunStartEvent`.
* The attacker system prompt (Fig A.1) is rendered once and used for
  the whole conversation.
* The first per-attempt PreCall triggers the initial-prompt template
  (Fig A.2); subsequent PreCalls trigger the follow-up template
  (Fig A.3).
* Per turn, the attacker's full reasoning (Observation, Thought,
  Strategy, Reply as a single JSON object) is appended to `C_A`
  (Algorithm 1 line 12); only the parsed `response` slot is forwarded
  to the target via `ControllableInjection` (line 13).
* All seven attacks from paper Table 1 are bundled and listed
  simultaneously in the attacker system prompt (paper main-result
  configuration). For per-attack ablation, instantiate the optimizer
  with `attacks=(SOME_ATTACK,)`.
* `max_turns=5` and `max_attempts=10` defaults match the paper's
  default conversation-turn cap and ASR@10 metric.

## Output Format Contract — Deliberate Departure

The paper does not constrain the attacker's output format. We require
a single JSON object with keys `observation`, `thought`, `strategy`,
`response` so the `response` slot can be lifted cleanly off the
chain-of-thought and injected into the target.

We tolerate optional markdown fences around the JSON; we reject
output that is not valid JSON, is missing a required key, has a
non-string field, or has an empty `response`. On parse failure we do
not retry (matching Crescendo's pattern); the optimizer substitutes a
benign fallback prompt for that single turn so the attempt budget is
not burned.

## Attack Examples Are Empty

The paper Table 1 prints attack names and definitions but explicitly
withholds representative example dialogues "to effectively balance
disclosure of reproducible results and enabling of malicious actors."
Accordingly, we render the `Technique Examples:` slot as
`(omitted; see paper Table 1)`. Fabricating examples would be a
departure from the paper.

## Single Pre-Call Controllable

GOAT injects into a single user-message-style controllable per
attempt. The first non-`system_prompt` `ControllablePreCallEvent` seen
in a run is locked in as the primary; later events with different
controllables get `ControllableNoInjection`.

`ControllablePreCallEvent`s for `system_prompt` (e.g. the pre-loop
event ChatbotTarget emits) are passed on without locking — the
paper's attacks live entirely in user turns, and the user-message
channel must remain eligible for the lock when it arrives. This is
hardcoded for simplicity (matches FlipAttack's explicit name handling).

## Trajectory-First Response Recovery

For each turn, `r_T` is recovered with the following preference order:

1. The latest in-scope response observable on the trajectory
   (`response`, `model_response`, or `assistant_response` by default;
   configurable via `response_observable_names`).
2. The most recent `ControllablePostCallEvent.answer` paired with the
   active turn.
3. Empty string — the paper's "If the model response (M) is empty"
   branch is reused as the graceful default for missing responses.

The trajectory is preferred so behaviour stays consistent across
scope configurations.

## Adversarial Information-Access Settings

The optimizer naturally operates in all four settings — there is no
setting knob. The framework's scope filter and `include_feedback`
flag select which information surfaces are visible:

| Setting | What the optimizer sees |
| --- | --- |
| 1 | user query injection only — no responses, no feedback |
| 2 | user query injection + `RunEndEvent.evaluation` |
| 3 | user query injection + target responses (no feedback) |
| 4 | user query injection + target responses + feedback |

In all settings the optimizer never errors on missing information;
it logs and continues. Settings 1 and 2 produce a degenerate run
because the attacker can't observe `r_T` between turns, so each turn
falls back to the "first prompt" reasoning branch with `M=""`.
Setting 3 reproduces the paper's main in-conversation loop. Setting 4
additionally enables early-stop: if `RunEndEvent.evaluation.success`
is True the optimizer signals `done=True` immediately, matching ASR@k
semantics.

`max_no_response_runs` (default 0, disabled) bounds setting-1 / 2
cost by terminating the optimizer after that many consecutive runs in
which no response observable was visible.

## One Run = One Conversation, ASR@k Across Runs

Each superred run is one independent `K`-turn attack conversation
(default `K=5`). Multiple runs map to the paper's ASR@k metric
(default `max_attempts=10`).

## No In-Loop Judge

The optimizer does not run any LLM-based scoring of target replies
during a conversation. The paper does not run an in-conversation
judge either. Success is signalled by the framework via
`RunEndEvent.evaluation.success`, which the security claim's
`Task.evaluate` produces post-run.

## No Backtracking

The paper does not backtrack within a conversation; neither does this
implementation. The attacker LLM is expected to switch attack
techniques on its own through the
`Observation → Thought → Strategy` reasoning step.

## Single LLM Model

The optimizer uses the controller-provided `LLMClient` (`self.llm`)
for all attacker calls. The model and budget are chosen by the
threat-model configuration, not by the optimizer.

## Staged-Running Compatibility

GOAT is compatible with staged running: per-attempt state is reset
on `RunStartEvent`, a fresh attacker conversation history is created
per attempt, and no LLM work happens during `RunStartEvent` itself
(attacker construction is a pure string operation).
