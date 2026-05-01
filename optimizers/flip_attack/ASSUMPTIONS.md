# Optimizer Assumptions

## Controllable Mapping

FlipAttack supports two common target setups:

1. Single-channel targets (one user-like controllable): injects one
   combined payload (system prompt + transformed TASK), then returns
   `NoInjection` on later pre-calls in the same run.
2. Chatbot-style split channels (`system_prompt`, `user_message`):
   injects decode instructions into `system_prompt`, injects one TASK
   payload into `user_message`, then stops injecting more user turns.

For post-call scoring, FlipAttack accepts the first plausible response
channel after injection (same controllable as pre-call, or request
matching the injected payload), then locks to that channel for the run.
Scoring is finalized at next pre-call / run-end and prefers
trajectory-visible response observables (`response`, `model_response`,
`assistant_response`), with post-call answer as fallback when the
response observable is not in scope.

Response observable matching is configurable through
`response_observable_names`. FlipAttack also uses name-based hints
(`response`, `assistant`, `reply`, `output`, `completion`) so targets
with slightly different observable names still work by default.

To avoid silently burning budget, FlipAttack tracks consecutive runs
with no in-scope response feedback and can stop early after
`max_no_feedback_runs` (default: 2).

## Single LLM Model

FlipAttack uses the controller-provided LLM client (`self.llm`) only to
score responses. The attack itself is a text transform and does not call
an LLM. The model is chosen by threat-model config, not by the optimizer.

## Staged Running Compatibility

### FlipAttack — Compatible

Each strategy attempt maps to one SuperRed run cycle. The optimizer
rotates through flip modes (FWO, FCW, FCS, FMM) with one attempt
per run. No LLM work during RunStartEvent — the flip transformation
is a pure string operation. Fully compatible with staged running.

**Departure from original:** This implementation rotates through flip
modes (FWO, FCW, FCS, FMM) across run cycles. The original FlipAttack
uses a single mode per attack instance. Strategy rotation is a framework
enhancement for automated red-teaming. For paper-faithful single-mode
behavior, configure one mode explicitly (for example, `flip_modes=["FCS"]`).

**Prompt faithfulness:** Prompt text and structure follow the official
FlipAttack code:
1. Vanilla/LangGPT Rule wording, triple `NEVER`, and Step 1/2 structure.
2. FMM mismatch behavior (FCS disguise with FWO-style decode instruction).
3. Few-shot example #2 hardcoded as `'noitcurtsni ym wolloF'`.

**System prompt folding:** For single-channel and limited-scope runs,
decode instructions are folded into the user payload. When a visible
`system_prompt` controllable is present, instructions are injected there
and the user payload carries only the transformed TASK.

**CoT placement:** CoT follows the original prompt family behavior.
In vanilla mode, CoT text appears in system-prompt Step 2.
In LangGPT mode, CoT text appears in the appended user-side Step 2
prompt. Few-shot examples keep the original per-mode placement.

## FlipAttack Evaluation Modes

FlipAttack supports two scoring styles for comparability:

1. `asr_gpt` (default): official judge prompt + `Rating: [[n]]` parser,
   normalized to 0.0-1.0. Success criterion follows the original code:
   only `rating == 10` counts as success.
2. `asr_dict`: refusal-string matching heuristic (0.0 refusal, 1.0
   non-refusal) using the original rejection-string list (`1.0` means success).
