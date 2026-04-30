# Optimizer Assumptions

## Controllable Mapping

FlipAttack supports two target patterns:

1. Single-channel targets (one user-like controllable): injects one
   folded payload (system prompt + transformed TASK) and then returns
   NoInjection on later pre-calls for that run.
2. Chatbot-style split channels (`system_prompt`, `user_message`):
   injects decode instructions into `system_prompt`, injects one TASK
   payload into `user_message`, then stops further user turns.

For post-call scoring, FlipAttack accepts the first plausible response
channel after injection (same controllable as pre-call, or request
matching the injected payload), then locks to that channel for the run.
Scoring is finalized at next pre-call / run-end and prefers
trajectory-visible response observables (`response`, `model_response`,
`assistant_response`), with post-call answer as fallback when the
response observable is not in scope.

## Single LLM Model

FlipAttack uses the controller-provided LLM client (`self.llm`) only for
response scoring (the attack itself is a pure text transformation). The
model is chosen by the threat model configuration, not by the optimizer.

## Staged Running Compatibility

### FlipAttack — Compatible

Each strategy attempt maps to one superred run cycle. The optimizer
rotates through flip modes (FWO, FCW, FCS, FMM) with one attempt
per run. No LLM work during RunStartEvent — the flip transformation
is a pure string operation. Fully compatible with staged running.

**Departure from original:** This implementation rotates through flip
modes (FWO, FCW, FCS, FMM) across run cycles. The original FlipAttack
uses a single mode per attack instance. Strategy rotation is a framework
enhancement for automated red-teaming.

**Prompt faithfulness:** Prompt text and structure follow the official
FlipAttack code:
1. Vanilla/LangGPT Rule wording, triple `NEVER`, and Step 1/2 structure.
2. FMM mismatch behavior (FCS disguise with FWO-style decode instruction).
3. Few-shot example #2 hardcoded as `'noitcurtsni ym wolloF'`.

**System prompt folding:** For single-channel and limited-scope runs,
decode instructions are folded into the user payload. When a visible
`system_prompt` controllable is present, instructions are injected there
and the user payload carries only the transformed TASK.

**CoT placement:** CoT/decode instructions live in the system prompt
template (Step 2 / Target), matching the original structure. The user
payload carries only examples plus flipped TASK text.

## FlipAttack Evaluation Modes

FlipAttack supports two scoring styles for comparability:

1. `asr_gpt` (default): official judge prompt + `Rating: [[n]]` parser,
   normalized to 0.0-1.0.
2. `asr_dict`: refusal-string matching heuristic (0.0 refusal, 1.0
   non-refusal) using the original rejection-string list.
