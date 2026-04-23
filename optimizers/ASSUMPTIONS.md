# Optimizer Assumptions

## Single Pre-Call Controllable

Crescendo assumes a single controllable injection point (the user query).
It injects into the first ControllablePreCallEvent it receives per run
and ignores additional controllables with ControllableNoInjection.

## Post-Call Controllable Mapping

Crescendo can process PostCall events from a dedicated response
controllable that differs from the PreCall injection controllable. It
locks onto the first PostCall controllable seen after an injected turn
and ignores unrelated PostCall controllables for that attempt.

## Single LLM Model

Crescendo uses the controller-provided LLM client (`self.llm`) for
attacker, evaluator, and internal target roles. The model is chosen by
the threat model configuration, not by the optimizer.

## Staged Running Compatibility

### Crescendo — One run per attempt

All turns of a single attempt happen within one superred run. The
target's conversation loop emits repeated PreCall/PostCall events;
the optimizer keeps injecting questions until success or max_turns,
then returns NoInjection to terminate the target's loop. Variant
rotation across attempts maps to separate runs.

**Departure from paper/PyRIT:** This implementation rotates through
prompt variants across attempts. PyRIT selects a single variant at
construction time with no rotation. Here, each "attempt" is a fully
independent attack with reset state and a different variant prompt,
cycling through all available variants before exhaustion.

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
