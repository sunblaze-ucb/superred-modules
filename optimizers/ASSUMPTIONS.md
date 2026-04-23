# Optimizer Assumptions

## Single Controllable

Crescendo assumes a single controllable injection point (the user query).
It injects into the first ControllablePreCallEvent it receives per run
and ignores additional controllables with ControllableNoInjection.

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
