# Optimizer Assumptions

## Single Controllable

TAP assumes a single controllable injection point (the user query).
It injects into the first ControllablePreCallEvent it receives per run
and ignores additional controllables with ControllableNoInjection.

## Single LLM Model

TAP uses the controller-provided LLM client (`self.llm`) for attacker,
evaluator, and internal target roles. The model is chosen by the threat
model configuration, not by the optimizer.

## Staged Running Compatibility

### TAP — Compatible (after v0.2.0 refactor)

Each tree depth maps to one superred run cycle. Internal work (branching,
prompt generation, on-topic pruning, internal target queries, scoring)
happens during RunStartEvent. The best candidate is tested against the
real target via PreCall/PostCall. A staged runner can pause/resume
between depth iterations.

Note: TAP still does significant internal LLM work during RunStartEvent
(querying all candidates against an internal simulated target). The real
target only sees the best candidate per depth.
