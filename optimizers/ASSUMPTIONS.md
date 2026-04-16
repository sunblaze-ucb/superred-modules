# Optimizer Assumptions

## Single Controllable

Both TAP and Crescendo assume a single controllable injection point
(the user query). They inject into the first ControllablePreCallEvent
they receive per run. Multiple controllables are not explicitly handled —
additional ControllablePreCallEvents get a default EventResponse.

## Single LLM Model

Both optimizers use the controller-provided LLM client (`self.llm`) for
all internal roles (attacker, evaluator, and in TAP's case, internal
target simulation). The model is chosen by the threat model configuration,
not by the optimizer.

## Staged Running Compatibility

### Crescendo — Compatible

Each turn maps to one superred run cycle. The optimizer maintains state
across runs via instance variables. A staged runner can pause/resume
between runs naturally.

### TAP — Compatible (after v0.2.0 refactor)

Each tree depth maps to one superred run cycle. Internal work (branching,
prompt generation, on-topic pruning, internal target queries, scoring)
happens during RunStartEvent. The best candidate is tested against the
real target via PreCall/PostCall. A staged runner can pause/resume
between depth iterations.

Note: TAP still does significant internal LLM work during RunStartEvent
(querying all candidates against an internal simulated target). The real
target only sees the best candidate per depth.
