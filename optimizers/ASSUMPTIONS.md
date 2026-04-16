# Optimizer Assumptions

## Single Controllable

TAP, Crescendo, and FlipAttack assume a single controllable injection
point (the user query). They inject into the first ControllablePreCallEvent
they receive per run and ignore additional controllables with
ControllableNoInjection.

## Single LLM Model

All optimizers use the controller-provided LLM client (`self.llm`).
TAP and Crescendo use it for attacker, evaluator, and internal target
roles. FlipAttack uses it only for response scoring (the attack itself
is a pure text transformation). The model is chosen by the threat model
configuration, not by the optimizer.

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

### FlipAttack — Compatible

Each strategy attempt maps to one superred run cycle. The optimizer
rotates through flip modes (FWO, FCW, FCS, FMM) with one attempt
per run. No LLM work during RunStartEvent — the flip transformation
is a pure string operation. Fully compatible with staged running.
