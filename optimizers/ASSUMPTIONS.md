# Optimizer Assumptions

## Single Controllable

FlipAttack assumes a single controllable injection point (the user query).
It injects into the first ControllablePreCallEvent it receives per run
and ignores additional controllables with ControllableNoInjection.

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

**System prompt folding:** Because the optimizer assumes a single
controllable injection point, the decode instructions (normally a
system prompt) are folded into the user message. Targets that support
separate system prompt configuration can read the `system_prompt`
property directly.
