# Optimizer Assumptions

## Single Controllable

ManyShot assumes a single controllable injection point (the user query).
It injects into the first ControllablePreCallEvent it receives per run
and ignores additional controllables with ControllableNoInjection.

## Single LLM Model

ManyShot uses the controller-provided LLM client (`self.llm`) for
response scoring and optional example generation when the schedule
exceeds the bundled dataset. The model is chosen by the threat model
configuration, not by the optimizer.

## Staged Running Compatibility

### ManyShot — Compatible

Each attempt (escalating example count) maps to one superred run cycle.
No LLM work during RunStartEvent for counts <= bundled dataset size —
the prompt is assembled from static examples. For counts exceeding the
bundled set, LLM-based example generation happens during RunStartEvent.
Fully compatible with staged running.
