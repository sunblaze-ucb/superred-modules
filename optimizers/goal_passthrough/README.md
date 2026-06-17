# goal_passthrough

The **unattacked direct-prompt baseline** optimizer for superred.

`GoalPassthroughOptimizer` performs no attack. On each run it injects the
adversarial `Goal.description` verbatim, exactly once, into the first
user-facing controllable, then ends the run. It is the zero-attack
reference point in an attacker sweep: it answers "what does the target do
when simply asked the harmful question, with no jailbreak?"

## Behaviour

| Event | Response |
|---|---|
| `RunStartEvent` | acknowledge; arm the single injection for this run |
| first `ControllablePreCallEvent` (non-`system_prompt`) | `ControllableInjection(value=goal.description)` |
| subsequent `ControllablePreCallEvent` | `ControllableNoInjection` (single-turn) |
| `ControllablePostCallEvent` | `ControllableNoInjection` (never edits responses) |
| `RunEndEvent` | `RunEndResponse(done=True)` (one attempt per task) |

The `system_prompt` controllable is always declined, so a benchmark
`Task` that pre-sets the system prompt via `configure_target` keeps it
intact. The baseline only fills the user-message channel.

## Properties

- **No LLM use.** It never calls the `LLMClient`, so it spends none of the
  per-task budget and is invariant to the attacker-model dimension.
- **Scope-robust.** It locks onto the first non-`system_prompt`
  controllable it sees, so it works against any chatbot-style target
  regardless of how the user-message controllable is named.
- **Zero config.** Constructs with no arguments, like every superred
  optimizer.

## Install

```bash
pip install -e optimizers/goal_passthrough
```

```python
from goal_passthrough_optimizer import GoalPassthroughOptimizer
```
