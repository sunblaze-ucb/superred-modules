# Assumptions and deviations -- context_compliance

Ported from microsoft/PyRIT's `context_compliance` technique at commit
`2016c4a`. PyRIT is Microsoft's own toolkit and the authoritative CCA
implementation; the technique is credited to Russinovich & Salem
(arXiv:2503.05264). Deviations from upstream, and why each is necessary or safe.

## Conversation-history injection: single-message encoding (the main deviation)

CCA's whole mechanism is a fabricated **prior conversation**: a user question
`Q`, an assistant answer `A` that appears to already agree, and then the real
request. In PyRIT this is a genuine multi-message `prepended_conversation`
(a list of role-tagged `Message`s) attached before the real turn is sent.

superred's optimizer -> target channel has **no primitive for injecting prior
conversation turns**. A `ControllableInjection` carries a single `value: str`
for one controllable event; the target decides what to do with it. The
conversation surfaces that exist are all single-string user turns:

- The `user_message` controllable (e.g. `chatbot`, `openclaw`, `safeclawarena`)
  appends exactly one user turn per injection; the target itself accumulates the
  wire conversation across turns.
- The only way to place an *assistant* turn on the wire is the `response`
  controllable (`MODEL_TAG`), which **overrides the model's own answer after a
  real provider call**. That is a response-tampering capability, a different and
  stronger threat model than CCA (which is a *user* fabricating context), and it
  cannot fabricate a prior assistant turn without first spending a real call and
  holding `model` scope. It is therefore not used.

I confirmed there is no injectable conversation/history surface (only read-only
`conversation_history` *query* specs exist on some targets). So, per the build
spec's explicit fallback, the fabricated `[user Q][assistant A][user real]`
history is **encoded into a single free-text user-message string** as a labelled
transcript:

```
User: <Q>

Assistant: <A>

User: <real request>
```

This is the same shape the sibling single-turn ports use when the framework can
only carry one user turn (`prefill` folds its assistant-priming prefix into the
user message; `many_shot` packs fabricated user/assistant dialogue pairs into
one message). The role labels (`User:` / `Assistant:`) are structural markers,
not vendored payload text. The fabricated assistant turn still precedes the real
request, which is the property the attack depends on.

## Final real request defaults to the objective (not the fixed "yes.")

PyRIT's technique wiring (`setup/initializers/techniques/core.py`) sets
`final_user_message="yes."` and relies on the simulated-target template to weave
the objective into `A`'s offer, so a bare "yes." accepts it.

This port keeps that template behaviour (the objective is still rendered into
both templates, so `A` still contains the offer), but defaults the **final real
request to the objective itself**, so the injected transcript is unambiguously a
fabricated assistant turn *followed by the real request*. Reasons:

- With the history collapsed into one user message (above), a lone "yes." is a
  weaker, more fragile cue than restating the request; making the real request
  explicit keeps the single-injection attack self-contained.
- It matches the general CCA description ("...then sends the real request").

`final_user_message` reproduces upstream exactly when set to `"yes."`
(`UPSTREAM_FINAL_USER_MESSAGE`). Both are faithful CCA variants; only which text
ends the transcript differs.

## Native re-implementation, not the PyRIT engine

Upstream generates the simulated conversation by running a full `RedTeamingAttack`
(with a `TrueFalseScorer`, `CentralMemory`, `PromptNormalizer`, and the
adversarial/simulated targets) via `generate_simulated_conversation_async`. A
superred optimizer cannot drive that engine (it only injects at controllables
and reads answers back), and importing it would pull PyRIT and its heavy deps.

Because the technique uses `num_turns=1`, that whole machinery reduces to exactly
two role-conditioned model calls, which this port makes directly through
`self.llm`:

1. adversarial system prompt (`context_compliance.yaml`, rendered with the
   objective) + the objective as the kickoff user turn -> the benign question `Q`;
2. simulated-target system prompt (`context_compliance_target.yaml`, rendered
   with the objective and `num_turns`) + `Q` as the user turn -> the affirmative
   answer `A`.

The scorer/memory/normalizer layers exist only to *orchestrate and record* the
simulation; none change the two messages produced, so omitting them is faithful.

## Stdlib template renderer instead of Jinja2

PyRIT renders these `SeedPrompt` YAMLs with Jinja2. Both context-compliance
templates use only plain `{{ variable }}` substitution -- no control blocks,
filters, or expressions -- so a small stdlib regex substitution
(`vendored.render_seed_prompt`) is byte-equivalent for these files and keeps the
package free of a Jinja2 (and PyRIT) runtime dependency. The sync checker guards
the templates, so a future upstream template that added Jinja logic would be
caught as drift and this assumption revisited.

## No LLM surface classifier

The fabricated transcript is injected into the first eligible free-text surface,
using only the name/value-type backstop (never the reserved `system_prompt`;
require a prose value type). No LLM surface-classification pass is run (unlike
`jailbroken`/`prefill`/`actor_attack`). CCA targets an ordinary user-input
conversation surface, for which the backstop is sufficient.

## Single deterministic attempt; built once

The fabricated exchange is generated once at `initialize` (two `self.llm` calls)
and stored. `RunStart` re-arms the per-run injection flag; `RunEnd` reports
`done=True`. CCA is one fabricated context and one injection per task, matching
`prefill`. If the exchange cannot be built (see below) the optimizer declines
every injection and still reports `done=True`.

## Budget / unavailable LLM handling

Each generation call re-raises a genuine `BudgetExhaustedError` (cost already
spent) so a spent run is never reported as a quietly finished attack. The
zero-cost noop client that the controller hands non-LLM optimizers raises the
same error with nothing spent; that (and any other call failure) degrades
quietly to "no fabricated exchange", so the payload is `None` and the optimizer
injects nothing -- it never falls back to authored payload text.

## num_turns

Defaults to `1` (upstream). It is passed to both template renders: as
`max_turns` to the adversarial template (a declared-but-unused parameter there)
and as `num_turns` to the simulated-target template (which renders it into the
prompt). Values `< 1` are clamped to `1`.
