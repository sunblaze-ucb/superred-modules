# Assumptions and deviations — skeleton_key

Ported from Microsoft PyRIT `pyrit/executor/attack/single_turn/skeleton_key.py`
at commit `2016c4a8`. Deviations from upstream, and why each is necessary or
safe.

## Prepended conversation → one injected turn (no editable history)

Upstream `SkeletonKeyAttack._setup_async` sets
`context.prepended_conversation = [Message(role="user", preamble),
Message(role="assistant", acceptance)]` and then `PromptSendingAttack` sends the
objective. PyRIT's `PrependedConversationConfig` explicitly handles targets
**without editable conversation history** by rendering that prepended history and
the live request together as one text block, via a `MessageStringNormalizer`
(default `ConversationContextNormalizer`, whose format is
`"Turn N:\n<role>: <content>"`).

superred's controllable interface injects a single string per turn and exposes
no editable history, so this port takes exactly that no-editable-history path:
the fabricated `[user: preamble, assistant: acceptance]` exchange is flattened
ahead of the objective using PyRIT's own `ConversationContextNormalizer`
transcript:

```
Turn 1:
user: <preamble>
assistant: <acceptance>
Turn 2:
user: <objective>
```

This preserves the technique's essence — the model is primed by a skeleton-key
preamble and a simulated agreement **before** it sees the objective — and keeps
the preamble strictly before the objective. It is faithful to how PyRIT itself
renders this attack against a single-turn / non-editable-history target; a
target that *did* expose editable history would instead receive two real
role-structured messages, which superred does not model.

## Prompt-only: no model calls, no self-judge

Skeleton Key is deterministic and prompt-only. Upstream's optional
`AttackConverterConfig` (request converters) and `AttackScoringConfig` (scorers)
are attack-orchestration wrappers, not part of the technique itself. This port
makes **no** model calls: `self.llm` is stored per the optimizer contract but is
never used, and no LLM surface classifier is run (unlike `deepinception` /
`prefill`; matching `mtjb_coa`, the name/value-type backstop selects the
surface). Success is not self-judged here — it is decided by the target
evaluation / paired security claim, consistent with how superred separates
attacker action from claim scoring.

## Retry budget across runs

Upstream exposes `max_attempts_on_failure` (default `0`) and retries the send on
a scored failure. superred runs the optimizer per target run and re-arms per-run
state on `RunStartEvent`; there is no in-optimizer scorer to condition a retry
on. This port therefore offers up to `1 + max_attempts_on_failure` sends of the
(identical, deterministic) payload across successive runs and relies on the
controller / claim to stop the task early on success. With the default `0` the
optimizer reports `done=True` after a single run — faithful to upstream's
single-send default.

## One self-contained turn per run; surface selection

Each run injects the full primed payload once, into the first eligible free-text
controllable (the per-run `_injected` flag prevents a second injection in the
same run). The reserved `system_prompt` controllable is never injected into
(Skeleton Key targets the user turn, not the system prompt), and non-free-text
value types (e.g. `json`) are declined so a prose payload is not discarded. No
surface is pinned across runs, because each run is an independent single-turn
attack rather than a continued conversation.

## SeedDataset loading without PyRIT

Upstream loads each `.prompt` file with
`SeedDataset.from_yaml_file(path).prompts[0].value`, which pulls in PyRIT's
Pydantic seed-model stack. To avoid importing PyRIT (and its heavy transitive
dependencies), `vendored.py` reads the same byte-identical YAML with stdlib +
PyYAML and returns the first prompt-typed seed's `value`, reproducing
`.prompts[0].value` exactly for these single-seed files.
