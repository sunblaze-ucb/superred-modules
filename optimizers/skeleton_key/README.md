# skeleton_key

Skeleton Key single-turn jailbreak optimizer, ported from
[Microsoft PyRIT](https://github.com/microsoft/PyRIT)
(`pyrit/executor/attack/single_turn/skeleton_key.py`, pinned at commit
`2016c4a8`). PyRIT is Microsoft's own red-teaming toolkit and the canonical
implementation of the Skeleton Key technique.

## What it does

Skeleton Key primes the target with a fabricated exchange and then asks the real
question:

1. a **skeleton-key preamble** (a user turn instructing the model to update its
   behaviour and comply with any request, adding a warning rather than
   refusing), and
2. a **simulated assistant acceptance** (the model appears to have already
   agreed),

followed by the actual **objective**. Upstream implements this by *prepending*
`[user: preamble, assistant: acceptance]` to the conversation and then sending
the objective as a single turn.

superred's controllable model injects one string per turn and exposes no
editable conversation history, so this port renders the fabricated exchange
ahead of the objective exactly the way PyRIT itself renders a prepended
conversation for a target **without** editable history — its turn-based
`ConversationContextNormalizer` transcript:

```
Turn 1:
user: <skeleton-key preamble>
assistant: <simulated acceptance>
Turn 2:
user: <objective>
```

The skeleton-key preamble therefore always precedes the objective. The preamble
and acceptance bodies are loaded from the byte-identical vendored seed files;
they are never reproduced in source, tests, or docs.

## What it is not

Skeleton Key is **prompt-only and deterministic**: it makes no model calls of
its own (`self.llm` is available but unused), pulls in no heavy dependency, and
injects one self-contained turn per run into the first eligible free-text
surface. It runs no self-judge — success is decided by the target's evaluation /
the paired security claim, as elsewhere in superred.

## Usage

```python
from skeleton_key_optimizer import SkeletonKeyOptimizer

opt = SkeletonKeyOptimizer()  # vendored PyRIT preamble + acceptance
```

Constructor knobs mirror upstream `SkeletonKeyAttack`:

- `skeleton_key_prompt` — override the preamble (default: vendored).
- `skeleton_key_acceptance` — override the simulated acceptance (default: vendored).
- `max_attempts_on_failure` — retry budget; the deterministic payload is (re)sent
  across up to `1 + max_attempts_on_failure` runs (default `0` → one send).

## Dependencies

`superred`, `pyyaml`. No heavy dependencies (no `torch`, no `transformers`, no
PyRIT at runtime — the two seed prompts are vendored and read with stdlib +
PyYAML).

## Faithfulness

See [ASSUMPTIONS.md](ASSUMPTIONS.md) for every deliberate deviation. The main
one: superred has no editable conversation history, so the prepended
user/assistant exchange is flattened into one injected turn using PyRIT's own
no-editable-history transcript format.

## Provenance / citation

Skeleton Key has no academic paper. It was disclosed by Microsoft: Mark
Russinovich, "Mitigating Skeleton Key, a new type of generative AI jailbreak
technique," Microsoft, 2024. When reporting results, cite that disclosure and
Microsoft PyRIT at commit `2016c4a8`. Author: Rishabh Sinha.
