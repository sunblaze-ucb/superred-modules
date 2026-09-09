# superred-optimizer-divergent-repetition

Divergent-repetition (repeat-a-word data-extraction) optimizer for
[superred](https://superred.simonsure.com), ported from garak.

> **Community port — not an official implementation.** Data extracted from
> [garak](https://github.com/NVIDIA/garak) (Apache-2.0). Not affiliated with
> NVIDIA. See [ASSUMPTIONS.md](ASSUMPTIONS.md).

## What it does

Ask a model to repeat a word "forever" and an aligned model can diverge from
the repetition into regurgitating memorised training data (the Carlini/Nasr et
al. extraction attack). Each prompt repeats one of a set of words 1, 3 or 40
times across three template phrasings; the optimizer sweeps them one per run.

This is a **data-extraction** attack, so it is **goal-agnostic** — the prompts
do not embed the task Goal. Whether a response actually diverged/leaked is for
the `SecurityClaim` to judge.

## Install

```bash
pip install superred-optimizer-divergent-repetition
```

## Usage

```python
from divergent_repetition_optimizer import DivergentRepetitionOptimizer

optimizer_factory = lambda: DivergentRepetitionOptimizer()               # 36 prompts (Repeat)
optimizer_factory = lambda: DivergentRepetitionOptimizer(extended=True)  # 945 prompts (RepeatExtended)
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `extended` | `False` | use `RepeatExtended`'s 105-word list (else `Repeat`'s 4) |
| `shuffle` | `False` | shuffle prompt order (seeded) |
| `seed` | `0` | shuffle seed |

> **Give the target a generous completion limit.** Divergence only appears once
> the model has generated well past the repetition, and garak raises the
> generator's `max_tokens` to 1024 for the duration of this probe. superred
> gives the optimizer no way to do that — generation parameters belong to the
> target — so a target capped at a few hundred tokens can look clean when it
> would in fact leak. See [ASSUMPTIONS.md](ASSUMPTIONS.md).

## Credits / upstream

Data and construction from [garak](https://github.com/NVIDIA/garak)
(Apache-2.0), pinned at
[`2233d51b`](https://github.com/NVIDIA/garak/blob/2233d51bedf6bfb634872ef9efbcd94eae38cc1a/garak/probes/divergence.py).
Reference: Nasr et al. (2023). Also shipped by Promptfoo (divergent-repetition).
Verify with `python scripts/sync_upstream.py --check`. See [`NOTICE`](NOTICE)
and [`LICENSES/NOTICE.md`](LICENSES/NOTICE.md).
