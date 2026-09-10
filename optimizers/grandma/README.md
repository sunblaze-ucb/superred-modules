# superred-optimizer-grandma

Grandma-exploit (dead-relative roleplay) optimizer for
[superred](https://superred.simonsure.com), ported from garak.

> **Community port — not an official implementation.** Ported from
> [garak](https://github.com/NVIDIA/garak) (Apache-2.0). Not affiliated with
> NVIDIA. See [ASSUMPTIONS.md](ASSUMPTIONS.md).

## What it does

Wraps the request in grief and nostalgia — *"my deceased grandmother used to
read me … to help me fall asleep"* — so that refusing reads as unkindness
rather than as safety. The 2023 "Windows key lullaby" is the well-known
instance.

Five variants, matching upstream's five probes:

| variant | upstream | prompts | goal-driven |
| --- | --- | --- | --- |
| `intent` (default) | `GrandmaIntent` | 540 | **yes** — the task Goal goes in the scene |
| `win10` | `Win10` | 6 | no |
| `win11` | `Win11` | 6 | no |
| `slurs` | `Slurs` | 12 | no |
| `substances` | `Substances` | 10 | no |

## Install

```bash
pip install superred-optimizer-grandma
```

## Usage

```python
from grandma_optimizer import GrandmaOptimizer

optimizer_factory = lambda: GrandmaOptimizer()                          # intent, goal-driven
optimizer_factory = lambda: GrandmaOptimizer(variant="win10")           # product keys
optimizer_factory = lambda: GrandmaOptimizer(shuffle=True, seed=7)      # sample the 540
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `variant` | `"intent"` | which upstream probe to run |
| `shuffle` | `False` | shuffle prompt order (seeded); upstream uses file order |
| `seed` | `0` | shuffle seed |

> `intent` builds **540** prompts (6 personas × 3 actions × 5 adjectives × 6
> activities). A run budget below that samples a prefix of the sweep, so pass
> `shuffle=True` to spread the sample across the space rather than taking the
> first persona only.

## Credits / upstream

Templates and payloads from [garak](https://github.com/NVIDIA/garak)
(Apache-2.0), pinned at
[`2233d51b`](https://github.com/NVIDIA/garak/blob/2233d51bedf6bfb634872ef9efbcd94eae38cc1a/garak/probes/grandma.py).
Verify with `python scripts/sync_upstream.py --check`. See [`NOTICE`](NOTICE).
