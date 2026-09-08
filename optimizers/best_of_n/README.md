# superred-optimizer-best-of-n

Best-of-N jailbreaking optimizer (text arm) for
[superred](https://superred.simonsure.com).

> **Community port — not an official implementation.** Ported from
> [jplhughes/bon-jailbreaking](https://github.com/jplhughes/bon-jailbreaking)
> (MIT). Not affiliated with the authors. See [ASSUMPTIONS.md](ASSUMPTIONS.md).

## What it does

Best-of-N is a black-box search: rewrite the request under a random
augmentation, send it, and repeat until one gets through. Each run sends one
variant of the goal with

- **word scrambling** — shuffle the interior of words longer than 3 characters,
- **random capitalization** — flip letter case,
- **ASCII noising** — shift characters by ±1,

each applied at a rate set by `sigma`. No LLM is used to build the variants,
so an N-sample run costs N target calls and nothing else.

## Install

```bash
pip install superred-optimizer-best-of-n

# only if you want the random token prefix/suffix (off by default)
pip install "superred-optimizer-best-of-n[tokens]"
```

## Usage

```python
from best_of_n_optimizer import BestOfNOptimizer

optimizer_factory = lambda: BestOfNOptimizer()

# a wider search with heavier augmentation
optimizer_factory = lambda: BestOfNOptimizer(n_samples=100, sigma=0.6)

# capitalisation only
optimizer_factory = lambda: BestOfNOptimizer(
    word_scrambling=False, ascii_perturbation=False
)
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `n_samples` | `20` | variants to try (upstream `n_steps` × `num_concurrent_k`) |
| `sigma` | `0.4` | proportion of characters to augment |
| `word_scrambling` / `random_capitalization` / `ascii_perturbation` | `True` | enable each augmentation |
| `random_prefix_length` / `random_suffix_length` | `0` | random tokens to prepend/append (needs `[tokens]`) |
| `seed_start` | `0` | variant *i* uses `seed_start + i` |

Variants are seeded, so a run is reproducible.

## Credits / upstream

Ported from [bon-jailbreaking](https://github.com/jplhughes/bon-jailbreaking)
(MIT), pinned at
[`c1182103`](https://github.com/jplhughes/bon-jailbreaking/tree/c11821039caa3afab3e17123c0400c7a236b0cd6).
Paper: Hughes et al., *Best-of-N Jailbreaking* (arXiv:2412.03556). Verify with
`python scripts/sync_upstream.py --check`. See [`NOTICE`](NOTICE) and
[`LICENSES/NOTICE.md`](LICENSES/NOTICE.md).
