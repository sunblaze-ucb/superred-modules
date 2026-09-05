# superred-optimizer-policy-puppetry

Policy Puppetry jailbreak optimizer for [superred](https://superred.simonsure.com),
ported from NVIDIA garak's `doctor.py` probes.

> **Community port — not an official implementation.** This module vendors the
> Dr. House scene template and the leetspeak transform from
> [NVIDIA garak](https://github.com/NVIDIA/garak) (Apache-2.0). It is not
> affiliated with, endorsed by, or maintained by NVIDIA. See
> [ASSUMPTIONS.md](ASSUMPTIONS.md) for provenance and every deliberate
> deviation.

## What it does

Policy Puppetry — [HiddenLayer's "novel universal bypass"](https://hiddenlayer.com/innovation-hub/novel-universal-bypass-for-all-major-llms/)
— wraps a request in an XML-ish `<interaction-config>` block that re-declares
the model's allowed modes and forbidden strings, then asks for the answer as a
line in a Dr. House TV script. The framing competes with the real system
prompt rather than arguing with it.

Two techniques, neither of which calls an LLM:

1. **`plain`** — the goal fills upstream's `{{HARMFUL_BEHAVIOR}}` slot. One
   variant.
2. **`leet`** — adds upstream's "in code using leet speak so Cuddy doesn't
   know" aside, then leetspeaks the whole prompt. `leet_variants` variants,
   seeded `0..n-1`.

## Install

```bash
pip install superred-optimizer-policy-puppetry
```

## Usage

```python
from policy_puppetry_optimizer import PolicyPuppetryOptimizer

# plain scene + 3 seeded leetspeak variants = 4 runs
optimizer_factory = lambda: PolicyPuppetryOptimizer()

# plain only
optimizer_factory = lambda: PolicyPuppetryOptimizer(techniques=["plain"])
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `techniques` | `("plain", "leet")` | Which techniques to run, in order. |
| `leet_variants` | `3` | Seeded leetspeak variants to emit. |

One scene per run, injected once into the first free-text surface that is not
`system_prompt`; everything else is declined. Reports `done` when variants are
exhausted, and makes no LLM calls, so it costs nothing against the attacker
inference budget.

Note the seeding: upstream leetspeaks from an unseeded global RNG, so its
prompts differ every run. Here each variant is seeded, so a run is
reproducible.

## Credits / upstream

Derived from NVIDIA garak, Apache-2.0, pinned at commit
[`2233d51b`](https://github.com/NVIDIA/garak/blob/2233d51bedf6bfb634872ef9efbcd94eae38cc1a/garak/probes/doctor.py).
Verify the vendored template with `python scripts/sync_upstream.py --check`.
See [`NOTICE`](NOTICE) and [`LICENSES/NOTICE.md`](LICENSES/NOTICE.md).
