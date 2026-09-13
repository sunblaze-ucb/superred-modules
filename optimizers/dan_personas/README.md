# superred-optimizer-dan-personas

DAN persona jailbreak optimizer for [superred](https://superred.simonsure.com),
ported from NVIDIA garak's DAN probe family.

> **Community port — not an official implementation.** This module vendors the
> 14 persona preambles from [NVIDIA garak](https://github.com/NVIDIA/garak)
> (Apache-2.0). It is not affiliated with, endorsed by, or maintained by
> NVIDIA. See [ASSUMPTIONS.md](ASSUMPTIONS.md) for provenance and every
> deliberate deviation from the reference code.

## What it does

A DAN prompt is a role-play primer: it tells the model to adopt an
unrestricted alter ego (DAN, STAN, DUDE, Developer Mode, …) and then to wait
for the real request. Dan_11_0 says so outright — *"don't begin the data
pairing until after my next message."*

So this optimizer sends the persona on the first turn of a run and the goal on
the second, one persona per run, cycling through all 14 until they are
exhausted. It makes no LLM calls.

## Install

```bash
pip install superred-optimizer-dan-personas
```

## Usage

```python
from dan_personas_optimizer import DANPersonasOptimizer

# all 14 personas, persona then goal
optimizer_factory = lambda: DANPersonasOptimizer()

# a subset, and a target that only offers one turn per run
optimizer_factory = lambda: DANPersonasOptimizer(
    personas=["Dan_11_0", "STAN"], mode="single_message"
)
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `personas` | all 14 | Upstream persona names, tried in order. |
| `generator_name` | from observables | Fills garak's `{generator.name}`; falls back to `ChatGPT`. |
| `mode` | `"two_turn"` | `"single_message"` joins persona and goal into one injection. |

**Pick the right mode.** `two_turn` is what the preambles ask for and works
against any target that keeps offering the same surface within a run (the
`chatbot` target does — it sends PreCall events until the optimizer declines).
Against a target that offers each surface only *once* per run, the second turn
never arrives and the goal would never be sent; use `single_message` there.

Both turns land on the surface that received the persona — a primer only
applies to the conversation it started — and `system_prompt` and non-free-text
surfaces are never injected into.

## Credits / upstream

Derived from NVIDIA garak, Apache-2.0, pinned at commit
[`2233d51b`](https://github.com/NVIDIA/garak/blob/2233d51bedf6bfb634872ef9efbcd94eae38cc1a/garak/probes/dan.py).
Refresh the vendored corpus with `python scripts/sync_upstream.py`.
See [`NOTICE`](NOTICE) and [`LICENSES/NOTICE.md`](LICENSES/NOTICE.md).
