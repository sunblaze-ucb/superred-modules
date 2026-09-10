# superred-optimizer-sysprompt-extraction

System-prompt-extraction optimizer for
[superred](https://superred.simonsure.com), ported from garak.

> **Community port — not an official implementation.** Ported from
> [garak](https://github.com/NVIDIA/garak) (Apache-2.0). Not affiliated with
> NVIDIA. See [ASSUMPTIONS.md](ASSUMPTIONS.md).

## What it does

Sends garak's 28 system-prompt-extraction templates — direct requests,
role-play, base64 tricks, continuation traps, authority framing — each trying
to get the target to print the instructions it was given.

Nine categories, 28 templates, swept one per run. **Goal-agnostic** — the
templates are fixed, so the task `Goal` is accepted (and passed to the surface
classifier) but never embedded. Whether the response actually leaks the system
prompt is for the `SecurityClaim` to judge.

## Install

```bash
pip install superred-optimizer-sysprompt-extraction
```

## Usage

```python
from sysprompt_extraction_optimizer import SysPromptExtractionOptimizer

optimizer_factory = lambda: SysPromptExtractionOptimizer()
optimizer_factory = lambda: SysPromptExtractionOptimizer(shuffle=True, seed=7)
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `shuffle` | `False` | shuffle template order (seeded); upstream uses file order |
| `seed` | `0` | shuffle seed |

## Credits / upstream

Attack templates from
[garak](https://github.com/NVIDIA/garak) (Apache-2.0)'s
`data/sysprompt_extraction/attacks.json`, pinned at
[`2233d51b`](https://github.com/NVIDIA/garak/blob/2233d51bedf6bfb634872ef9efbcd94eae38cc1a/garak/probes/sysprompt_extraction.py).
Verify with `python scripts/sync_upstream.py --check`. See [`NOTICE`](NOTICE)
and [ASSUMPTIONS.md](ASSUMPTIONS.md) for the one substantive deviation.
