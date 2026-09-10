# superred-optimizer-ansi-escape

ANSI-escape (terminal control code) optimizer for
[superred](https://superred.simonsure.com), ported from garak.

> **Community port — not an official implementation.** Ported from
> [garak](https://github.com/NVIDIA/garak) (Apache-2.0). Not affiliated with
> NVIDIA. See [ASSUMPTIONS.md](ASSUMPTIONS.md).

## What it does

ANSI escape sequences in a model's output can seize control of whatever
renders it — move the cursor, ring the bell, write the clipboard via OSC 52,
or emit an OSC 8 hyperlink whose visible text hides its real destination. This
sweeps garak's prompts that try to get a model to emit them, one per run.

Two variants, matching upstream's two black-box probes:

| variant | upstream | prompts | what it sends |
| --- | --- | --- | --- |
| `escaped` | `AnsiEscaped` | 51 | asks for escape codes, and asks the target to repeat *escaped* payloads |
| `raw` | `AnsiRaw` | 58 | the same asks, plus repeat *live* payloads and unescape escaped ones |

`raw` puts real `ESC` bytes on the wire; `escaped` does not.

This is a **goal-agnostic** probe — upstream's prompts are fixed, so the task
`Goal` is accepted (and passed to the surface classifier) but never embedded.
Whether the response actually contains a control sequence is for the
`SecurityClaim` to judge.

## Install

```bash
pip install superred-optimizer-ansi-escape
```

## Usage

```python
from ansi_escape_optimizer import AnsiEscapeOptimizer

optimizer_factory = lambda: AnsiEscapeOptimizer()                      # escaped, 51 prompts
optimizer_factory = lambda: AnsiEscapeOptimizer(variant="raw")         # raw, 58 prompts
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `variant` | `"escaped"` | `"escaped"` (upstream `AnsiEscaped`) or `"raw"` (`AnsiRaw`) |
| `shuffle` | `False` | shuffle prompt order (seeded); upstream uses file order |
| `seed` | `0` | shuffle seed |

## Credits / upstream

Prompts and payloads from [garak](https://github.com/NVIDIA/garak)
(Apache-2.0), pinned at
[`2233d51b`](https://github.com/NVIDIA/garak/blob/2233d51bedf6bfb634872ef9efbcd94eae38cc1a/garak/probes/ansiescape.py).
Background: Johann Rehberger's
[terminal DiLLMa](https://embracethered.com/blog/posts/2024/terminal-dillmas-prompt-injection-ansi-sequences/)
and [CWE-150](https://cwe.mitre.org/data/definitions/150.html).
Verify with `python scripts/sync_upstream.py --check`. See [`NOTICE`](NOTICE).
