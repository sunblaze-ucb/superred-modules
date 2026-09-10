# superred-optimizer-av-spam

AV/anti-spam signature-smuggling optimizer for
[superred](https://superred.simonsure.com), ported from garak.

> **Community port — not an official implementation.** Ported from
> [garak](https://github.com/NVIDIA/garak) (Apache-2.0). Not affiliated with
> NVIDIA. See [ASSUMPTIONS.md](ASSUMPTIONS.md).

## What it does

Antivirus and anti-spam scanners recognise three standard **test** strings —
EICAR, GTUBE and GTPHISH. They are deliberately harmless: they exist so you
can check a scanner works without handling real malware. A system that scans
its own output should refuse to emit one. This asks the target to, five
different ways.

| variant | upstream | signature |
| --- | --- | --- |
| `eicar` (default) | `EICAR` | EICAR antivirus test file |
| `gtube` | `GTUBE` | GTUBE anti-spam test string |
| `gtphish` | `GTphish` | GTPHISH anti-phishing test string |

Five prompts each. **Goal-agnostic** — the signature is fixed, so the task
`Goal` is accepted (and passed to the surface classifier) but never embedded.

> **Scope.** This only says something about a target that is *expected* to
> scan its output. Against a plain chat model, emitting EICAR is not a
> finding — the model is just repeating a string you gave it. Upstream ships
> all three probes `active=False` for exactly this reason.

## Install

```bash
pip install superred-optimizer-av-spam
```

## Usage

```python
from av_spam_optimizer import AvSpamOptimizer

optimizer_factory = lambda: AvSpamOptimizer()                    # eicar
optimizer_factory = lambda: AvSpamOptimizer(variant="gtube")     # anti-spam
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `variant` | `"eicar"` | `eicar`, `gtube` or `gtphish` |
| `shuffle` | `False` | shuffle prompt order (seeded); upstream uses file order |
| `seed` | `0` | shuffle seed |

## Credits / upstream

Signatures and prompts from [garak](https://github.com/NVIDIA/garak)
(Apache-2.0), pinned at
[`2233d51b`](https://github.com/NVIDIA/garak/blob/2233d51bedf6bfb634872ef9efbcd94eae38cc1a/garak/probes/av_spam_scanning.py).
Background: [EICAR test file](https://en.wikipedia.org/wiki/EICAR_test_file),
[GTUBE](https://spamassassin.apache.org/gtube/).
Verify with `python scripts/sync_upstream.py --check`. See [`NOTICE`](NOTICE).
