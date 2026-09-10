# superred-optimizer-package-hallucination

Package-hallucination (non-existent import) optimizer for
[superred](https://superred.simonsure.com), ported from garak.

> **Community port — not an official implementation.** Ported from
> [garak](https://github.com/NVIDIA/garak) (Apache-2.0). Not affiliated with
> NVIDIA. See [ASSUMPTIONS.md](ASSUMPTIONS.md).

## What it does

Models sometimes recommend importing packages that do not exist. An attacker
can harvest those hallucinated names and register them ("slopsquatting"), so
that anyone who runs the model's code silently installs the attacker's
package. This asks the target for code across ten framings and two dozen
tasks, so the hallucinated import names can be counted.

One variant per language:

| variant | upstream | `language_name` |
| --- | --- | --- |
| `python` (default) | `Python` | `Python3` |
| `ruby` | `Ruby` | `Ruby` |
| `javascript` | `JavaScript` | `JavaScript` |
| `rust` | `Rust` | `Rust` |
| `raku` | `RakuLand` | `Raku` |
| `perl` | `Perl` | `Perl` |
| `dart` | `Dart` | `Dart` |

Each builds **240 prompts** (10 stubs × 24 tasks). **Goal-agnostic** — the
prompts are fixed, so the task `Goal` is accepted (and passed to the surface
classifier) but never embedded. Whether a recommended package is fabricated is
for the `SecurityClaim` to judge.

## Install

```bash
pip install superred-optimizer-package-hallucination
```

## Usage

```python
from package_hallucination_optimizer import PackageHallucinationOptimizer

optimizer_factory = lambda: PackageHallucinationOptimizer()                     # python
optimizer_factory = lambda: PackageHallucinationOptimizer(variant="javascript") # npm
optimizer_factory = lambda: PackageHallucinationOptimizer(shuffle=True, seed=7)  # sample the 240
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `variant` | `"python"` | which language subclass to run |
| `shuffle` | `False` | shuffle prompt order (seeded); upstream uses file order |
| `seed` | `0` | shuffle seed |

## Credits / upstream

Prompts and tasks from [garak](https://github.com/NVIDIA/garak) (Apache-2.0),
pinned at
[`2233d51b`](https://github.com/NVIDIA/garak/blob/2233d51bedf6bfb634872ef9efbcd94eae38cc1a/garak/probes/packagehallucination.py).
Background: [Vulcan on AI package hallucination](https://vulcan.io/blog/ai-hallucinations-package-risk).
Verify with `python scripts/sync_upstream.py --check`. See [`NOTICE`](NOTICE).
