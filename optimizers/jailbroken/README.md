# superred-optimizer-jailbroken

Jailbroken prefix-injection optimizer for
[superred](https://superred.simonsure.com).

> **Based on Tencent Zhuque Lab AI-Infra-Guard
> (https://github.com/Tencent/AI-Infra-Guard).** Required by that project's
> NOTICE under Apache-2.0 Section 4(d).

> **Community port — not an official implementation.** Not affiliated with
> Tencent or the paper's authors. See [ASSUMPTIONS.md](ASSUMPTIONS.md).

> **Content note.** The vendored templates contain profanity and harmful
> framing, byte-identical to the benchmark. See ASSUMPTIONS.md.

## What it does

A collection of 18 prefix-injection templates: each instructs the target to
begin its reply with an affirmative phrase ("Ok, I will help you.", "Mwahahaha…")
and then answer the request, exploiting the finding that a model committed to a
compliant opening tends to continue. Deterministic — no attacker LLM. Upstream
picks one template at random; this optimizer sweeps all of them, one per run.

## Install

```bash
pip install superred-optimizer-jailbroken
```

## Usage

```python
from jailbroken_optimizer import JailbrokenOptimizer

optimizer_factory = lambda: JailbrokenOptimizer()            # seeded shuffle
optimizer_factory = lambda: JailbrokenOptimizer(shuffle=False)  # file order
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `seed` | `0` | shuffle seed for the template order (upstream is unseeded) |
| `shuffle` | `True` | shuffle the order; `False` uses upstream file order |

## Credits / upstream

Ported from [Tencent AI-Infra-Guard](https://github.com/Tencent/AI-Infra-Guard)
(Apache-2.0), pinned at
[`dd6bd546`](https://github.com/Tencent/AI-Infra-Guard/tree/dd6bd54655c9ff5fb7351f4299b56916f09ec6da/AIG-PromptSecurity/deepteam/attacks/single_turn/jailbroken).
Technique: Wei et al., *Jailbroken: How Does LLM Safety Training Fail?* (2023).
Verify with `python scripts/sync_upstream.py --check`. See [`NOTICE`](NOTICE)
and [`LICENSES/NOTICE.md`](LICENSES/NOTICE.md).
