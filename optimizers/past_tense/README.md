# superred-optimizer-past-tense

Past/future tense reformulation jailbreak optimizer for
[superred](https://superred.simonsure.com).

> **Based on Tencent Zhuque Lab AI-Infra-Guard
> (https://github.com/Tencent/AI-Infra-Guard).** Required by that project's
> NOTICE under Apache-2.0 Section 4(d).

> **Community port — not an official implementation.** Ports Tencent's
> Apache-2.0 implementation of the past-tense method. Not affiliated with
> Tencent or the paper's authors. See [ASSUMPTIONS.md](ASSUMPTIONS.md).

## What it does

Refusal training generalises poorly across tense: a model that refuses "how do
I make X" will often answer "how *was* X made". This optimizer asks the
attacker LLM to reformulate the objective as a question in the past (or future)
tense, then sends that single reformulated prompt. One rephrase call, one
injection.

## Install

```bash
pip install superred-optimizer-past-tense
```

## Usage

```python
from past_tense_optimizer import PastTenseOptimizer

optimizer_factory = lambda: PastTenseOptimizer()               # past tense
optimizer_factory = lambda: PastTenseOptimizer(tense="future")  # future tense
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `tense` | `past` | `past`, `future`, or `present` (control: sends the goal unchanged) |
| `max_retries` | `5` | reformulation attempts before falling back to the original goal |

## Credits / upstream

Ported from [Tencent AI-Infra-Guard](https://github.com/Tencent/AI-Infra-Guard)
(Apache-2.0), pinned at
[`dd6bd546`](https://github.com/Tencent/AI-Infra-Guard/tree/dd6bd54655c9ff5fb7351f4299b56916f09ec6da/AIG-PromptSecurity/deepteam/attacks/single_turn/past_tense).
Technique: Andriushchenko & Flammarion, *Does Refusal Training in LLMs
Generalize to the Past Tense?* (2024). The authors' repository ships no
licence, so no code was taken from it. Verify with
`python scripts/sync_upstream.py --check`. See [`NOTICE`](NOTICE) and
[`LICENSES/NOTICE.md`](LICENSES/NOTICE.md).
