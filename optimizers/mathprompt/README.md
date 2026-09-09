# superred-optimizer-mathprompt

MathPrompt (symbolic-math encoding) jailbreak optimizer for
[superred](https://superred.simonsure.com), ported from DeepTeam.

> **Community port — not an official implementation.** Vendors the templates
> from [DeepTeam](https://github.com/confident-ai/deepteam) (Apache-2.0). Not
> affiliated with Confident AI. See [ASSUMPTIONS.md](ASSUMPTIONS.md).

## What it does

MathPrompt rewrites the objective as a formal mathematics problem — set theory,
abstract algebra — whose solution encodes the harmful request, exploiting the
gap between a model's safety training on natural language and its willingness
to "solve maths". The attacker LLM produces the encoding, then an instruction
is appended asking for a concrete real-world translation of the answer. Each
candidate is gated by a refusal check and an "is this actually a math problem"
check, retried up to `max_retries` times, falling back to the original goal.

## Install

```bash
pip install superred-optimizer-mathprompt
```

## Usage

```python
from mathprompt_optimizer import MathPromptOptimizer

optimizer_factory = lambda: MathPromptOptimizer()
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `max_retries` | `5` | gated encoding attempts before falling back to the goal |

## Credits / upstream

Ported from [DeepTeam](https://github.com/confident-ai/deepteam) (Apache-2.0),
pinned at
[`dc148aad`](https://github.com/confident-ai/deepteam/tree/dc148aad62f71330cfec7121d6afb4c620dfa683/deepteam/attacks/single_turn/math_problem).
Technique: Bethany et al., *MathPrompt* (2024). Also shipped by Promptfoo
(`math-prompt`). Verify with `python scripts/sync_upstream.py --check`. See
[`NOTICE`](NOTICE) and [`LICENSES/NOTICE.md`](LICENSES/NOTICE.md).
