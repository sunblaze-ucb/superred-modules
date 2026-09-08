# superred-optimizer-bad-likert-judge

Bad Likert Judge multi-turn jailbreak optimizer for
[superred](https://superred.simonsure.com), ported from DeepTeam.

> **Community port — not an official implementation.** This module vendors
> DeepTeam's Bad Likert Judge prompt templates
> ([confident-ai/deepteam](https://github.com/confident-ai/deepteam),
> Apache-2.0) byte-identically. Not affiliated with Confident AI or Palo Alto
> Networks. See [ASSUMPTIONS.md](ASSUMPTIONS.md).

## What it does

The attacker's own LLM is cast as a harmfulness **evaluator**. Asked to
illustrate a Likert scale for a category, it produces a benign score-1 example
and a maximally harmful score-3 one — then that score-3 example is rewritten
into the next turn sent to the target. Because the model is "just scoring
examples", the harmful content arrives as evaluation material rather than a
request.

Each turn runs upstream's chain on the attacker LLM:

1. `likert_generate_examples` — score-1 / score-3 pair for the category
2. `likert_refine_score_3` — sharpen the score-3 example (optional)
3. `likert_generate_attack_from_example` — rewrite it as the next turn
4. `non_refusal` — if the attacker LLM balked, spend a backtrack, send nothing

Only step 3's output reaches the target. The conversation is pinned to one
surface, so later turns continue it rather than restarting.

## Install

```bash
pip install superred-optimizer-bad-likert-judge
```

## Usage

```python
from bad_likert_judge_optimizer import BadLikertJudgeOptimizer

optimizer_factory = lambda: BadLikertJudgeOptimizer()

# a specific guideline set, fewer turns, skipping the refinement step
optimizer_factory = lambda: BadLikertJudgeOptimizer(
    category="cybercrime", num_turns=3, enable_refinement=False
)
```

Defaults are upstream's, taken from `BadLikertJudge.__init__`:

| Argument | Default | Meaning |
| --- | --- | --- |
| `category` | `bias` | one of the 58 categories upstream ships guidelines for |
| `num_turns` | `5` | turns to send |
| `enable_refinement` | `True` | run the `likert_refine_score_3` step |
| `max_backtracks` | `3` | attacker-LLM refusals/unusable replies to absorb before stopping |

At these defaults a turn costs four attacker-LLM calls (examples, refine,
attack, non-refusal check), so a full five-turn run is ~20 calls before the
target is touched. Pass `enable_refinement=False` and a lower `num_turns` to
trade attack strength for budget.

`BadLikertJudgeOptimizer.SUPPORTED_CATEGORIES` lists every category.

## Credits / upstream

Templates from [DeepTeam](https://github.com/confident-ai/deepteam)
(Apache-2.0), pinned at
[`dc148aad`](https://github.com/confident-ai/deepteam/tree/dc148aad62f71330cfec7121d6afb4c620dfa683/deepteam/attacks/multi_turn/bad_likert_judge).
Technique: Palo Alto Networks Unit 42, "Bad Likert Judge". Verify the vendored
files with `python scripts/sync_upstream.py --check`. See [`NOTICE`](NOTICE)
and [`LICENSES/NOTICE.md`](LICENSES/NOTICE.md).
