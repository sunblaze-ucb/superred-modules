# superred-optimizer-codeattack

CodeAttack code-completion jailbreak optimizer for
[superred](https://superred.simonsure.com).

> **Community port — not an official implementation.** This module vendors the
> code templates from [renqibing/CodeAttack](https://github.com/renqibing/CodeAttack)
> (MIT) and reproduces its tokenisation. Not affiliated with the authors. See
> [ASSUMPTIONS.md](ASSUMPTIONS.md).

## What it does

CodeAttack hides the request inside a code-completion task: the goal is
tokenised into a Python data structure (a list, a stack/`deque`, or a string)
embedded in a code snippet, and the model is asked to "follow the comments and
complete the code" — decoding and acting on the smuggled request. One prompt
per run; no LLM calls.

## Install

```bash
pip install superred-optimizer-codeattack
```

## Usage

```python
from codeattack_optimizer import CodeAttackOptimizer

# python_stack (upstream's headline variant)
optimizer_factory = lambda: CodeAttackOptimizer()

# encode the goal as an ordered list, or a plain string
optimizer_factory = lambda: CodeAttackOptimizer(variant="python_list")
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `variant` | `python_stack` | `python_stack`, `python_list`, or `python_string` |

`python_list` appends the goal's words in order; `python_stack` appends them
reversed (a stack pops in order); `python_string` embeds the whole goal. The
rendered prompt is injected once into the first free-text non-`system_prompt`
surface, and the run ends after one attempt.

## Credits / upstream

Templates and tokenisation from
[CodeAttack](https://github.com/renqibing/CodeAttack) (MIT), pinned at
[`6777ed03`](https://github.com/renqibing/CodeAttack/tree/6777ed03b11567a91908f23bb8ccedca6103772c).
Paper: Ren et al., arXiv:2403.07865. `python_list` and `python_string` render
byte-identical to CodeAttack's own shipped prompts (verify with
`python scripts/sync_upstream.py --check`). See [`NOTICE`](NOTICE) and
[`LICENSES/NOTICE.md`](LICENSES/NOTICE.md).
