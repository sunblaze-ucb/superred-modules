# superred-optimizer-renellm

ReNeLLM generalized nested-jailbreak optimizer for
[superred](https://superred.simonsure.com).

Ports **ReNeLLM** from
[NJUNLP/ReNeLLM](https://github.com/NJUNLP/ReNeLLM) (pinned at commit
`a61c39e`), the official implementation of:

> Peng Ding, Jun Kuang, Dan Ma, Xuezhi Cao, Yunsen Xian, Jiajun Chen, and
> Shujian Huang. "A Wolf in Sheep's Clothing: Generalized Nested Jailbreak
> Prompts can Fool Large Language Models Easily." NAACL 2024.
> [arXiv:2311.08268](https://arxiv.org/abs/2311.08268).

## Technique

ReNeLLM generalizes prompt jailbreaks into two auxiliary-LLM-driven stages:

1. **Prompt rewriting** — a random count and order of six semantics-preserving
   rewrite operations (paraphrase-shorten, misspell sensitive words, reorder
   words, insert meaningless characters, partial translation, and style/slang
   change) are applied to the goal. A binary LLM judge confirms the rewrite is
   still harmful; if not, it is retried from the original goal.
2. **Scenario nesting** — the rewritten goal is embedded into one of three
   benign carriers (code completion, table filling, or text continuation).

The nested prompt is sent to the model under attack, and its reply is scored by
the same judge. The loop repeats up to a budget until the reply is judged
harmful.

## Mapping to superred

- **One superred run == one ReNeLLM outer iteration.** On `RunStart` the per-run
  state is re-armed and the rewritten+nested prompt is produced **pre-send**
  (superred has no mid-conversation rewind, so all refinement happens before the
  send).
- The nested prompt is injected on the first eligible free-text **user** surface
  at `ControllablePreCall`. The model under attack is the **target itself**, not
  an LLM this optimizer constructs.
- The reply is read from `ControllablePostCall` (with a trajectory-observable
  fallback) and scored at `RunEnd` to decide stop/continue, up to `iter_max`
  runs.
- Every **auxiliary** LLM call (the six rewrite operations and the judge) runs
  the byte-identical vendored upstream code, routed to the constrained
  `self.llm` (superred's `LLMClient`). No sampling temperature is ever set.

The upstream rewrite/nest/judge helpers are vendored byte-for-byte under
`src/renellm_optimizer/_vendor/renellm/utils/` and executed; only upstream's
openai/anthropic SDK completion helper is replaced (by
`src/renellm_optimizer/_shim.py`). See `ASSUMPTIONS.md` for every deviation and
`NOTICE` for attribution.

## Usage

```python
from renellm_optimizer import ReNeLLMOptimizer

optimizer = ReNeLLMOptimizer(
    iter_max=20,               # max runs (outer iterations); upstream default
    max_rewrite_attempts=20,   # cap on the rewrite-retry loop
    seed=None,                 # reproducible operation/scenario selection
)
```

The optimizer is instantiated by the superred controller, which supplies the
goal, controllables, observables, and the constrained `LLMClient`.

## Development

```shell
# offline test battery
PYTHONPATH=src python -m pytest -q tests

# verify vendored files are byte-identical to the pinned upstream commit
python scripts/sync_upstream.py --check
```

## License

MIT (see `LICENSE`). Vendored ReNeLLM code is MIT (Copyright (c) 2024 NJUNLP);
its text is preserved in `LICENSES/ReNeLLM-MIT.txt`.
