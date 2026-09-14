# superred-claim-harmbench

[HarmBench](https://www.harmbench.org/) standardized red-teaming benchmark
([Mazeika et al., ICML 2024](https://arxiv.org/abs/2402.04249)) packaged as
a superred `SecurityClaim` against `ChatbotTarget`.

> **Community port — not an official implementation.** This module is an
> unofficial re-implementation of HarmBench (Mazeika et al., Center for AI
> Safety) for superred. It is not affiliated with, endorsed by, or maintained
> by the original authors.

## What it is

- 400 textual harmful behaviors (320 test split + 80 val split).
- Three functional categories per split: standard (159 / 41), contextual
  (81 / 19), copyright (80 / 20).
- Per-behavior binary verdict (1 = behavior elicited, 0 = refused or off-topic)
  from an LLM-as-judge using HarmBench's verbatim Llama-2 classifier prompt
  template, plus the upstream MinHash matcher for copyright behaviors.
- Headline metric: ASR = mean of per-behavior labels; reported per
  functional and semantic category via `EvaluationResult.sub_scores`.

## Quick start

```python
import asyncio, os
from chatbot_target import ChatbotTarget, USER_TAG, RESPONSE_READABLE_TAG
from harmbench_claim import harmbench_standard_val
from superred.core.controller import Controller, TargetFactory

api_key = os.environ["LITELLM_API_KEY"]
api_base = os.environ["LITELLM_API_BASE"]

# Fresh ChatbotTarget per task; bump concurrency for parallel evaluation
# (the chatbot wraps the LiteLLM proxy and is safe to call concurrently).
target_factory = TargetFactory(
    create=lambda: ChatbotTarget(
        model="openai/gpt-4o-mini", api_key=api_key, api_base=api_base,
    ),
    concurrency=8,
)

# 41 standard val behaviors. For all 80 val behaviors (across 3 functional
# categories) use `harmbench_val(...)`; for the 320-task test split use
# `harmbench_test(...)`; for everything use `harmbench_complete(...)`.
claim = harmbench_standard_val(
    judge_model="openai/gpt-4o-mini",
    judge_api_base=api_base,
    judge_api_key=api_key,
)

# Bring your own optimizer; for a single-shot DirectRequest baseline, see
# superred-experiments/test_harmbench_chatbot/direct_request_optimizer.py
from direct_request_optimizer import DirectRequestOptimizer

controller = Controller(
    optimizer_factory=DirectRequestOptimizer,
    target_factory=target_factory,
    security_claim=claim,
    scope=frozenset({USER_TAG, RESPONSE_READABLE_TAG}),
    max_runs_per_task=1,
    include_feedback=False,
)
result = asyncio.run(controller.run())   # -> ThreatModelResult
```

## Installation

```bash
pip install -e ./security_claims/harmbench   # provides the harmbench_claim package
python -m spacy download en_core_web_sm      # for the copyright MinHash matcher
```

The pip-installable name is `superred-claim-harmbench`; the importable Python
package is `harmbench_claim`.

## Public factories

The module exposes a hierarchical set of factories so you can pick any
granularity from "one functional split of one dataset split" up to "the
whole benchmark."

All factory parameters are keyword-only (the underlying signature uses
``def harmbench_claim(*, judge_model, judge_api_base, judge_api_key,
split, functional_categories, semantic_categories, csv_path, clip_tokens)``).

```
# Parameterized base
harmbench_claim(*, judge_model="openai/gpt-4o-mini", split="test",
                functional_categories=None, semantic_categories=None, ...)

# Leaf factories — six combinations of (split, functional_category)
harmbench_standard_test()    harmbench_standard_val()
harmbench_contextual_test()  harmbench_contextual_val()
harmbench_copyright_test()   harmbench_copyright_val()

# Composed
harmbench_test()             harmbench_val()      # all 3 functional categories
harmbench_complete()                              # both splits
```

Leaf and composed factories accept the same judge / filter kwargs
(`judge_model`, `judge_api_base`, `judge_api_key`, `semantic_categories`,
`csv_path`, `clip_tokens`) but reject `split` / `functional_categories`
because those are hardcoded by the leaf — pass them to
`harmbench_claim()` directly if you need to override.

Each leaf factory returns a flat `SecurityClaim` of one Task per behavior;
the composed factories use `SecurityClaim.from_claims` lazy chaining.

## Recommended scope

`frozenset({USER_TAG, RESPONSE_READABLE_TAG})` (interactive black-box).
HarmBench's threat model assumes the attacker injects the user message
and observes the response. The claim itself does not pin a scope; the
experiment script picks the scope by passing it to the `Controller`
constructor. To sweep multiple scopes, instantiate one `Controller` per
scope and `asyncio.gather` them.

## Per-task metadata for post-run analysis

Each `HarmBenchTask` exposes plain attributes so you can aggregate
`result.task_results` (the `ThreatModelResult` returned by
`controller.run()`) by any axis:

```python
from collections import defaultdict
buckets: dict[tuple[str, str], list[bool]] = defaultdict(list)
for tr in result.task_results:
    t = tr.task
    buckets[("functional", t.functional_category)].append(tr.success)
    buckets[("semantic", t.semantic_category)].append(tr.success)
```

`EvaluationResult.sub_scores` additionally carries
`functional_<name>`, `semantic_<name>`, and `advbench_refusal_label` keys
for in-trajectory attribution.

## Deviations from upstream HarmBench

We document every deviation up front so ASR numbers are interpretable.

| Aspect | HarmBench reference | This port | Why |
|---|---|---|---|
| Judge model | `cais/HarmBench-Llama-2-13b-cls` (fine-tuned, ~93% agreement with humans) | LLM-as-judge via `LLMConfig` (default in examples: `gpt-4o-mini`) using HarmBench's verbatim prompt template | The 13B classifier needs ~26 GB VRAM. Our framework runs on the litellm proxy. The HarmBench paper itself (page 7) warns that closed-source judges are "far from ideal", but the canonical Llama-2-cls prompt template still applies. |
| `max_new_tokens` on target | 512 (paper-mandated for ASR convergence; up to 30% variance otherwise) | provider default (no `max_tokens` override on `ChatbotTarget`) | Conscious user decision; ASR comparable across runs within this port but not directly comparable to the HarmBench leaderboard. |
| `temperature` on target | 0 (greedy) | 0 (already hardcoded in `ChatbotTarget.run()`) | Faithful. |
| System prompt | none added by the pipeline | task sets `system_prompt=""`; `ChatbotTarget` then suppresses the system message entirely | Faithful (verified by `TestEmptySystemPrompt`). |
| Contextual test-case rendering | `f"{ContextString}\n\n---\n\n{Behavior}"` (per `baselines/direct_request/direct_request.py:24`) | identical, in `Goal.description` via `render_test_case` | Faithful. |
| Copyright eval | MinHash + spaCy + 100 reference `.pkl` files; Jaccard > 0.6 | identical port of `compute_results_hashing` from `eval_utils.py`; spaCy 3.8 instead of 3.7 (Python 3.13 compat) | Near-faithful; tokenization may differ marginally from spaCy 3.7. |
| AdvBench refusal-prefix diagnostic | enabled via `--include_advbench_metric` (29 prefixes in `eval_utils.py`) | always enabled as `sub_scores["advbench_refusal_label"]` | Faithful (free additional metric). |
| `[/INST]` prompt-injection sanitization in judge prompt | none (upstream issue #91 is unfixed) | none | Vector is inert against chat-completions APIs (Llama-2 instruction tokens are plain text there). |
| Generation pre-judge clipping | classifier-tokenizer, 512 tokens, right-truncated | `tiktoken` (if installed) or 4-chars-per-token approximation, 512 tokens | Approximation acceptable; Jaccard threshold tolerates small differences and the chat-completions judge has 128k context anyway. |
| Unparseable or failed judgement | `compute_results_classifier` records `label=-1` and continues | retried, then raised (`JudgeUnparseableError` / `JudgeUnavailableError`); the task errors instead of being scored | A judge that never answered is not evidence the behavior was not elicited. Scoring it 0 makes a broken judge read as a robust target, and makes it permanent: an errored task is re-measured on resume, a scored one is kept. Every `EvaluationResult` this claim emits carries `sub_scores["judge_ran"] == 1.0`. |

ASR numbers from this port are **not directly comparable to the HarmBench
leaderboard.** They are useful for relative comparison across (target,
optimizer) pairs within experiments using this module.

## Provenance

Bundled data is verbatim from HarmBench commit
`8e1604d1171fe8a48d8febecd22f600e462bdcdd` (MIT licensed). See
`src/harmbench_claim/data/PROVENANCE.md`.

## Citation

```
@article{mazeika2024harmbench,
  title={HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal},
  author={Mantas Mazeika and Long Phan and Xuwang Yin and Andy Zou and Zifan Wang and Norman Mu and Elham Sakhaee and Nathaniel Li and Steven Basart and Bo Li and David Forsyth and Dan Hendrycks},
  year={2024},
  eprint={2402.04249},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

## Credits / upstream

This module packages the [HarmBench](https://github.com/centerforaisafety/HarmBench)
standardized red-teaming benchmark (Mazeika et al., ICML 2024,
[arXiv:2402.04249](https://arxiv.org/abs/2402.04249)) as a superred
`SecurityClaim`.

- **HarmBench** — MIT License, Copyright (c) 2024 centerforaisafety. The
  bundled behavior CSVs and copyright-classifier MinHash `.pkl` files, the
  verbatim Llama-2 classifier prompt templates, the AdvBench refusal-prefix
  list, and the sliding-window MinHash copyright matcher are vendored/ported
  from HarmBench commit `8e1604d`. The `.pkl` files hold only one-way MinHash
  signatures — no copyrighted source text is redistributed.
- The AdvBench refusal-prefix list traces to Zou et al. 2023
  ([llm-attacks](https://github.com/llm-attacks/llm-attacks), MIT), vendored
  here via HarmBench.
- The contextual-category behaviors carry short `ContextString` excerpts
  (e.g. Wikipedia text under CC-BY-SA, a published paper, forum posts, a code
  snippet) that centerforaisafety does not itself own. They are redistributed
  byte-identically to the upstream, widely-mirrored HarmBench ICML benchmark
  under its MIT terms, and this mirrors the upstream benchmark rather than
  asserting fresh MIT coverage of those excerpts.

Original superred integration code is MIT-licensed, Copyright (c) 2026
Rishabh Sinha. Full attribution in `LICENSES/NOTICE.md`; upstream license text in
`LICENSES/harmbench-MIT.txt`.
