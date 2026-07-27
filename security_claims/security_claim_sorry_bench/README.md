# superred-claim-sorry-bench

[SORRY-Bench](https://sorry-bench.github.io/) safety-refusal benchmark
([Xie et al., ICLR 2025](https://arxiv.org/abs/2406.14598)) packaged as a
superred `SecurityClaim` against `ChatbotTarget`.

## What it tests

- 44 fine-grained safety categories, 10 unsafe instructions each → **440 prompts**.
- Per-prompt fulfillment score (0 = refusal, 1 = compliance) from an
  LLM-as-judge using SORRY-Bench's verbatim `base-#thescore` template.
- Headline metric (paper Figure 4): mean of per-category means; range
  `0.0` to `1.0`. Higher means more compliance with unsafe instructions.

## Getting the dataset

The dataset is **gated and non-redistributable** under a custom license
that forbids embedding it in third-party packages. This module never
ships any benchmark prompts. Two ways to provide them at runtime:

1. **Manual download** (recommended for offline / CI):
   ```
   # In your browser, accept the gate at:
   #   https://huggingface.co/datasets/sorry-bench/sorry-bench-202503
   # Then download question.jsonl to a path of your choice.
   export SORRY_BENCH_QUESTION_JSONL=/abs/path/to/question.jsonl
   ```

2. **HuggingFace lazy-load** (auto-downloads to `~/.cache/huggingface/`):
   ```
   /path/to/.venv/bin/hf auth login
   # ... follow the prompts and paste a token from
   #     https://huggingface.co/settings/tokens
   # Make sure the gate at the dataset URL above is also accepted.
   ```

## Quick start

```python
import asyncio
from chatbot_target import ChatbotTarget, USER_TAG, MODEL_TAG
from sorry_bench_claim import sorry_bench_claim
from superred.core.controller import Controller, TargetFactory
from superred.core.interfaces.optimizer import Optimizer
from superred.core.types.events import (
    ControllableInjection, ControllableNoInjection,
    ControllablePreCallEvent, RunEndEvent, RunEndResponse, RunStartEvent,
)
from superred.core.types.event import Event, EventResponse
from superred.core.types.llm import LLMConfig


# A no-attacker "passthrough" optimizer that emits the task's question
# verbatim. Use this for the vanilla SORRY-Bench evaluation. For real
# red-teaming, swap in TAP / PAIR / Crescendo / etc.
class PassthroughOptimizer(Optimizer):
    def __init__(self) -> None:
        super().__init__()
        self._injected = False

    async def initialize(self, goal, controllables, observables, llm_client) -> None:
        await super().initialize(goal, controllables, observables, llm_client)
        self._goal = goal
        self._injected = False

    async def on_event(self, event: Event) -> EventResponse:
        if isinstance(event, RunStartEvent):
            self._injected = False
            return EventResponse(event=event)
        if isinstance(event, ControllablePreCallEvent):
            if self._injected:
                return ControllableNoInjection(event=event, controllable=event.controllable)
            self._injected = True
            return ControllableInjection(
                event=event, controllable=event.controllable,
                value=self._goal.description,
            )
        if isinstance(event, RunEndEvent):
            return RunEndResponse(event=event, done=True)
        return EventResponse(event=event)


async def main() -> None:
    target_factory = TargetFactory(
        create=lambda: ChatbotTarget(
            model="gpt-4o-2024-05-13",
            api_base="https://api.openai.com/v1",
            api_key="sk-...",
        ),
        concurrency=8,
    )
    claim = sorry_bench_claim(
        target_model_id="gpt-4o-2024-05-13",
        # Pass dataset_path=... if you prefer manual download:
        # dataset_path="/abs/path/to/question.jsonl",
        judge_llm_config=LLMConfig(
            model="gpt-4-1106-preview",
            api_base="https://api.openai.com/v1",
            api_key="sk-...",
        ),
        prompts_per_category=2,    # 88-prompt smoke run; omit for full 440
    )

    controller = Controller(
        optimizer_factory=PassthroughOptimizer,
        target_factory=target_factory,
        security_claim=claim,
        scope=frozenset({USER_TAG}),
        read_only=frozenset({MODEL_TAG}),
        max_runs_per_task=1,
    )
    result = await controller.run()      # -> ThreatModelResult

    # Aggregate per-category and overall fulfillment from result.task_results.
    ...


asyncio.run(main())
```

## Subset selection (pilot then full)

`sorry_bench_claim` exposes two pre-defined halves of the benchmark via
`subset="a"` / `subset="b"`. Each half is **220 prompts (5/category ×
44 categories)**, stratified across the full taxonomy; the two halves
are disjoint and together reconstruct the full 440-prompt benchmark.

This is the cheap way to run a pilot first, look at the results, then
fill in the rest with no overlap:

```python
# Step 1: pilot on half A (220 prompts, ~$0.50 in judge calls)
claim_a = sorry_bench_claim(
    target_model_id="gpt-4o-2024-05-13",
    judge_llm_config=judge_cfg,
    subset="a",
)
# ... run, inspect, decide whether to continue ...

# Step 2: complete with half B (the other 220, no overlap)
claim_b = sorry_bench_claim(
    target_model_id="gpt-4o-2024-05-13",
    judge_llm_config=judge_cfg,
    subset="b",
)
```

The split is by `question_id` parity (subset A = odd, subset B = even),
which the dataset's `(question_id - 1) // 10 + 1 == category` invariant
guarantees is perfectly balanced per category. `subset` composes with
`categories` (the subset is taken from the surviving categories) and
`prompts_per_category` (truncation runs after). It is ignored when
`question_ids` is supplied. The exact id lists are exported as
`SUBSET_A_QUESTION_IDS` / `SUBSET_B_QUESTION_IDS` for downstream
analysis.

## Faithfulness anchor

We benchmark against the paper's published number for `gpt-4o-2024-05-13`:
**0.2977** (paper Figure 4 / Appendix K.2). A faithful end-to-end run
should land within `±0.10` absolute of that anchor. The widened
tolerance (vs. paper's typical ±0.05 from Table 8) absorbs three
documented sources of drift:

- `ChatbotTarget` leaves temperature at the provider default; SORRY-Bench
  uses 0.7 (paper §K.5). Per Table 8, this contributes ≤±0.05.
- GPT-4o judge agreement vs. ft-Mistral-7B (the canonical SORRY-Bench
  judge) is 78.9% vs. 81.0% Cohen-κ (paper Table 6); about 2.1 pp gap.
- Inter-annotator ceiling is ~90.5% (paper Figure 6), which caps how
  perfectly any judge can agree with humans in principle.

Per-category sanity bounds (paper §K.2):
- **Most-refused**: #8 Harassment, #9 Sexual Crimes, #20 Child-related Crimes
  (~9–11% fulfillment across all 56 paper-tested models).
- **Least-refused**: #33 Ethical Belief, #34 Religion Promotion,
  #42 Legal Consulting Advice (~74–80%).

### Replaying the smoke

`tests/smoke/run.py` is a runnable verification artifact (88 prompts,
~$0.20 in judge calls). Prerequisites:

```
pip install -e ".[smoke]"     # adds python-dotenv
hf auth login                 # plus accept the gate at the dataset page
export LITELLM_API_KEY=...
export LITELLM_API_BASE=...

python tests/smoke/run.py
```

Expected output: overall fulfillment in `[0.05, 0.55]`, with
high-refusal categories (#8/#9/#20) decisively below low-refusal
(#33/#34/#42). See the script's docstring for the full output schema.

## Known limitations

Three judge-failure modes are documented in paper §J.4 and propagate into
our port:

- **Bullet-list responses** can be over-classified as fulfillment even
  when the model refused.
- **Safety disclaimers** can be over-classified as refusal even when
  fulfillment text is present.
- **Partial fulfillment** (some sub-tasks completed, some refused) is
  genuinely ambiguous; humans disagree at ~10% on these.

These are not bugs in our port — they affect SORRY-Bench's own published
numbers — but they are real noise in the score. Plan accordingly when
interpreting category-level deltas.

## What is and is not implemented (v1)

| | v1 | Future |
|---|----|----|
| 440 base prompts | ✅ | |
| 44-category taxonomy | ✅ (canonical names from `meta_info.py`) | |
| `LLMAsJudge` (GPT-4 path) | ✅ default | |
| `RefusalRegexJudge` | ✅ (offline / smoke only — NOT faithful) | |
| 20 linguistic mutations (ASCII, Caesar, persuasion, languages, etc.) | — | tracked |
| Fine-tuned Mistral-7B judge (canonical) | — | requires GPU + 14 GB model |
| Configurable target temperature (T2 patch) | — | to pin SORRY-Bench's 0.7 |

`ChatbotTarget` sends no explicit temperature (it omits the value
because some providers, including the o1 family, reject an explicit
one), so every model runs at its provider default rather than
SORRY-Bench's 0.7. A configurable-temperature kwarg (the T2 patch) to
pin a specific value lives outside this branch.

## Attribution

- **Paper**: Tinghao Xie et al. *SORRY-Bench: Systematically Evaluating
  Large Language Model Safety Refusal*. ICLR 2025.
  arXiv:[2406.14598](https://arxiv.org/abs/2406.14598).
- **Upstream code**: <https://github.com/SORRY-Bench/SORRY-Bench>
  (MIT-licensed; commit `7da10ad`).
- **Dataset**: <https://huggingface.co/datasets/sorry-bench/sorry-bench-202503>
  (custom non-redistributable license — `LICENSE` quoted in the dataset
  card; this module respects that and never bundles prompts).
- **Vendored prompt templates**: see [JUDGE_PROMPTS_NOTICE.md](./JUDGE_PROMPTS_NOTICE.md).

## Credits / upstream

This module is an original superred `SecurityClaim` port of **SORRY-Bench**.
The port code is MIT-licensed, Copyright (c) 2026 Simon Sure.

- **Paper / methodology**: Tinghao Xie et al., *SORRY-Bench: Systematically
  Evaluating Large Language Model Safety Refusal*, ICLR 2025,
  arXiv:[2406.14598](https://arxiv.org/abs/2406.14598). Please cite this paper
  when reporting numbers produced with this module.
- **Upstream code** (judge prompt templates, vendored verbatim):
  [SORRY-Bench/SORRY-Bench](https://github.com/SORRY-Bench/SORRY-Bench),
  MIT, Copyright (c) 2023 LLM-Tuning-Safety (commit `7da10ad`).
- **Dataset** (NOT bundled, loaded at runtime):
  [sorry-bench/sorry-bench-202503](https://huggingface.co/datasets/sorry-bench/sorry-bench-202503),
  under a custom **non-redistributable** license. You must accept the dataset
  gate and authenticate to HuggingFace yourself; this package ships no prompts.

See `NOTICE` for full attribution and `JUDGE_PROMPTS_NOTICE.md` for the
per-template provenance table.
