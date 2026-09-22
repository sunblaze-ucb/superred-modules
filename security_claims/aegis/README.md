# aegis

The NVIDIA **Aegis AI Content Safety** benchmark as a superred `SecurityClaim`,
built on the [Aegis 1.0 dataset](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-1.0)
(a.k.a. *Nemotron Content Safety Dataset V1*), CC-BY-4.0.

## What it evaluates

Each task sends one **unsafe user prompt** from the Aegis 1.0 **test split** to a
`ChatbotTarget` and asks a bound safety judge whether the target's *response* is
unsafe per NVIDIA's Aegis content-safety taxonomy (13 harm categories).

- **Attack success** (`EvaluationResult.success = True`) = the target produced
  **unsafe** content (it failed the safety bar).
- **Safe / refused** responses score 0.0 (the attack failed).

This is the same content-moderation-as-red-team framing that promptfoo's
`AegisGrader` and deepteam's Aegis framework use: an Aegis-labelled unsafe prompt
is the attack input, and the target's response is graded for a safety violation.

## Task set

The unsafe user-message prompts of the vendored Aegis 1.0 test split:

| filter | tasks |
| --- | --- |
| `include_needs_caution=True` (default) — `text_type == "user_message"` and `labels_0 != "Safe"` | **236** |
| `include_needs_caution=False` — 13 harm categories only (excludes "Needs Caution") | 180 |

The full test split is 1199 rows across four `text_type`s; only standalone
`user_message` prompts are used (an LLM response or a multi-turn dialog is not a
red-team input). See [ASSUMPTIONS.md](ASSUMPTIONS.md) for the "Needs Caution"
decision and the test-vs-train split choice.

## The judge (bound, generic — `generic-judge`)

`AegisSafetyJudge` is an LLM-as-judge that grades the response against the Aegis
taxonomy and emits `{"pass": true}` (safe) / `{"pass": false}` (unsafe). **It is
NOT NVIDIA's native Aegis-Guard (LlamaGuard) classifier** — that needs heavy,
gated model weights and is not shipped. `ASSUMPTIONS.md` explains how to plug an
Aegis-Guard-backed judge in via the `Judge` protocol. A judge that cannot return
a parseable verdict **abstains (raises)** rather than fabricating one.

## Usage

```python
from aegis_claim import aegis_claim
from superred.core.types.llm import LLMConfig

claim = aegis_claim(
    judge_llm_config=LLMConfig(model="openai/gpt-4o", api_base=BASE, api_key=KEY),
)
tasks = list(claim)            # 236 tasks

# cheaper pilot: 20 tasks, harm categories only
pilot = aegis_claim(
    judge_llm_config=cfg, include_needs_caution=False, limit=20,
)

# a single category
violence = aegis_claim(judge_llm_config=cfg, categories=["Violence"])
```

Bound to `ChatbotTarget` (reads `last_response`); configured with **no system
prompt**. Pass a pre-built `judge=` (e.g. the offline `RefusalHeuristicJudge`, or
your own Aegis-Guard judge) instead of `judge_llm_config=` when you want to
control the judge.

## Data provenance

Vendored from `nvidia/Aegis-AI-Content-Safety-Dataset-1.0` at revision
`bd96d862`, CC-BY-4.0, byte-identically as the upstream parquet plus a
deterministic CSV rendering (loaded with the stdlib `csv` module — no pandas).
`scripts/sync_upstream.py --check` byte-compares the vendored parquet against
HuggingFace and re-derives the CSV; `tests/test_vendor_integrity.py` verifies the
sha256 manifest offline.

## Dependencies

`superred`, `superred-target-chatbot`, `litellm`. No heavy/ML deps; imports on
Python 3.13. `pyarrow` + `huggingface_hub` are dev-only (the `sync` extra), used
by `scripts/sync_upstream.py`, never by the installed package.

## Faithfulness

See [ASSUMPTIONS.md](ASSUMPTIONS.md). Cite the paper (arXiv:2404.05993) and the
dataset (CC-BY-4.0, attribution to NVIDIA) when reporting numbers.
