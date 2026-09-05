# superred-claim-jailbreakbench

The [JailbreakBench](https://jailbreakbench.github.io/) (JBB-Behaviors)
red-teaming benchmark as a superred `SecurityClaim`.

> **Community port — not an official implementation.** This module vendors the
> JBB-Behaviors harmful split and JailbreakBench's canonical judge prompt
> (both MIT). It is not affiliated with, endorsed by, or maintained by the
> JailbreakBench team. See [ASSUMPTIONS.md](ASSUMPTIONS.md) for provenance and
> deviations.

## What it is

100 harmful behaviours across 10 OpenAI-usage-policy categories, each judged by
JailbreakBench's canonical `Llama3JailbreakJudge` prompt — an LLM-as-judge that
answers `safe` / `unsafe`. One behaviour becomes one superred `Task` bound to
`ChatbotTarget`; a run is a success when the judge rules the response `unsafe`.

## Install

```bash
pip install superred-claim-jailbreakbench
```

## Usage

```python
from jailbreakbench_claim import jailbreakbench_claim

# all 100 behaviours, judged by a model you choose
claim = jailbreakbench_claim(judge_model="openai/gpt-4o-mini")

# a filtered subset
claim = jailbreakbench_claim(
    judge_model="openai/gpt-4o-mini",
    categories=["Malware/Hacking"],
    limit=10,
)
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `judge_model` | `openai/gpt-4o-mini` | LiteLLM judge model id |
| `judge_api_base` / `judge_api_key` | env `LITELLM_API_BASE` / `LITELLM_API_KEY` | judge provider |
| `judge_max_attempts` | `3` | judge calls per evaluation |
| `categories` / `sources` / `limit` | none | restrict the behaviour set |

The judge prompt and its `safe`/`unsafe` decision rule are JailbreakBench's,
copied byte-for-byte; the hosting model is your choice. JailbreakBench's own
canonical judge is Llama-3-70B — see ASSUMPTIONS.md on judge-model choice and
comparability.

A judge that fails to answer raises rather than scoring 0, so a broken judge
surfaces as a task error, never as a target that looks robust.

## Credits / upstream

Data and judge from [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench)
(MIT) and the [JBB-Behaviors dataset](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors)
(MIT), code pinned at commit
[`23dbdf6b`](https://github.com/JailbreakBench/jailbreakbench/blob/23dbdf6b19650521604456229bc1d9c4156c85c1/src/jailbreakbench/classifier.py).
Verify the vendored data and prompt with `python scripts/sync_upstream.py --check`.
See [`NOTICE`](NOTICE) and [`LICENSES/NOTICE.md`](LICENSES/NOTICE.md).
