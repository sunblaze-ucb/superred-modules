# superred-optimizer-agentvigil-websentinel

An **AgentVigil** indirect prompt-injection optimizer for
[superred](https://superred.simonsure.com) agent targets.

> **Community port — not an official implementation.** This module is an
> unofficial re-implementation of AgentVigil for superred. It is not affiliated
> with, endorsed by, or maintained by the original authors. See
> [ASSUMPTIONS.md](ASSUMPTIONS.md) for every deliberate deviation from the
> paper and reference code.

AgentVigil is a black-box fuzzer for indirect prompt injection against LLM
agents. Starting from a corpus of injection seeds, it uses MCTS/UCT selection to
decide which seeds to reuse and which to explore, and an LLM mutator (expand,
shorten, rephrase, crossover, generate-similar) to evolve them — searching for
injections that land through content the agent reads (web pages, retrieved
documents, tool results, memory).

References: "AgentVigil: Generic Black-Box Red-teaming for Indirect Prompt
Injection against LLM Agents,"
[arXiv:2505.05849](https://arxiv.org/abs/2505.05849), and "WebSentinel: Detecting
and Localizing Prompt Injection Attacks for Web Agents,"
[arXiv:2602.03792](https://arxiv.org/abs/2602.03792). Official implementation:
[`wzunknown/agentvigil-websentinel`](https://github.com/wzunknown/agentvigil-websentinel).
See [ASSUMPTIONS.md](ASSUMPTIONS.md) for the full paper-to-code mapping and every
deliberate departure.

WebSentinel is the detector/localizer side of the upstream work; this module
implements the attack-generation side (the fuzzer described above) and does not
implement a separate WebSentinel defense.

## Install

```bash
pip install superred-optimizer-agentvigil-websentinel
```

## Usage

```python
from agentvigil_websentinel_optimizer import AgentVigilWebSentinelOptimizer

# Defaults: the official `new_seeds` web/content corpus, population_size=10,
# max_attempts=20 (mirrors the official fuzz-loop count).
optimizer = AgentVigilWebSentinelOptimizer()

# Also include the older official text-seed corpus.
optimizer = AgentVigilWebSentinelOptimizer(include_text_seeds=True)
```

This optimizer targets **agent-style** targets (AgentDojo, browser/RAG/memory/
MCP agents) that expose a content or tool-result surface. Wire it into a
`Controller` like any other superred optimizer — see the framework's
[architecture overview](https://superred.simonsure.com).

## License

This module is licensed under the [MIT License](LICENSE).

It bundles seed corpora and mutator prompt text extracted from the upstream
reference implementation,
[`wzunknown/agentvigil-websentinel`](https://github.com/wzunknown/agentvigil-websentinel)
(the mutator prompts from `mutate_prompts.py` and the seed corpora from
`new_seeds.py`/`seeds.py`). That repository publishes no license file; the
upstream authors granted the superred maintainers permission, in private
correspondence, to redistribute the ported material under MIT terms. See
[NOTICE](NOTICE) for the full attribution and [LICENSES/](LICENSES) for the
license text applied to the upstream material.
