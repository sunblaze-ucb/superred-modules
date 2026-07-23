# superred-optimizer-agentvigil-websentinel

An **AgentVigil** indirect prompt-injection optimizer for
[superred](https://superred.simonsure.com) agent targets.

AgentVigil is a black-box fuzzer for indirect prompt injection against LLM
agents. Starting from a corpus of injection seeds, it uses MCTS/UCT selection to
decide which seeds to reuse and which to explore, and an LLM mutator (expand,
shorten, rephrase, crossover, generate-similar) to evolve them — searching for
injections that land through content the agent reads (web pages, retrieved
documents, tool results, memory).

Reference: "AgentVigil/WebSentinel: Generic Black-Box Red-teaming for Indirect
Prompt Injection against LLM Agents,"
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

**No license is granted yet** (see [LICENSE](LICENSE)). The upstream reference
implementation,
[`wzunknown/agentvigil-websentinel`](https://github.com/wzunknown/agentvigil-websentinel),
publishes **no license file**. This module reproduces literals it references
(the mutator prompts from `mutate_prompts.py` and the seed corpora from
`new_seeds.py`/`seeds.py`). Absent an express upstream license grant, that
material is presumed all-rights-reserved. Written permission to use and
redistribute it **has been sought from the upstream author but not yet
received**; pending that permission (or a subsequently published upstream
license), this module is withheld and is **not** released under the MIT License
that applies to other superred optimizers. Once permission is obtained or the
upstream-derived literals are fully replaced with original content, a definitive
license will be applied here.
