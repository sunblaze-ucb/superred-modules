# superred-optimizer-agentbreaker

Agent Breaker agentic tool-exploitation optimizer for
[superred](https://superred.simonsure.com), ported from NVIDIA garak's Agent
Breaker probe.

> **Community port — not an official implementation.** This module vendors the
> prompt catalogue from [NVIDIA garak](https://github.com/NVIDIA/garak)
> (Apache-2.0). It is not affiliated with, endorsed by, or maintained by
> NVIDIA. See [ASSUMPTIONS.md](ASSUMPTIONS.md) for provenance and deviations.

## What it does

Agent Breaker attacks agents that use tools. It reads the agent's tool
catalogue, has a red-team LLM analyse each tool for weaknesses and draft
attack prompts, then sends those prompts one per run — learning from each
response to generate fresh exploits once the seed prompts are spent, and
moving to the next tool after a few tries.

Unlike the model-level ports in this repo, this is a genuine LLM-driven,
feedback-using optimizer: it spends attacker inference budget.

## Install

```bash
pip install superred-optimizer-agentbreaker
```

## Usage

```python
from agentbreaker_optimizer import AgentBreakerOptimizer

optimizer_factory = lambda: AgentBreakerOptimizer()

# bound the cost: at most 3 attempts on each of the 2 highest-priority tools
optimizer_factory = lambda: AgentBreakerOptimizer(
    max_attempts_per_tool=3, max_tools=2
)
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `max_attempts_per_tool` | `5` | attempts before moving to the next tool (upstream default) |
| `max_tools` | `None` | cap on tools attacked, bounding LLM cost; `None` attacks all |

The optimizer reads the agent's tools and purpose from the target's
**observables** (`tool_catalog_listing`, `tool_list`, `system_prompt`, …), runs
one analysis call, then injects one wrapped attack per run into the first
free-text non-`system_prompt` surface. It reports `done` once every analysed
tool has been attacked, and needs an agentic target that exposes a tool
catalogue — against a bare chatbot it finds no tools and declines.

## Credits / upstream

Derived from NVIDIA garak, Apache-2.0, pinned at commit
[`2233d51b`](https://github.com/NVIDIA/garak/blob/2233d51bedf6bfb634872ef9efbcd94eae38cc1a/garak/probes/agent_breaker.py).
Verify the vendored prompts with `python scripts/sync_upstream.py --check`.
See [`NOTICE`](NOTICE) and [`LICENSES/NOTICE.md`](LICENSES/NOTICE.md).
