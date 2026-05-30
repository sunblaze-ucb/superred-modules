# agentdojo-target

A composite [superred](../../../superred) Target that ports the [AgentDojo benchmark environment](https://github.com/ethz-spylab/agentdojo) (Debenedetti et al., NeurIPS 2024, [arXiv:2406.13352](https://arxiv.org/abs/2406.13352)) into the framework.

## What it does

One `AgentDojoTarget` instance exposes the **union of all four AgentDojo suites** simultaneously: banking, workspace, slack, travel. Roughly 74 tools and four independent sub-environments are available to the agent at all times; tasks configure the parts of the environment they care about, the agent decides what to call. The benchmark version is pinned to the latest released (`v1.2.2`) via the public `agentdojo_target.BENCHMARK_VERSION` constant.

Two attacker capability surfaces:

1. **On-demand content injection** on every readable tool. When the agent reads, the wrapper computes the legitimate value, fires a `ControllablePostCallEvent`, and substitutes the agent-visible return with the optimizer's `ControllableInjection.value` if any. This is strictly more expressive than AgentDojo's static `{slot}` substitution.
2. **Tool catalogue editing** as four separate controllables: register a new tool, replace an existing tool, unregister, rewrite description. Catalogue edits fire once per agent turn before the LLM call.

The security domain forest has three trees:

- `system`: prompt, tool_catalogue, model_identity, agent_trace (with messages, tool_calls, tool_responses children)
- `user`: a single tag for the user prompt
- `tools`: a 2x2 grid (`content_1p_data_1p`, `content_1p_data_3p`, `content_3p_data_1p`, `content_3p_data_3p`) classifying every readable data field by content provider and data store

## Install

```bash
# from the superred-modules repo root
pip install -e ./targets/agentdojo
```

The target depends on `superred` and the upstream `agentdojo` package (used for its `BaseUserTask`/`BaseInjectionTask` instances and for loading the v1 environment YAMLs).

## Quick start

```python
from agentdojo_target import AgentDojoTarget
from agentdojo_target.security_tags import USER_TAG
from superred.core.controller import Controller, TargetFactory
from superred.core.types.llm import LLMConfig

target_factory = TargetFactory(
    create=lambda: AgentDojoTarget(
        pipeline_model="openai/gpt-4o-2024-05-13",
        api_base=BASE,
        api_key=KEY,
    ),
    concurrency=1,
)
controller = Controller(
    optimizer_factory=lambda: MyOptimizer(),
    target_factory=target_factory,
    security_claim=my_claim,
    scope=frozenset({USER_TAG}),
)
await controller.run()
```

## See also

- [`security_claim_agentdojo`](../../security_claims/agentdojo/): the paired SecurityClaim package with three layers (original AgentDojo tasks, system-purpose-violation tasks, combined).
- `ASSUMPTIONS.md`: enumerates every divergence from AgentDojo upstream.

## Status

v0.1.0 alpha. The public API is unstable and may change without notice.
