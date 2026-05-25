# security-claim-agentdojo

A [superred](../../../superred) `SecurityClaim` package paired with [`agentdojo-target`](../../targets/agentdojo).  Three composable layers:

1. **Layer 1: original AgentDojo injection tasks.**  One paired Task per canonical (user_task, injection_task) pair, 27 tasks total across the four AgentDojo suites (banking, workspace, slack, travel).  Subdivided by attack-semantic category per the table the user confirmed.  Benchmark version pinned to the latest released (`v1.2.2`) via the target's `seed_loader.BENCHMARK_VERSION`.
2. **Layer 2: bespoke system-purpose-violation goals.**  Standalone goals tailored to the composite multi-suite agent.  Deterministic security predicates, NOT LLM-as-judge.  Each task carries a benign user prompt; the attacker chooses delivery based on scope.
3. **Layer 3: combined.**  Composition of Layers 1 and 2.

## Install

```bash
# from the superred-modules repo root
pip install -e ./targets/agentdojo
pip install -e ./security_claims/agentdojo
```

## Quick start

```python
from agentdojo_target import AgentDojoTarget
from security_claim_agentdojo import agentdojo_layer1_claim
from superred.core.controller import Controller, TargetFactory

target_factory = TargetFactory(
    create=lambda: AgentDojoTarget(
        pipeline_model="openai/gpt-4o-2024-05-13",
        api_base=BASE, api_key=KEY,
    ),
    concurrency=1,
)
claim = agentdojo_layer1_claim()  # all 27 canonical pairs

controller = Controller(
    optimizer_factory=lambda: MyOptimizer(),
    target_factory=target_factory,
    security_claim=claim,
    scope=frozenset({USER_TAG, CONTENT_3P_DATA_3P_TAG}),  # example scope
)
await controller.run()
```

See `ASSUMPTIONS.md` for divergences from AgentDojo upstream and Layer-2 design rationale.

## Status

v0.1.0 alpha.  Public surface is unstable.
