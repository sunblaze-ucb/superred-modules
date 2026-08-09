# security-claim-agentdojo

A [superred](../../../superred) `SecurityClaim` package paired with [`agentdojo-target`](../../targets/agentdojo).  Three composable layers:

1. **Layer 1: original AgentDojo injection tasks.**  One paired Task per canonical (user_task, injection_task) pair, 27 tasks total across the four AgentDojo suites (banking, workspace, slack, travel).  Subdivided by attack-semantic category per the table the user confirmed.  Benchmark version pinned to the latest released (`v1.2.2`) via the target's `seed_loader.BENCHMARK_VERSION`.
2. **Layer 2: bespoke system-purpose-violation goals.**  Standalone goals tailored to the composite multi-suite agent.  Deterministic security predicates, NOT LLM-as-judge.  Each task carries a benign user prompt; the attacker chooses delivery based on scope.
3. **Layer 3: combined.**  Composition of Layers 1 and 2.

> **Community port — not an official implementation.** This module is an
> unofficial re-implementation of AgentDojo (Debenedetti et al., ETH Zurich SPY
> Lab) for superred. It is not affiliated with, endorsed by, or maintained by
> the original authors. See [ASSUMPTIONS.md](ASSUMPTIONS.md) for every
> deliberate deviation from the paper and reference code.

## Install

```bash
# from the superred-modules repo root
pip install -e ./targets/agentdojo
pip install -e ./security_claims/agentdojo
```

## Quick start

```python
from agentdojo_target import AgentDojoTarget
from agentdojo_claim import agentdojo_layer1_claim
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
    scope=frozenset({USER_TAG, TOOLS_TAG}),  # example scope
)
await controller.run()
```

See `ASSUMPTIONS.md` for divergences from AgentDojo upstream and Layer-2 design rationale.

## Status

v0.1.0 alpha.  Public surface is unstable.

## Credits / upstream

This package's source is original superred work (MIT, Copyright (c) 2026
Simon Sure). It bundles no third-party code or data.

**Layer 1** adapts the [AgentDojo](https://github.com/ethz-spylab/agentdojo)
benchmark (Debenedetti et al., 2024; MIT, Copyright (c) 2024 Edoardo
Debenedetti, Jie Zhang, Mislav Balunovic, Luca Beurer-Kellner, Marc Fischer,
and Florian Tramèr). AgentDojo is a runtime dependency (`agentdojo>=0.1.35`),
installed from PyPI and not redistributed here; its injection/user tasks are
run and scored via superred's `agentdojo-target`. Please cite AgentDojo when
reporting Layer-1 results.

**Layer 2** goals are bespoke, hand-authored superred content (deterministic
security predicates, not derived from AgentDojo's tasks). See `ASSUMPTIONS.md`.
