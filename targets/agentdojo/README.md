# agentdojo-target

A composite [superred](../../../superred) Target that ports the [AgentDojo benchmark environment](https://github.com/ethz-spylab/agentdojo) (Debenedetti et al., NeurIPS 2024, [arXiv:2406.13352](https://arxiv.org/abs/2406.13352)) into the framework.

## What it does

One `AgentDojoTarget` instance exposes the **union of all four AgentDojo suites** simultaneously: banking, workspace, slack, travel. Roughly 74 tools and four independent sub-environments are available to the agent at all times; tasks configure the parts of the environment they care about, the agent decides what to call. The benchmark version is pinned to the latest released (`v1.2.2`) via the public `agentdojo_target.BENCHMARK_VERSION` constant.

Two attacker capability surfaces:

1. **On-demand content injection** on every readable tool. When the agent reads, the wrapper computes the legitimate value, fires a `ControllablePostCallEvent`, and substitutes the agent-visible return with the optimizer's `ControllableInjection.value` if any. This is strictly more expressive than AgentDojo's static `{slot}` substitution.
2. **Tool catalogue editing** as four separate controllables: register a new tool, replace an existing tool, unregister, rewrite description. The four controllables are tagged by capability: register at `tool_catalogue_add`, replace and rewrite description at `tool_catalogue_edit`, unregister at `tool_catalogue_remove`. Catalogue edits fire once at run start, before the first LLM call, then stay fixed for the run.

The model is a construction concern, fixed by the `AgentDojoTarget(pipeline_model=...)` constructor argument, not a per-run config slot; calling `set_config("pipeline_model", ...)` raises.

The security domain forest has three trees:

- `system`: system_prompt, tool_catalogue (a pure grouping root subsuming three capability children: `tool_catalogue_add`, `tool_catalogue_edit`, `tool_catalogue_remove`), model_identity, agent_trace (with a single `agent_trace_messages` child)
- `user`: a single tag for the user prompt
- `tools`: a per-service, per-store forest. `TOOLS_TAG` is a pure grouping root (nothing is emitted at it). Under it sit four service nodes (`BANKING_TAG`, `WORKSPACE_TAG`, `SLACK_TAG`, `TRAVEL_TAG`), and under each service sit store leaves matching the real data stores: banking has `banking_bank_account`, `banking_filesystem`, `banking_user_account`; workspace has `workspace_inbox`, `workspace_calendar`, `workspace_cloud_drive`; slack has `slack_slack`, `slack_web`; travel has `travel_hotels`, `travel_restaurants`, `travel_car_rental`, `travel_flights`, `travel_user`, `travel_calendar`, `travel_reservation`, `travel_inbox`. Granting a service grants its stores; granting `TOOLS_TAG` grants everything. Each read tool is tagged at the store leaf it reads from, and each write tool's observation is tagged at the store leaf it mutates, so reading from and acting on the same store share one label.

Store contents are not mirrored as observables: values reachable through a read controllable appear only on that controllable's events, and the full environment is available only post-run to the scorer via the query specs. Writes additionally surface a one-way `write_call` observation tagged at the store they mutate. The tool call itself (function + args) and its return are not double-emitted as observables: they live exactly once on that tool's `ControllablePostCallEvent`. The `agent_trace` subtree now carries only the non-tool internal message stream (assistant/user/system text, with tool-result messages skipped and the `tool_calls` field stripped from assistant messages).

Read-only access to a surface is not a separate tag: grant it per threat model by listing the tag in the Controller's `read_only` set instead of its read & write `scope`. For example `read_only={SYSTEM_PROMPT_TAG}` lets the optimizer see the system prompt on the trajectory (the Phase-1 controllable event carries it) without being able to override it, and `read_only={WORKSPACE_INBOX_TAG}` lets the optimizer watch the agent's inbox reads without injecting into them. Every piece of information is emitted exactly once: values that flow through a controllable appear only on that controllable's events. The `agent_trace` projection carries only the non-tool internal message stream and does not re-carry tool calls or agent-seen tool returns; those live solely on the per-tool `ControllablePostCallEvent`.

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
