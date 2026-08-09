# muzzle-optimizer

A superred port of **MUZZLE** (arXiv:2602.09222, "Adaptive Agentic Red-Teaming of Web
Agents Against Indirect Prompt Injection Attacks"), an adaptive **indirect-prompt-injection
(IPI)** attacker for tool-using agents.

> **Community port — not an official implementation.** This module is an
> unofficial re-implementation of MUZZLE (Syros et al.) for superred, and
> vendors PAIR (Chao et al.). It is not affiliated with, endorsed by, or
> maintained by the original authors. See [ASSUMPTIONS.md](ASSUMPTIONS.md) for
> every deliberate deviation from the paper and reference code.

Given an agentic target and an adversarial goal, `MuzzleOptimizer`:

1. **observes** one benign run and summarizes the victim's trajectory into a *playbook*;
2. **ranks** the in-scope content-injection surfaces (*vessels*) by salience;
3. **probes** a vessel by planting a placeholder and checking it surfaces in the victim's
   context;
4. **crafts** a context-aware payload with an offline PAIR loop (a simulated victim
   conditioned on the real system prompt and the localized context);
5. **attacks** with one real run, then **refines** from feedback: a missed delivery moves to
   the next vessel, an ineffective payload is re-refined by PAIR.

Success is decided by the SecurityClaim (`RunEndEvent.evaluation.success`), never by the
optimizer. See [`ASSUMPTIONS.md`](ASSUMPTIONS.md) for the faithfulness ledger and every
deliberate deviation from the upstream `github.com/gsiros/muzzle` (SHA `ed611c0`).

## Install

```bash
pip install -e ./optimizers/muzzle
```

Depends only on `superred` and `PyYAML`. The Chao et al. PAIR core is **vendored** under
`muzzle_optimizer/pair/` (no cross-module dependency).

## Usage

```python
from muzzle_optimizer import MuzzleOptimizer
from superred.core.controller import Controller, TargetFactory

controller = Controller(
    optimizer_factory=lambda: MuzzleOptimizer(),   # zero-arg; a fresh instance per task
    target_factory=TargetFactory(create=make_agent_target, concurrency=1),
    security_claim=ipi_claim,
    scope=root_scope,
    llm_config=attacker_cfg,     # the one fixed attacker model MUZZLE's roles share
    max_runs_per_task=12,        # see "Run budget" below
)
result = await controller.run()
```

The optimizer adapts entirely from `initialize(...)`; it takes **no required constructor
arguments**. Useful keyword knobs include `max_attempts`, the PAIR caps
(`pair_n_streams`, `pair_n_iterations`), `target_controllable_name` (pin a vessel for
ablations), and `response_observable_names`.

## How it maps to the framework

| MUZZLE component | superred |
| --- | --- |
| Explorer (Docker/Zoo/vLLM-proxy/red-team browser agent) | the controller's run loop + the trajectory + a direct `ControllableInjection` |
| BENIGN / PROBE / ACTUAL victim runs | successive `target.run()` calls in the multi-run loop |
| vessel (web UI element) | an in-scope content **Controllable** (PostCall: `tool:<name>` / `read__*` / `opi_tool_observation`), with PreCall fallback |
| curl assertion oracle | `RunEndEvent.evaluation.success` (the SecurityClaim) |
| Summarizer / Grafter / Prompter / Judge (gpt-4o agents) | helper classes over the single `self.llm`, each with its byte-identical prompt |
| PAIR (offline, bare victim LLM) | `PairBridge`: offline PAIR with `self.llm` as the surrogate victim |

The optimizer runs a three-phase state machine over the multi-run loop:
**OBSERVE** (passthrough recon -> playbook -> ranked vessels) -> **PROBE** (placeholder
delivery check per vessel; a miss advances the vessel and is *not* scored) -> **ATTACK**
(inject the PAIR payload; success ends the task, a UI-attributed failure tries the next
vessel, an instruction-attributed failure re-refines).

## Scope behavior

- **Content vessels in scope:** the intended IPI setting; rank, probe, attack.
- **PreCall-only:** degrade to a `user_prompt` (preferred) or `system_prompt` fallback
  vessel (less indirect, deprioritized).
- **No injectable surface:** one OBSERVE passthrough, then `done` (the passthrough baseline).
  Never crashes.
- **`include_feedback=False`:** steers by the trajectory presence/PROBE checks only and never
  self-certifies success.

## Run budget

OBSERVE is a pure recon run, so a single phase cannot attack. Allow at least **~4-5 runs per
goal** (`max_runs_per_task`) so OBSERVE + PROBE + ACTUAL can complete; the offline PAIR loop
consumes only the optimizer's LLM cost budget, not target runs. `BudgetExhaustedError` stops
the task cleanly.

## Tests

```bash
pip install -e "./optimizers/muzzle[test]"
pytest optimizers/muzzle/tests -q
mypy --strict optimizers/muzzle/src
ruff check optimizers/muzzle
```

`smoke/` holds env-gated live end-to-end checks against real agentic targets (`agentdojo`,
`inspect_agent`); they need an LLM proxy (`LITELLM_API_KEY` / `LITELLM_API_BASE`).

## Credits / upstream

This module ports the **MUZZLE** red-teaming method to superred and vendors the
**PAIR** attacker core. Our integration code is MIT (see `LICENSE`); upstream
components keep their original licenses (see `NOTICE` and `LICENSES/`).

- **MUZZLE** — Syros et al., "MUZZLE: Adaptive Agentic Red-Teaming of Web Agents
  Against Indirect Prompt Injection Attacks", arXiv:2602.09222 (2026).
  Source: https://github.com/gsiros/muzzle (Apache-2.0). The five agent prompt
  YAMLs, the four injection templates, the judge payload-presence heuristic, and
  the OBSERVE/PROBE/ATTACK control logic derive from this repo. Deviations from
  upstream are logged in `ASSUMPTIONS.md`.
- **PAIR** — Chao et al., "Jailbreaking Black Box Large Language Models in Twenty
  Queries" (2023). Source: https://github.com/patrickrchao/JailbreakingLLMs
  (MIT, Copyright (c) 2023 PAIR Team). Vendored under `muzzle_optimizer/pair/`.

Please cite both papers when reporting results produced with this module.
