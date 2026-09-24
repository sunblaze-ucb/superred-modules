# mtjb_coa

Chain-of-Attack (CoA) multi-turn jailbreak optimizer, ported from
[MT-JailBench](https://arxiv.org/abs/2605.11002) (`engine/attacks/coa`, pinned
at commit `cb8184e0`). MT-JailBench's CoA is itself adapted from
[YancyKahn/CoA](https://github.com/YancyKahn/CoA).

## What it does

At setup the attacker LLM generates several candidate multi-round "chains" of
prompts that walk from an innocuous opener toward the objective, and keeps the
chain with the widest semantic range. Each round's prompt is sent as one
conversation turn. When attack update is enabled (the default), the next
round's prompt is refined **before it is sent**, using the previous round's
answer, so the semantic trajectory keeps climbing toward the goal. A binary
judge (`[[0]]`/`[[1]]`) decides success after each turn.

The optimizer sends one turn per superred run and carries state on the
instance, pinning a single controllable surface for the whole conversation.
Every model call — chain generation, prompt update, the binary judge, and the
LLM-approximation similarity scorer — is routed through the constrained
`self.llm`. No external SDKs, no similarity endpoint, no `sentence-transformers`.

## Usage

```python
from mtjb_coa_optimizer import MtjbCoAOptimizer

opt = MtjbCoAOptimizer(max_turns=5)  # chain length = number of rounds
```

Constructor knobs mirror the upstream config: `max_turns`, `n_init_chains`,
`max_update_retries`, `semantic_update_slack`, `enable_attack_update`,
`max_judge_retries`.

## Dependencies

`superred`, `pyyaml`. No heavy dependencies: CoA is the lightest of the ported
MT-JailBench attacks (no `textgrad`, no `tiktoken`, no `requests`).

## Scope

MT-JailBench ships seven attack families. This package ports **CoA only**. The
sibling package `mtjb_xteaming` ports X-Teaming and `mtjb_mix` ports the Mix
composite. The following are intentionally **not** ported:

- `actor`, `crescendo`, `fitd` — superred already ships these techniques as
  first-class optimizers, so re-porting MT-JailBench's variants would duplicate
  them.
- `interactive` — a human-in-the-loop `input()`/`print()` console harness that
  upstream itself comments is "not a real attack"; it cannot run in an
  autonomous async optimizer.

## Faithfulness

See [ASSUMPTIONS.md](ASSUMPTIONS.md) for every deliberate deviation. The
biggest one: superred exposes only a full between-run target reset, not the
partial conversation rewind CoA's `RETRY`/`JUMP_TO` flow assumes, so refinement
is moved to pre-send and the mirror-target requery (`base_response_sem`) is
disabled.
