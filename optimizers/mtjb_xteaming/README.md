# mtjb_xteaming

X-Teaming strategy-driven multi-turn jailbreak optimizer, ported from
[MT-JailBench](https://arxiv.org/abs/2605.11002) (`engine/attacks/xteaming`,
pinned at commit `cb8184e0`).

## What it does

At setup the attacker LLM generates a set of attack strategies — each a persona,
a context, an approach, and a per-turn conversation plan — and one is selected.
Each turn's prompt is generated from the plan (first / nth / final templates)
and sent as one conversation turn. A judge scores every response 1–5; the flow
advances the plan when a turn improves on the best score so far and otherwise
refines. Upstream's refinement is a **TextGrad** textual-gradient step that
rewrites the attacker prompt toward a score of 5; the already-observed target
response is fed into the gradient so the target is never re-queried.

The optimizer sends one turn per superred run, pins one controllable surface for
the whole conversation, and routes strategy / turn / judge calls through
`self.llm`.

## TextGrad is optional

The refine step needs `textgrad` (a pure-Python wheel). Install it with the
`refine` extra:

```
pip install superred-optimizer-mtjb-xteaming[refine]
```

Without `textgrad` the optimizer still runs end to end: a would-be refine
advances the strategy plan instead (a documented no-rewind fallback). The
module imports and the strategy/turn/judge path work with no heavy dependency.

## Usage

```python
from mtjb_xteaming_optimizer import MtjbXTeamingOptimizer

opt = MtjbXTeamingOptimizer(max_total_turns=10, max_refines_per_turn=4)
```

## Dependencies

Required: `superred`, `pyyaml`. Optional (`refine` extra): `textgrad`,
`tiktoken`.

## Scope

Ports **X-Teaming only**. See `mtjb_coa` (CoA) and `mtjb_mix` (the composite).
Not ported: `actor` / `crescendo` / `fitd` (superred already ships those
techniques) and `interactive` (a human-in-the-loop console harness upstream
itself calls "not a real attack").

## Faithfulness

See [ASSUMPTIONS.md](ASSUMPTIONS.md). superred has only a full between-run reset,
so upstream's in-turn `RETRY` and strategy-rotating `JUMP_TO` are realised as
sequential advance with pre-send refinement; the TextGrad step runs in a worker
thread bridged back to `self.llm`.
