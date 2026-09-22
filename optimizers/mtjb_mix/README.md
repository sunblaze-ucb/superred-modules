# mtjb_mix

Mix composite multi-turn jailbreak optimizer, ported from
[MT-JailBench](https://arxiv.org/abs/2605.11002) (`engine/attacks/mix`, pinned at
commit `cb8184e0`).

## What it does

Mix picks three roles independently and drives them through the shared engine:

- **generator** — which family produces each turn's prompt,
- **updater** — which family refines a prompt when the flow retries,
- **judge_and_flow** — which family's judge scores responses and whose flow
  controller decides continue / retry / stop.

Each role is one of `Crescendo`, `ActorBreaker`, `ChainOfAttack`,
`FootInTheDoor`, `XTeaming`. Because reimplementing all five families natively
would duplicate the whole ecosystem, Mix **reuses MT-JailBench's engine
verbatim**: this package vendors the `engine/` subtree and drives it, replacing
only the model-access layer with a shim that routes every call to `self.llm`.

## Requires the `refine` extra

The vendored generator/updater import `textgrad` (and `tiktoken`) at load, so the
Mix engine only runs with them installed:

```
pip install superred-optimizer-mtjb-mix[refine]
```

Without the extra the optimizer still **imports** and is a well-behaved
no-op: it ends the run without sending (the engine cannot start). `jinja2` and
`tenacity` are always required (Crescendo templating and retry).

## Usage

```python
from mtjb_mix_optimizer import MtjbMixOptimizer

opt = MtjbMixOptimizer(
    generator="ChainOfAttack",
    updater="XTeaming",
    judge_and_flow="XTeaming",
    max_turns=5,
)
```

Invalid role names raise at construction.

## Scope

Ports **Mix only**; see `mtjb_coa` and `mtjb_xteaming` for the standalone CoA and
X-Teaming ports. Not ported: `actor` / `crescendo` / `fitd` as standalone
optimizers (superred already ships those techniques) and `interactive` (a
human-in-the-loop console harness upstream calls "not a real attack"). Mix does
*compose* the Actor / Crescendo / FITD families internally — their prompt
internals are vendored for that purpose.

## Faithfulness

See [ASSUMPTIONS.md](ASSUMPTIONS.md). The engine loop is reconstructed against
superred's one-turn-per-run model; because superred has only a full between-run
reset, upstream `RETRY` sends the refined prompt as the next turn and `JUMP_TO`
ends the attack. The vendored judge/flow/generator/updater logic is otherwise
run unchanged.
