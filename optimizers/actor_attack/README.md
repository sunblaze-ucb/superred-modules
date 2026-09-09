# superred-optimizer-actor-attack

ActorAttack multi-turn optimizer for
[superred](https://superred.simonsure.com).

> **Based on Tencent Zhuque Lab AI-Infra-Guard
> (https://github.com/Tencent/AI-Infra-Guard).**
> This attribution is required by that project's NOTICE under Apache-2.0
> Section 4(d).

> **Community port — not an official implementation.** Ports Tencent's
> Apache-2.0 implementation of the published ActorAttack method. Not affiliated
> with Tencent or the paper's authors. See [ASSUMPTIONS.md](ASSUMPTIONS.md).

## What it does

The objective is never requested directly. The attacker LLM first invents a
network of **actors** — people or entities related to the goal — then
interrogates the target about one of them. Each turn is written from the
target's *previous answer*, so the conversation drifts toward the objective
one innocuous question at a time.

Per turn: send the probe → classify the answer as refusal or not → score it
with the attacker-side judge → write the next probe from that answer. The
attack stops when the judge rates a turn 10, when the turn budget runs out, or
when the attacker marks a probe final.

## Install

```bash
pip install superred-optimizer-actor-attack
```

## Usage

```python
from actor_attack_optimizer import ActorAttackOptimizer

optimizer_factory = lambda: ActorAttackOptimizer()

# ask for more candidate actors and allow a longer interrogation
optimizer_factory = lambda: ActorAttackOptimizer(
    num_actors=5, max_turns_per_actor=6
)
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `num_actors` | `3` | actors to request (upstream uses the first; clamped to ≥1) |
| `max_turns_per_actor` | `4` | turns before stopping (clamped to ≥2) |

Both clamp rather than reject, matching upstream.

## Credits / upstream

Ported from [Tencent AI-Infra-Guard](https://github.com/Tencent/AI-Infra-Guard)
(Apache-2.0), pinned at
[`dd6bd546`](https://github.com/Tencent/AI-Infra-Guard/tree/dd6bd54655c9ff5fb7351f4299b56916f09ec6da/AIG-PromptSecurity/deepteam/attacks/multi_turn/actor_attack).
Technique: Ren et al., *Derail Yourself: Multi-turn LLM Jailbreak Attack
through Self-discovered Clues*. The authors' own repository ships no licence,
so no code was taken from it. Verify with
`python scripts/sync_upstream.py --check`. See [`NOTICE`](NOTICE) and
[`LICENSES/NOTICE.md`](LICENSES/NOTICE.md).
