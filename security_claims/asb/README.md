# superred-claim-asb

Agent Security Bench (ASB) security claim for the [superred](https://superred.simonsure.com)
red-teaming framework. It exposes one `Task` per `(agent, benign task, attacker tool)`,
with ASB's attack-success / utility / refusal predicates ported verbatim, plus an
adversarial success judge and hierarchical factories by scenario, aggressiveness, and
attack type. Pairs with the `superred-target-asb` target.

## Credits / upstream

This module ports **Agent Security Bench (ASB)** into the superred framework.

- **ASB** — https://github.com/agiresearch/ASB (commit `1f561dcc`), MIT,
  Copyright (c) 2024 AGI Research. Paper: Zhang, Huang, Mei, Yao, Wang, Zhan,
  Wang, Zhang, "Agent Security Bench (ASB)", ICLR 2025.

The ASB benchmark data (agent tasks, attacker tools, normal tools) is vendored
verbatim under `src/asb_claim/data/` and the upstream MIT license is
preserved in `data/ASB_LICENSE`. ASB's attack-success / utility / refusal
predicates and refusal-judge prompt are ported from `main_attacker.py`; see
`ASSUMPTIONS.md` for deviations. Our SecurityClaim wiring, the adversarial
success judge, and the tool-invocation success predicate are original
(MIT, see `LICENSE`).

When reporting results produced with this module, please cite ASB (ICLR 2025).
