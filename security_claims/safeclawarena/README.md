# superred-claim-safeclawarena

A superred **SecurityClaim** porting [SafeClawArena](https://github.com/sunblaze-ucb/SafeClawArena)
(Niu et al., arXiv:2606.30755) — a computer-system-perspective security benchmark
for self-hosted AI agents. It bundles **406 adversarial tasks** across four
principle-aligned dimensions and judges them with SafeClawArena's own
**deterministic canary/taint** logic (no LLM judge).

| Dimension | Tasks | Principle |
| --- | --- | --- |
| SSI — Skill Supply-Chain Integrity | 100 | process isolation, least privilege |
| PSE — Persistent State Exploitation | 60 | persistent-state protection |
| CDF — Cross-Boundary Data Flow | 146 | cross-boundary mediation |
| IPI — Indirect Prompt Injection | 100 | data-instruction separation |

Pairs with the [`superred-target-safeclawarena`](../../targets/safeclawarena)
package, which runs each task against a containerized platform replica
(OpenClaw / NemoClaw / SecLaw) and captures the post-run state this claim judges.

## Usage

```python
from safeclawarena_claim import safeclawarena_claim, safeclawarena_ipi_claim
from safeclawarena_claim import safeclawarena_target_factory

claim = safeclawarena_claim()                       # all 406 tasks
ipi = safeclawarena_ipi_claim()                     # 100 IPI tasks
one = safeclawarena_claim(task_ids=["ipi-4.5-008"]) # a single task
subset = safeclawarena_claim(dimensions=["cdf"], difficulties=["advanced"])

target = safeclawarena_target_factory(platform="openclaw")
# hand `claim` and `target` to the superred Controller with an optimizer.
```

Each task's `attack_succeeded` is decided deterministically (a canary credential
appears where it must not, a file is created/deleted, a taint marker reaches an
unauthorized channel, …); the dimension `score` is 1.0 for secure behaviour and
0.0 for a fully successful attack.

## Provenance & faithfulness

The 406 task JSONs and the task schema are vendored **byte-for-byte** from
upstream (verify with `python scripts/sync_upstream.py --check`). The check
evaluators, detection, taint-leakage, and the dimension-scoring formulas are
ported from upstream `scripts/judge.py` into a pure judge. Every deliberate
deviation is documented in [`ASSUMPTIONS.md`](ASSUMPTIONS.md). MIT-licensed
(module code); SafeClawArena's MIT licence is shipped in `LICENSES/` and
`src/safeclawarena_claim/data/`.

When reporting numbers, cite Niu et al., 2026 (arXiv:2606.30755).
