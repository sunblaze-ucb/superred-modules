# minja-optimizer

A **MINJA (Memory INJection Attack)** optimizer for
[superred](https://superred.simonsure.com) agent targets.

MINJA poisons an LLM agent's long-term memory using only ordinary queries — no
privileged access. Across a series of innocuous-looking questions it plants a
"bridge" memory record that links a victim term to an attacker-chosen target.
Later, when a clean victim query is asked, the agent retrieves the poisoned
memory and produces the attacker's target instead.

Reference: Dong et al., "Memory Injection Attacks on LLM Agents via Query-Only
Interaction," [arXiv:2503.03704](https://arxiv.org/abs/2503.03704)
(NeurIPS 2025). Official implementation:
[`dsh3n77/MINJA`](https://github.com/dsh3n77/MINJA). See
[ASSUMPTIONS.md](ASSUMPTIONS.md) for the full paper-to-code mapping and every
deliberate departure.

## Install

From the repository root, with the shared virtual environment active:

```bash
pip install -e ../superred        # the framework first
pip install -e optimizers/minja
```

## Usage

```python
from minja_optimizer import MinjaOptimizer

# By default the scenario (victim -> target pair, bridge, questions) is derived
# from the superred goal and visible observables. Defaults inject_num=15,
# num_benign=50, test_num=30 match the official rap/minja.py.
optimizer = MinjaOptimizer()

# Supply an explicit scenario for exact custom content.
from minja_optimizer import MinjaScenario, VictimTargetPair
optimizer = MinjaOptimizer(scenario=MinjaScenario(...))
```

This optimizer targets **agent-style** targets whose memory persists across runs
within a task. Wire it into a `Controller` like any other superred optimizer —
see the framework's [architecture overview](https://superred.simonsure.com).

## Assumptions

Plain-language summary; see [ASSUMPTIONS.md](ASSUMPTIONS.md) for the details.

- **Scenario-driven.** A `MinjaScenario` bundles the victim term, target term,
  progressive-shortening indication prompts, bridge template, and victim
  questions. By default it is derived from the goal; if the context clearly
  points at the paper's RAP/WebShop setup, the official RAP content is used.
  Ambiguous goals fail loudly rather than running an unrelated fallback.
- **Needs persistent memory.** Injection runs poison the target's durable memory,
  which survives between runs within a task; a later trigger run issues a clean
  victim query to retrieve it. A target whose memory does not persist cannot be
  attacked by MINJA, by construction.
- **Query-only by default, with opt-in extensions.** The core attack uses only
  ordinary queries. When the threat model grants extra surfaces (writable system
  prompt, memory/tool post-calls, tool catalog), the optimizer can use them on
  injection stages — capability extensions beyond the paper's query-only setting.
- **Success scoring.** Success means the target term appears in the agent's
  output on a *trigger* run (a clean victim query, no indication prompt), so it
  reflects real memory retrieval. Framework evaluation is authoritative when
  present.

## License

MIT for this port's code (see [LICENSE](LICENSE)). The upstream reference
implementation, [`dsh3n77/MINJA`](https://github.com/dsh3n77/MINJA), carries
**no license file** (default all-rights-reserved). This port does not vendor
upstream runtime source; the RAP/WebShop scenario data it reads is reconstructed
from that publicly observable repository, and the attack is reimplemented against
superred's event model. If upstream publishes a license in the future, this
notice should be revisited.
