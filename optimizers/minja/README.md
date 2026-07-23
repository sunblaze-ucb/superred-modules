# superred-optimizer-minja

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

```bash
pip install superred-optimizer-minja
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

## License

**No license is granted yet** (see [LICENSE](LICENSE)). The upstream reference
implementation, [`dsh3n77/MINJA`](https://github.com/dsh3n77/MINJA), carries
**no license file** (default all-rights-reserved), and this module bundles
RAP/WebShop scenario data reproduced from it. Because there is no upstream
license to redistribute that material under, this module is withheld pending the
upstream author's permission and is not released under MIT (unlike other superred
optimizers). Once permission is obtained or the data is fully replaced with
original content, a proper license will be applied here.
