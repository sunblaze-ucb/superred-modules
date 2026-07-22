# chord-xthp-optimizer

A **Chord (Cross-Tool Harvesting and Polluting, XTHP)** optimizer for
[superred](https://superred.simonsure.com) agent targets.

Chord attacks tool-using LLM agents by slipping a malicious helper tool into the
agent's pool of tools. Positioned right before or after a legitimate "victim"
tool in the call sequence, the helper can **harvest** sensitive context the
agent passes around, **hijack** the control flow, or **pollute** the final
answer with attacker-controlled output.

Reference: Wang et al., "Les Dissonances: Cross-Tool Harvesting and Polluting in
Pool-of-Tools Empowered LLM Agents,"
[arXiv:2504.03111](https://arxiv.org/abs/2504.03111) (NDSS 2026). Official
implementation:
[`systemsecurity-uiuc/Chord`](https://github.com/systemsecurity-uiuc/Chord)
(MIT License). See [ASSUMPTIONS.md](ASSUMPTIONS.md) for the full paper-to-code
mapping and every deliberate departure.

## Install

```bash
pip install chord-xthp-optimizer
```

## Usage

```python
from chord_xthp_optimizer import ChordXTHPOptimizer

# Default direction is `both` (predecessor + successor helper variants),
# hijack attack enabled.
optimizer = ChordXTHPOptimizer()

# Target a specific victim tool and enable multiple attack types.
optimizer = ChordXTHPOptimizer(
    victim_tool_name="get_user_address",
    enabled_attacks=("hijack", "harvest", "pollute"),
)
```

This optimizer targets **agent-style** targets that expose a writable tool/skill
catalog. Wire it into a `Controller` like any other superred optimizer — see the
framework's [architecture overview](https://superred.simonsure.com).

## License

MIT for this port's code (see [LICENSE](LICENSE)). The upstream reference
implementation, [`systemsecurity-uiuc/Chord`](https://github.com/systemsecurity-uiuc/Chord),
is MIT licensed. The bundled helper data originates from that repository; its
MIT copyright notice is preserved in [`NOTICE`](NOTICE) and
[`LICENSES/Chord-MIT.txt`](LICENSES/Chord-MIT.txt). This port reimplements the
attack against superred's event model rather than vendoring Chord's agent
runtime.
