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

From the repository root, with the shared virtual environment active:

```bash
pip install -e ../superred        # the framework first
pip install -e optimizers/chord_xthp
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

## Assumptions

Plain-language summary; see [ASSUMPTIONS.md](ASSUMPTIONS.md) for the details.

- **Writable tool catalog is required.** Every Chord attack needs the malicious
  helper registered so the agent can call it, and only a catalog controllable can
  do that. If no writable tool/skill catalog is in scope, the optimizer finishes
  immediately instead of burning attempts — even if a system or user prompt is
  writable.
- **Official data reused.** The packaged helper names/descriptions, sensitive-
  argument mappings, and example queries come from Chord; success follows Chord's
  HSR (hijack) and HASR (harvest) rules, with an LLM judge for harvest matching
  the official prompt.
- **Injects, does not re-run Chord.** Chord's own runtime owns a full
  LangChain/LlamaIndex agent loop; superred already owns the controller, target,
  and trajectory, so this optimizer injects Chord-style helpers through granted
  superred surfaces instead of running Chord's runtime inside superred.
- **Victim selection.** A visible tool catalog is the source of truth for
  choosing the victim tool; official names and goal hints are used only when they
  match the visible catalog or when no catalog is visible.

## License

MIT for this port's code (see [LICENSE](LICENSE)). The upstream reference
implementation, [`systemsecurity-uiuc/Chord`](https://github.com/systemsecurity-uiuc/Chord),
is MIT licensed. The bundled helper data and prompts originate from that
repository. This port reimplements the attack against superred's event model
rather than vendoring Chord's agent runtime.
