# superred-optimizer-eia-agent

An **EIA (Environmental Injection Attack)** optimizer for
[superred](https://superred.simonsure.com) web-agent targets.

> **Community port — not an official implementation.** This module is an
> unofficial re-implementation of EIA (Liao et al., OSU NLP Group) for
> superred. It is not affiliated with, endorsed by, or maintained by the
> original authors. See [ASSUMPTIONS.md](ASSUMPTIONS.md) for every deliberate
> deviation from the paper and reference code.

EIA attacks web agents by changing the environment rather than the instruction.
It plants hidden or visually subtle HTML elements — persuasive labels, injected
form fields, aria-labels — on a page the agent reads. The agent then grounds its
next action to the attacker's element and types private information into it,
leaking data without ever being told to.

Reference: Liao et al., "EIA: Environmental Injection Attack on Generalist Web
Agents for Privacy Leakage,"
[arXiv:2409.11295](https://arxiv.org/abs/2409.11295) (ICLR 2025). Official
implementation:
[`OSU-NLP-Group/EIA_against_webagent`](https://github.com/OSU-NLP-Group/EIA_against_webagent).
See [ASSUMPTIONS.md](ASSUMPTIONS.md) for the full paper-to-code mapping and every
deliberate departure.

## Install

```bash
pip install superred-optimizer-eia-agent
```

## Usage

```python
from eia_agent_optimizer import EIAAgentOptimizer

# Defaults follow the paper/code mainline (action_grounding + form_type1,
# near_bot_1 placement, zero-opacity injection).
optimizer = EIAAgentOptimizer()

# Supply the leak target so the local no-feedback evaluator can verify the value.
optimizer = EIAAgentOptimizer(privacy_type="credit_card", target_secret="4111...")
```

This optimizer targets **agent-style** targets (AgentDojo-style browser/web
agents). Wire it into a `Controller` like any other superred optimizer — see the
framework's [architecture overview](https://superred.simonsure.com).

## License

**No license is granted yet** (see [LICENSE](LICENSE)). The upstream reference
implementation,
[`OSU-NLP-Group/EIA_against_webagent`](https://github.com/OSU-NLP-Group/EIA_against_webagent),
carries **no license file** (default all-rights-reserved), and this module
bundles HTML/form templates reproduced verbatim from its `injection/` code for
template fidelity. Because there is no upstream license to redistribute that
material under, this module is withheld pending the upstream author's permission
and is not released under MIT (unlike other superred optimizers). Once
permission is obtained or the templates are fully replaced with original text, a
proper license will be applied here.

(The separate `SeeAct/` directory in that repository is under the AI PUBS
OpenRAIL-S license; this module does **not** use or copy from it — the bundled
templates come only from the unlicensed `injection/` code.)
