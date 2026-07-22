# eia-agent-optimizer

An **EIA (Environmental Injection Attack)** optimizer for
[superred](https://superred.simonsure.com) web-agent targets.

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

From the repository root, with the shared virtual environment active:

```bash
pip install -e ../superred        # the framework first
pip install -e optimizers/eia_agent
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

## Assumptions

Plain-language summary; see [ASSUMPTIONS.md](ASSUMPTIONS.md) for the details.

- **Injects into read content.** The paper-faithful surface is the agent-observed
  webpage/read content: for AgentDojo-style targets the EIA payload is injected
  into `read__*` post-call results (the observation the agent reads), never into
  the call's request arguments.
- **Official HTML preserved.** The form, style, and submit-script templates from
  the released code are kept in `templates.py`, including the `<script>` that
  POSTs to the upstream exfil endpoint (retained only for fidelity — it never
  fires in superred).
- **One attempt by default.** The rendered payload is deterministic for a fixed
  configuration, so re-injecting adds nothing against a deterministic target;
  raise `max_attempts` only for stochastic targets. Placement/subtype sweeps are
  configured once per instance, not per run.
- **Success scoring.** Framework evaluation is authoritative when present.
  Otherwise the local evaluator mirrors the released ASR checks: it requires both
  the grounding marker and a leaked value that matches the target secret, and
  will not claim success on the marker alone. Optional ASR2/stealth checks need a
  supplied reference for the next step.

## License

MIT for this port's code (see [LICENSE](LICENSE)). The upstream reference
implementation,
[`OSU-NLP-Group/EIA_against_webagent`](https://github.com/OSU-NLP-Group/EIA_against_webagent),
carries **no license file** (default all-rights-reserved). The bundled HTML/form
templates are reproduced from that publicly observable repository for template
fidelity; no runtime source is vendored. If upstream publishes a license in the
future, this notice should be revisited.
