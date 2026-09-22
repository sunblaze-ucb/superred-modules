# context_compliance

Context Compliance Attack (CCA) optimizer, ported from
[microsoft/PyRIT](https://github.com/microsoft/PyRIT) (the `context_compliance`
technique), pinned at commit `2016c4a`. PyRIT is Microsoft's own toolkit and the
authoritative implementation of CCA. The technique is credited to Mark
Russinovich and Ahmed Salem, "Jailbreaking is (Mostly) Simpler Than You Think"
([arXiv:2503.05264](https://arxiv.org/abs/2503.05264)).

## What it does

CCA fabricates a benign prior exchange and presents it to the target as if it
had already happened, so the target continues from a position it appears to have
already agreed to. Two adversarial generations build the exchange at setup, both
routed through the constrained `self.llm`:

1. The **adversarial persona** (`context_compliance.yaml`) turns the objective
   into a benign opening user question `Q` about the general topic.
2. The **simulated target** (`context_compliance_target.yaml`) answers `Q` with
   the fabricated affirmative assistant turn `A` -- a brief benign answer
   followed by an offer to carry out the objective.

The fabricated `[user Q][assistant A][user real-request]` history is then
injected, once per run, into a single free-text conversation surface (never the
system prompt). Because the target sees its own fabricated prior agreement, it
tends to comply with the real request that follows.

## Injecting the fabricated history

superred's optimizer -> target channel injects a single string per controllable
event; it has **no primitive for prepending prior conversation turns**. The
fabricated multi-turn history is therefore encoded into one user-message string
as a labelled transcript (`User: ... / Assistant: ... / User: ...`). This is the
one significant deviation from a literal prepended conversation; see
[ASSUMPTIONS.md](ASSUMPTIONS.md).

## Usage

```python
from context_compliance_optimizer import ContextComplianceOptimizer

# Default: the real request that follows the fabricated exchange is the objective.
opt = ContextComplianceOptimizer()

# Reproduce PyRIT's wiring exactly: the fixed "yes." affirmation accepts the
# offer carried by the fabricated assistant turn.
opt = ContextComplianceOptimizer(final_user_message="yes.")
```

## Dependencies

`superred`, `pyyaml`. No heavy dependencies: the two vendored templates use only
plain `{{ variable }}` substitution, rendered by a small stdlib helper, so no
Jinja2 (or any PyRIT) runtime dependency is pulled in.

## Faithfulness

See [ASSUMPTIONS.md](ASSUMPTIONS.md) for every deliberate deviation. The two
CCA templates are vendored byte-identically from PyRIT (sha256-pinned in
`_vendor/SHA256SUMS`, checked by `scripts/sync_upstream.py --check` and the
offline `tests/test_vendor_integrity.py`); no prompt body is reproduced in
authored code.
