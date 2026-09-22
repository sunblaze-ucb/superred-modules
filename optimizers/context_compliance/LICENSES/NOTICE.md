# NOTICE

This module ports the Context Compliance Attack (CCA) from microsoft/PyRIT
(the Python Risk Identification Toolkit for generative AI), pinned at commit
`2016c4a`. PyRIT is Microsoft's own toolkit and the authoritative implementation
of CCA. Upstream is redistributed under the MIT License; the verbatim text is in
`PyRIT-MIT.txt`.

## Technique

CCA is credited to Mark Russinovich and Ahmed Salem, "Jailbreaking is (Mostly)
Simpler Than You Think", arXiv:2503.05264 (Microsoft),
https://arxiv.org/abs/2503.05264.

## Code (this module)

MIT, see the module `LICENSE`. The CCA control flow -- rendering the two
vendored templates, routing the adversarial question generation and the
simulated-target answer generation through the constrained `self.llm`, and
assembling the fabricated conversation history -- is reimplemented against
superred's async event model in `src/context_compliance_optimizer/`.

## Vendored prompt payloads

Vendored verbatim (sha256-pinned in `_vendor/SHA256SUMS`), never reproduced in
authored code:

- `_vendor/pyrit/datasets/executors/red_teaming/context_compliance/context_compliance.yaml`
- `_vendor/pyrit/datasets/executors/simulated_target/context_compliance_target.yaml`

License: MIT (Copyright (c) Microsoft Corporation).

## Citation

Cite the CCA paper (Russinovich & Salem, arXiv:2503.05264) and PyRIT
(https://github.com/microsoft/PyRIT, pinned commit `2016c4a`) when reporting
numbers produced with this module.
