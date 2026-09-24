# NOTICE

This module ports the **Anecdoctor** misinformation-elicitation technique from
Microsoft PyRIT (`pyrit/executor/promptgen/anecdoctor.py`), pinned at commit
`2016c4a`. Upstream is redistributed under the MIT License; the verbatim text is
in `PyRIT-MIT.txt`.

## Code (this module)

MIT, see the module `LICENSE`. The knowledge-graph build, few-shot framing, and
single framed-request emission are reimplemented against superred's async event
model in `src/anecdoctor_optimizer/`.

## Vendored prompt templates

Vendored verbatim (sha256-pinned in `_vendor/SHA256SUMS`), never reproduced in
authored code:

- `_vendor/pyrit/datasets/executors/anecdoctor/anecdoctor_build_knowledge_graph.yaml`
- `_vendor/pyrit/datasets/executors/anecdoctor/anecdoctor_use_fewshot.yaml`
- `_vendor/pyrit/datasets/executors/anecdoctor/anecdoctor_use_knowledge_graph.yaml`

License: MIT (Copyright (c) Microsoft Corporation).

## Citation

Cuevas, Dash, Nayak, Vann, and Daepp, "Anecdoctoring: Automated Red-Teaming
Across Language and Place," EMNLP 2025 (arXiv:2509.19143). Cite this paper and
PyRIT (pinned commit `2016c4a`) when reporting numbers produced with this
module.
