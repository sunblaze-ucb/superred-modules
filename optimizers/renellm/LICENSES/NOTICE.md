# NOTICE

This module ports ReNeLLM (generalized nested jailbreak: prompt rewriting +
scenario nesting) from NJUNLP/ReNeLLM, pinned at commit `a61c39e`. Upstream is
redistributed under the MIT License; the verbatim text is in `ReNeLLM-MIT.txt`.

## Code (this module)

MIT, see the module `LICENSE`. The ReNeLLM control flow (the rewrite-until-
harmful loop, scenario nesting, refusal short-circuit, and reply scoring) is
reimplemented against superred's async event model in
`src/renellm_optimizer/`. Upstream's SDK completion helper is replaced by a
stdlib shim that routes model calls through superred's constrained LLMClient.

## Vendored payloads

Vendored verbatim (sha256-pinned in `_vendor/SHA256SUMS`), executed but never
reproduced in authored code (torch-free helpers only; `llama/` and
`torch`/`transformers` paths are not bundled):

- `_vendor/renellm/utils/prompt_rewrite_utils.py` (six rewrite operations)
- `_vendor/renellm/utils/scenario_nest_utils.py` (three nesting scenarios)
- `_vendor/renellm/utils/harmful_classification_utils.py` (LLM judge)
- `_vendor/renellm/utils/data_utils.py` (rewrite post-processing helper)

License: MIT (Copyright (c) 2024 NJUNLP).

## Citation

Cite ReNeLLM (Peng Ding, Jun Kuang, Dan Ma, Xuezhi Cao, Yunsen Xian, Jiajun
Chen, Shujian Huang, "A Wolf in Sheep's Clothing: Generalized Nested Jailbreak
Prompts can Fool Large Language Models Easily", NAACL 2024, arXiv:2311.08268)
when reporting numbers produced with this module.
