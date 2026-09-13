# Per-file upstream attribution

## XSTest (CC-BY-4.0) — `XSTest-CC-BY-4.0.txt`

Upstream: https://github.com/paul-rottger/xstest — XSTest © Paul Röttger, Hannah
Rose Kirk, Bertie Vidgen, Giuseppe Attanasio, Federico Bianchi, Dirk Hovy, used
under the Creative Commons Attribution 4.0 International license. Paper: Röttger
et al., NAACL 2024 (aclanthology.org/2024.naacl-long.301/). Dataset mirror:
`huggingface.co/datasets/Paul/XSTest`.

Changes were made (CC-BY-4.0 §3(a)(1)(B)): the classifier code was adapted; the
prompt CSV is redistributed unmodified.

| Into (this module) | XSTest source | Change |
| --- | --- | --- |
| `src/xstest_claim/data/xstest_prompts.csv` | `xstest_prompts.csv` | none (byte-for-byte) |
| `src/xstest_claim/data/gpt_judge_prompt.txt` | `evaluation/classify_completions_gpt.py` (the `PROMPT` template) | extracted to a text resource, byte-for-byte |
| `judge.py` — `strmatch_label` | `evaluation/classify_completions_strmatch.py` — `strmatch_label` | ported to a pure function; logic unchanged |
| `judge.py` — `render_gpt_prompt`, `gpt_label` | `evaluation/classify_completions_gpt.py` — `get_completion` | ported to async litellm; bounded retry (see `ASSUMPTIONS.md`) |

The module's own code (scoring policy, `XSTestTask`, factories, loaders, tests)
is original and MIT-licensed (see top-level `LICENSE`). CC-BY-4.0 carries no
ShareAlike/NonCommercial term, so the original additions are MIT while the
upstream material remains under CC-BY-4.0 with the attribution above.

The upstream `model_completions/*.csv` are not vendored (they carry Meta / Mistral
/ OpenAI model-output licenses and are not needed to run the benchmark).
