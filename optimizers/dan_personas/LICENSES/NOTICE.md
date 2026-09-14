# Per-file upstream attribution

This directory records the provenance of third-party material redistributed by
`superred-optimizer-dan-personas`. The full text of the upstream license lives
alongside this file (see `garak-Apache-2.0.txt`). The top-level `NOTICE`
carries the same attribution in prose.

## NVIDIA garak (Apache-2.0, Copyright NVIDIA Corporation)

Upstream: https://github.com/NVIDIA/garak — license: `garak-Apache-2.0.txt`

Pinned upstream commit: `2233d51bedf6bfb634872ef9efbcd94eae38cc1a`

Refresh the vendored files with `python scripts/sync_upstream.py`.

| Vendored into (this module) | garak source |
| --- | --- |
| `src/dan_personas_optimizer/data/upstream/AntiDAN.json` | `garak/data/dan/AntiDAN.json` |
| `src/dan_personas_optimizer/data/upstream/ChatGPT_Developer_Mode_RANTI.json` | `garak/data/dan/ChatGPT_Developer_Mode_RANTI.json` |
| `src/dan_personas_optimizer/data/upstream/ChatGPT_Developer_Mode_v2.json` | `garak/data/dan/ChatGPT_Developer_Mode_v2.json` |
| `src/dan_personas_optimizer/data/upstream/ChatGPT_Image_Markdown.json` | `garak/data/dan/ChatGPT_Image_Markdown.json` |
| `src/dan_personas_optimizer/data/upstream/DAN_Jailbreak.json` | `garak/data/dan/DAN_Jailbreak.json` |
| `src/dan_personas_optimizer/data/upstream/DUDE.json` | `garak/data/dan/DUDE.json` |
| `src/dan_personas_optimizer/data/upstream/Dan_10_0.json` | `garak/data/dan/Dan_10_0.json` |
| `src/dan_personas_optimizer/data/upstream/Dan_11_0.json` | `garak/data/dan/Dan_11_0.json` |
| `src/dan_personas_optimizer/data/upstream/Dan_6_0.json` | `garak/data/dan/Dan_6_0.json` |
| `src/dan_personas_optimizer/data/upstream/Dan_6_2.json` | `garak/data/dan/Dan_6_2.json` |
| `src/dan_personas_optimizer/data/upstream/Dan_7_0.json` | `garak/data/dan/Dan_7_0.json` |
| `src/dan_personas_optimizer/data/upstream/Dan_8_0.json` | `garak/data/dan/Dan_8_0.json` |
| `src/dan_personas_optimizer/data/upstream/Dan_9_0.json` | `garak/data/dan/Dan_9_0.json` |
| `src/dan_personas_optimizer/data/upstream/STAN.json` | `garak/data/dan/STAN.json` |
| `src/dan_personas_optimizer/data/upstream/inthewild_jailbreak_llms.json` | `garak/data/inthewild_jailbreak_llms.json` (see dataset origin below) |
| `src/dan_personas_optimizer/data/upstream/ablation_dan_11_0_segments.json` | `garak/probes/dan.py` — `Ablation_Dan_11_0` segments, extracted verbatim from the probe's string literals (see `ASSUMPTIONS.md`) |
| `render()` in `src/dan_personas_optimizer/personas.py` | `garak/probes/dan.py` — `DANProbeMeta.probe()` |

Not vendored: `AutoDAN` / `AutoDANCached` — a genetic search, not a corpus; out
of scope here (superred ships `optimizers/autodan_turbo`). See `ASSUMPTIONS.md`.
(`ChatGPT_Image_Markdown`, `Ablation_Dan_11_0`, and `DanInTheWild` **are** all
vendored — see the table above and the dataset origin below.)

## Dataset origin (one layer up from garak): the in-the-wild jailbreak corpus

`inthewild_jailbreak_llms.json` is not garak's own data. It is the in-the-wild
jailbreak dataset from Shen et al., "'Do Anything Now': Characterizing and
Evaluating In-The-Wild Jailbreak Prompts on Large Language Models"
(arXiv:2308.03825), distributed as **verazuo/jailbreak_llms**
(https://github.com/verazuo/jailbreak_llms) and
**TrustAIRLab/in-the-wild-jailbreak-prompts**, Copyright (c) 2023 Vera, under the
**MIT License** (`jailbreak_llms-MIT.txt`). garak vendors it and credits the paper
via its `dan.py` `doc_uri`; this module vendors garak's byte-identical copy and
loads it as `corpus="in_the_wild"`. Provenance chain: verazuo/jailbreak_llms
(MIT) -> garak -> here.

The 14 DAN persona preambles, by contrast, are community-authored folklore that
garak curates; no distinct licensed upstream exists for them.
