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
| `render()` in `src/dan_personas_optimizer/personas.py` | `garak/probes/dan.py` — `DANProbeMeta.probe()` |

Not vendored: `ChatGPT_Image_Markdown` (multimodal), `Ablation_Dan_11_0`
(AutoDAN data), `DanInTheWild` (separate corpus). See `ASSUMPTIONS.md`.
