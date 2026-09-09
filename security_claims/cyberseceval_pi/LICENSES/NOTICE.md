# Per-file upstream attribution

## Meta PurpleLlama CyberSecEval (MIT) — `PurpleLlama-CybersecurityBenchmarks-MIT.txt`

Upstream: https://github.com/meta-llama/PurpleLlama (pinned `4be64c3a24442b51c76175e6ec67722cc3f5fe38`)

| Vendored into (this module) | CyberSecEval source |
| --- | --- |
| `data/prompt_injection.json` | `CybersecurityBenchmarks/datasets/prompt_injection/prompt_injection.json` |
| `data/prompt_injection_multilingual_machine_translated.json` | same dir, multilingual file |
| `PROMPT_JUDGE_RESPONSE` in `judge.py` | `benchmark/prompt_injection_benchmark.py` — `PROMPT_JUDGE_RESPONSE` |
| judge parse in `judge.py` | `prompt_injection_benchmark.py` — `was_injection_successful` |
| system-prompt + user-input mapping in `task.py` | `benchmark/query_llm.py` — `process_prompt_and_user_input` |

Verify with `python scripts/sync_upstream.py --check`.
