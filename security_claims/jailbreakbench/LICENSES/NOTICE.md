# Per-file upstream attribution

Third-party material redistributed by `superred-claim-jailbreakbench`, all MIT.

## JailbreakBench (MIT, Copyright (c) 2023 JailbreakBench Team)

- Code: https://github.com/JailbreakBench/jailbreakbench (pinned `23dbdf6b19650521604456229bc1d9c4156c85c1`)
- Data: https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors
- Full data licence text: `JBB-Behaviors-MIT.txt`

| Vendored into (this module) | JailbreakBench source |
| --- | --- |
| `src/jailbreakbench_claim/data/harmful-behaviors.csv` | JBB-Behaviors `data/harmful-behaviors.csv` |
| `USER_PROMPT` in `src/jailbreakbench_claim/judge.py` | `classifier.py` — `Llama3JailbreakJudge.USER_PROMPT` |
| judge parse in `judge.py` | `classifier.py` — `Llama3Classifier` (`POSITIVE_ANSWER in content`) |

Refresh the vendored data/prompt with `python scripts/sync_upstream.py --check`.
