# Judge prompt template attribution

`security_claim_sorry_bench/judge.py` exposes two prompt templates as
Python string constants:

- `BASE_THESCORE_TEMPLATE` — used by the default `LLMAsJudge` (GPT-4 path).
- `BASE_SCOREONLY_TEMPLATE` — alternative format used by other API judges.

Both strings are reproduced verbatim from the upstream MIT-licensed
[SORRY-Bench code repository](https://github.com/SORRY-Bench/SORRY-Bench),
specifically the file
`data/sorry_bench/judge_prompts.jsonl` at commit
`7da10addffb6790cfeb75281eaffb5a176861653` (2025-03-01):

| Local constant | Upstream entry | Line in `judge_prompts.jsonl` |
|---|---|---|
| `BASE_THESCORE_TEMPLATE` | `name: "base-#thescore"` | 2 |
| `BASE_SCOREONLY_TEMPLATE` | `name: "base-scoreonly"` | 1 |

## License

> MIT License
>
> Copyright (c) 2023 LLM-Tuning-Safety
>
> Permission is hereby granted, free of charge, to any person obtaining a
> copy of this software and associated documentation files (the
> "Software"), to deal in the Software without restriction, including
> without limitation the rights to use, copy, modify, merge, publish,
> distribute, sublicense, and/or sell copies of the Software, and to
> permit persons to whom the Software is furnished to do so, subject to
> the following conditions:
>
> The above copyright notice and this permission notice shall be
> included in all copies or substantial portions of the Software.

(See the upstream `LICENSE` file for the full text and disclaimer.)

## Why we vendor only two of the 23 templates

The upstream `judge_prompts.jsonl` ships 23 templates supporting many
judge variants (Llama-Guard, MD-Judge, fine-tuned Mistral, 3-class
per-domain prompts, etc.). For v1 we use only the two relied on by the
canonical GPT-4 judge path; the others remain referenceable via the
upstream repo. If a future task needs them programmatically, we will
vendor the full JSONL with this attribution preserved.
