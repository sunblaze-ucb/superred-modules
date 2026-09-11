# Per-file upstream attribution

## InjecAgent (MIT) — `InjecAgent-MIT.txt`

Upstream: https://github.com/uiuc-kang-lab/InjecAgent (pinned `f19c9f2`),
Copyright (c) 2023 Qiusi Zhan. Verbatim licence also at
`src/injecagent_target/data/INJECAGENT_LICENSE`.

| Into (this module) | InjecAgent source |
| --- | --- |
| `data/attacker_simulated_responses.json` | `data/attacker_simulated_responses.json` |
| `_vendor/agent_prompts.py` (SYS_PROMPT, USER_PROMPT, USER_PROMPT_SIMPLE) | `src/prompts/agent_prompts.py` |
| `_vendor/generation_prompts.py` (ds simulation prompt) | `src/prompts/generation_prompts.py` |
| `agent.py` — `get_tool_dict`, `transform_tool_format_gpt`, prompt/message builders | `src/utils.py`, `src/evaluate_prompted_agent.py`, `src/evaluate_finetuned_agent.py` |
| `simulator.py` — `simulate_attacker_tool_response`, `extract_content` | `src/utils.py` |

Verify vendored data byte-for-byte with `python scripts/sync_upstream.py --check`.

## ToolEmu (Apache-2.0) — `ToolEmu-Apache-2.0.txt`

Upstream: https://github.com/ryoungj/ToolEmu (Ruan et al., 2023). InjecAgent
adapted its tool schemas from ToolEmu, so the vendored `data/tools.json` derives
from ToolEmu and carries its Apache-2.0 licence in addition to InjecAgent's MIT.

| Into (this module) | Origin |
| --- | --- |
| `data/tools.json` (330 tool specs / 38 toolkits) | ToolEmu tool specifications, as curated/adapted by InjecAgent |

Apache-2.0 is permissive and MIT-compatible; this attribution satisfies
Apache-2.0 §4. Only the tool-specification *data* is bundled; no ToolEmu source
code is included.
