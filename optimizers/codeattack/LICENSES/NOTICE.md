# Per-file upstream attribution

## CodeAttack (MIT) — `CodeAttack-MIT.txt`

Upstream: https://github.com/renqibing/CodeAttack (pinned `6777ed03b11567a91908f23bb8ccedca6103772c`)

| Vendored into (this module) | CodeAttack source |
| --- | --- |
| `data/upstream/code_python_list.txt` | `prompt_templates/code_python_list.txt` |
| `data/upstream/code_python_stack.txt` | `prompt_templates/code_python_stack.txt` |
| `data/upstream/code_python_string.txt` | `prompt_templates/code_python_string.txt` |
| `_wrap_input()` / `render()` in `codeattack.py` | `data_preparation.py` — `CodeAttack.wrap` |

Verify the templates and tokenisation with `python scripts/sync_upstream.py --check`.
The C_string / go_string variants are not ported (upstream marks their injection
path TODO). See `ASSUMPTIONS.md`.
