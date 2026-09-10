# Per-file upstream attribution

## CodeAttack (MIT) — `CodeAttack-MIT.txt`

Upstream: https://github.com/renqibing/CodeAttack (pinned `6777ed03b11567a91908f23bb8ccedca6103772c`)

| Vendored into (this module) | CodeAttack source |
| --- | --- |
| `data/upstream/code_python_list.txt` | `src/codeattack/prompt_templates/code_python_list.txt` |
| `data/upstream/code_python_stack.txt` | `src/codeattack/prompt_templates/code_python_stack.txt` |
| `data/upstream/code_python_string.txt` | `src/codeattack/prompt_templates/code_python_string.txt` |
| `data/upstream/code_python_list_plus.txt` | `src/codeattack/prompt_templates/code_python_list_plus.txt` |
| `data/upstream/code_python_stack_plus.txt` | `src/codeattack/prompt_templates/code_python_stack_plus.txt` |
| `data/upstream/code_python_string_plus.txt` | `src/codeattack/prompt_templates/code_python_string_plus.txt` |
| `data/upstream/code_C_string.txt` | `src/codeattack/prompt_templates/code_C_string.txt` |
| `data/upstream/code_go_string.txt` | `src/codeattack/prompt_templates/code_go_string.txt` |
| `_wrap_input()` / `render()` in `codeattack.py` | `data_preparation.py` — `CodeAttack.wrap` |

All eight upstream templates are vendored and ported (see `ASSUMPTIONS.md`
section 3); `scripts/sync_upstream.py --check` re-downloads and diffs all eight.
