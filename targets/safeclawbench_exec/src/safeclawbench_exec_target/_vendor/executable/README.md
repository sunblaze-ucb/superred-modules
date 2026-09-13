# SafeClawBench-Exec

This directory contains an offline executable/mock-sandbox experiment path for
SafeClawBench. It is intentionally separate from the existing API and harness
runners.

Core pieces:

- `schema.py`: scenario dataclasses with `case_id`, `attack_type`,
  `user_prompt`, `initial_world_state`, `available_tools`, `tool_permissions`,
  `secret_objects`, `expected_safe_behavior`, `success_predicate`,
  `state_oracle`, and `reset`.
- `state.py`: mock world state namespaces for files, email, db, memory, web,
  and code.
- `tools.py`: permissioned mock implementations of `read_file`, `write_file`,
  `delete_file`, `send_email`, `database_query`, `memory_read`,
  `memory_write`, `web_search`, `web_browse`, and `execute_code`.
- `trajectory.py`: stepwise event log for agent messages, tool calls, harness
  blocks, and state deltas.
- `metrics.py`: offline metric computation for Intent-ASR placeholder,
  ToolCall-ASR, StateChange-ASR, HarmCompletion-ASR, Blocked-by-Harness, and
  Recovery Rate.

Run the offline self-test:

```bash
python workspaces/spec/iter_001/exp/safeclawbench/executable/self_test.py
```

Run the tiny subset:

```bash
python workspaces/spec/iter_001/exp/safeclawbench/scripts/run_exec_subset.py --policy unsafe
```
