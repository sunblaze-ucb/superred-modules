"""Trace capture: derive AgentDojo-shaped traces from agent messages.

Mirrors :func:`agentdojo.task_suite.task_suite.functions_stack_trace_from_messages`:
iterates assistant messages and concatenates every entry in each
message's ``tool_calls`` list. The resulting ``list[FunctionCall]`` is
suite-prefix-aware (function names include the ``{suite}__`` prefix); a
helper strips the prefix when handing the trace to upstream's
``*_from_traces`` methods.

Implementation pending.
"""

from __future__ import annotations
