"""Claude Code (claude_agent_sdk) DTAP agent target for superred.

Public surface:
- :class:`DtapClaudeCodeTarget` -- the concrete superred ``Target`` (frozen import
  path ``dtap_claudecode_target.DtapClaudeCodeTarget``), subclassing the shared
  ``dtap_scaffold.agent_base.DtapAgentTarget``.
- :func:`convert` -- the host-side transcript -> ``TrajectoryArtifact`` converter.

The Claude Agent SDK and the ``@anthropic-ai/claude-code`` CLI it drives are NOT
host dependencies; they live inside the agent Docker image and are used only by
``driver.py`` (run in that image). This package imports without them.
"""

from __future__ import annotations

from dtap_claudecode_target.target import (
    DEFAULT_IMAGE,
    OS_FILESYSTEM_DISALLOWED_TOOLS,
    DtapClaudeCodeTarget,
)
from dtap_claudecode_target.trajectory import convert

__all__ = [
    "DtapClaudeCodeTarget",
    "OS_FILESYSTEM_DISALLOWED_TOOLS",
    "DEFAULT_IMAGE",
    "convert",
]
