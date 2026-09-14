# NOTICE

superred-target-dtap-claudecode

This module's own code is licensed under the MIT License, Copyright (c) 2026
the superred module authors (see LICENSE).

## Upstream: DecodingTrust-Agent (DTAP)

This target is a faithful reimplementation of the Claude Agent SDK backend of
the DecodingTrust-Agent (DTAP) red-teaming platform. No upstream source files
are vendored; the code independently reproduces the structure and on-disk
schema of the upstream components (ClaudeSDKAgent, ClaudeSDKTraceProcessor,
ClaudeSDKTrajectoryConverter, MCPProxyServer, and
utils.agent_helpers.get_default_disallowed_tools), with superred-specific
adaptations. The only near-verbatim fragment is the short os-filesystem tool
deny list (OS_FILESYSTEM_CLAUDE_SDK_DISALLOWED_TOOLS). Deviations from upstream
are documented in ASSUMPTIONS.md.

- Project: DecodingTrust-Agent (DTAP)
- Source: https://github.com/AI-secure/DecodingTrust-Agent
- Paper: "DecodingTrust-Agent Platform (DTap): A Controllable and Interactive
  Red-Teaming Platform for AI Agents" (arXiv:2605.04808)
- License: Apache-2.0 (Copyright (c) The DecodingTrust-Agent authors, AI-secure)
- Upstream ships no NOTICE file; Apache-2.0 attribution and a statement of
  modifications are provided here and in ASSUMPTIONS.md.

No DTAP benchmark dataset is bundled in this module; only the agent-adapter
code is provided here.

## Build-time (not redistributed in this wheel)

The agent Docker image (docker/Dockerfile) installs, at build time from their
own registries, two Anthropic components that this wheel does NOT contain:

- claude-agent-sdk (PyPI) - MIT, Copyright (c) 2025 Anthropic, PBC.
  https://github.com/anthropics/claude-agent-sdk-python
- @anthropic-ai/claude-code (npm) - Anthropic's Claude Code CLI, governed by
  Anthropic's Commercial Terms of Service, not an open-source license.

These are dependencies of the image the user builds, obtained directly from
PyPI/npm; they are not part of, or redistributed by, this Python package.